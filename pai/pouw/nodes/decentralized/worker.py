import argparse
import binascii
import datetime
import json
import os.path
import pickle
import shutil
import time
from copy import copy
from distutils.util import strtobool

import mxnet as mx
import numpy as np
import pai.pouw.message_map
import pai.pouw.overdrive
from bitcoinrpc.proxy import JSONRPCException
from mock import MagicMock
from mxnet import gluon, autograd
from mxnet.gluon import nn
from pai.pouw.constants import BUCKET, BLOCK_COMMITMENT_INERATIONS_ANNOUNCED
from pai.pouw.message_map import build_message_map
from pai.pouw.mining.blkmaker.blkmaker import sha256_hexdigest
from pai.pouw.mining.gbtminer import Miner
from pai.pouw.mining.utils import get_batch_hash, save_successful_batch, \
    save_successful_model, delete_saved, file_sha256_hexdigest, file_sha256_digest
from pai.pouw.nodes.decentralized.committee_candidate import CommitteeCandidate


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Worker Node')
    parser.add_argument('--cuda', type=strtobool, default=False,
                        help='Train on GPU with CUDA')

    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='redis host address used for worker synchronization')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port used for connecting to redis database')

    parser.add_argument('--debug', type=strtobool, default=False,
                        help='provide more verbose logging messages')

    parser.add_argument('--server-ip', type=str, default='127.0.0.1',
                        help='PAICoin server IP (default is 127.0.0.1)')
    parser.add_argument('--paicoin-cfg-file', type=str, default=None,
                        help='Filepath to PAICoin configuration file containing rpcport, rpcuser and rpcpassword')
    parser.add_argument('--use-paicoin', type=strtobool, default=True,
                        help='enable/disable usage of paicoin for testing and debugging purposes')

    opt = parser.parse_args()
    # we are manually setting numpy rnd seed in order to insure data spliting gives same results
    # for each worker
    np.random.seed(7)

    # decide the context: GPU or CPU
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()

    # initialize redis connection

    node = WorkerNode(redis_host=opt.redis_host, redis_port=opt.redis_port,
                      context=ctx,
                      is_debug=opt.debug)

    if opt.use_paicoin:
        miner = Miner(BLOCK_COMMITMENT_INERATIONS_ANNOUNCED, opt.server_ip, opt.paicoin_cfg_file)
        node.miner = miner
    else:
        node.miner = MagicMock()
        node.miner.announce_new_block = MagicMock(return_value=b'')
        node.miner.mine_announced_block = MagicMock(return_value=(False, None))

    node.start_task_listening()


def transformer(data, label):
    data = data.reshape((-1,)).astype(np.float32) / 255
    return data, label


def test(ctx, net, validation_samples, validation_labels):
    accuracy = mx.metric.Accuracy()
    ce_loss = mx.metric.CrossEntropy()
    comp_metric = mx.metric.CompositeEvalMetric([accuracy, ce_loss])

    for data, label in zip(validation_samples, validation_labels):
        # Copy data to ctx if necessary
        data = mx.nd.array(data)
        label = mx.nd.array(label)

        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        output = net(data)
        comp_metric.update([label], [output.softmax()])

    return comp_metric.get()


def get_other_workers_local_data(redis_conn, additional_iteration_keys):
    iterations_data = []
    if len(additional_iteration_keys) > 0:
        json_maps = redis_conn.mget(additional_iteration_keys)
        iterations_data = [json.loads(data) for data in json_maps if data is not None]

    # print('Loaded {} iterations data'.format(len(iterations_data)))
    return iterations_data


class WorkerNode(CommitteeCandidate):

    def __init__(self, redis_host, redis_port, context, is_debug=False):
        CommitteeCandidate.__init__(self, redis_host, redis_port, is_debug)
        # lambda convenience methods for message map

        self.zero_setter = np.vectorize(lambda int_type: int_type & (~(1 << 31)),
                                        otypes=[np.uint32])
        self.one_setter = np.vectorize(lambda int_type: int_type | (1 << 31), otypes=[np.uint32])

        self.ctx = context

        self.network_time_to_pick_next_msg = None

        # the hash of the current batch
        self.batch_hash = None

        # the current state of the model
        self.net_hash_start = None
        self.net_hash_end = None

        # set the threshold for residual gradient
        self.tau = None

        # Collect all parameters from net and its children, then initialize them.
        self.net = None
        # self.net.hybridize()  # for performance reasons

        self.trainer = None

        self._peer_msg_ids = []
        self._consumed_peer_msg_ids = []

        # Add metrics: accuracy and cross-entropy loss
        accuracy = mx.metric.Accuracy()
        ce_loss = mx.metric.CrossEntropy()
        self.comp_metric = mx.metric.CompositeEvalMetric([accuracy, ce_loss])

        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()

        # variables needed for indexing
        self.gradients_sizes = None

        # cumulative sum of gradients used in indexing
        self.gradients_cumulative = None

        # gradients blueprint
        self.gradients_blueprint = []
        self.grads = None

        # flag for showing that this is the first batch
        # (some network parameters are lazily initialized, so we need this)
        self.gradient_residual = None

        self.last_updated_iteration_index = 0
        self.global_iteration_index = 0

        self.message_key_template = 'it_res_{}_{}_{}'
        self.node_output_directory = None
        self._model_path = None

        self.miner = None

    def reset_network_state(self):
        # Collect all parameters from net and its children, then initialize them.
        self.net = None

        self.net_hash_start = None
        self.net_hash_end = None

        self._peer_msg_ids = []
        self._consumed_peer_msg_ids = []

        # Add metrics: accuracy and cross-entropy loss
        accuracy = mx.metric.Accuracy()
        ce_loss = mx.metric.CrossEntropy()
        self.comp_metric = mx.metric.CompositeEvalMetric([accuracy, ce_loss])

        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()

        # variables needed for indexing
        self.gradients_sizes = None

        # cumulative sum of gradients used in indexing
        self.gradients_cumulative = None

        # gradients blueprint
        self.gradients_blueprint = []

        # flag for showing that this is the first batch
        # (some network parameters are lazily initialized, so we need this)
        self.gradient_residual = None
        self.grads = None

        self.last_updated_iteration_index = 0
        self.global_iteration_index = 0

        self._model_path = None

        self.tau = None

    def get_global_iteration_index(self):
        return self.conn.incr('{}_global_iteration_index'.format(self.task_id))

    def train(self, training_samples, training_labels, validation_samples, validation_labels):

        metrics = [0, 0, 0]
        # ITERATIONS PHASE
        for epoch in range(self.task_data['ml']['optimizer']['epochs']):
            data = None
            label = None
            self.grads = None

            # track start time of epoch
            epoch_time_started = datetime.datetime.now()

            # self.timedelta = LiveMedian()

            for i, (data, label) in enumerate(zip(training_samples, training_labels)):

                # START OF ITERATION
                self.comp_metric.reset()
                # track start time
                time_started = datetime.datetime.now()

                # Copy data to ctx if necessary
                data = mx.nd.array(data)
                label = mx.nd.array(label)

                data = data.as_in_context(self.ctx)
                label = label.as_in_context(self.ctx)

                # Calculate the mini-batch hash
                self.batch_hash = get_batch_hash(data, label)

                # if it's not the first batch, tell Gluon that we will modify gradients
                if not self.grads:
                    if self.gradient_residual:
                        self.trainer.allreduce_grads()
                    with autograd.record():
                        output = self.net(data)
                        L = self.loss(output, label)

                        # do back propagation
                        L.backward()

                    self.grads = [g.grad(self.ctx) for g in self.net.collect_params().values()
                                  if g._grad is not None]

                    if not self.gradients_sizes:
                        self.gradients_sizes = [g.size for g in self.grads]

                    if self.gradients_cumulative is None:
                        self.gradients_cumulative = np.insert(np.cumsum(self.gradients_sizes), 0, 0)[
                                                    :-1]
                    if self.gradients_blueprint is None or len(self.gradients_blueprint) == 0:
                        self.gradients_blueprint = [g.shape for g in self.grads]

                # save the model initial parameters
                model_path = os.path.join(self.node_output_directory, 'model')
                self.net.save_parameters(model_path + '-0000.params')
                self.net_hash_start = file_sha256_hexdigest(model_path + '-0000.params')

                self.receive_and_apply_peer_gradients(data, epoch, self.grads, label)

                # if it's not the first batch, tell Gluon that we will modify gradients
                if self.gradient_residual:
                    self.trainer.allreduce_grads()

                with autograd.record():
                    output = self.net(data)
                    L = self.loss(output, label)

                    # do back propagation
                    L.backward()

                # collect gradients
                self.grads = [g.grad(self.ctx) for g in self.net.collect_params().values()
                              if g._grad is not None]

                if not self.gradients_sizes:
                    self.gradients_sizes = [g.size for g in self.grads]
                    self.gradients_cumulative = np.insert(np.cumsum(self.gradients_sizes), 0, 0)[
                                                :-1]

                # store the gradients blueprint
                if len(self.gradients_blueprint) == 0:
                    self.gradients_blueprint = [g.shape for g in self.grads]

                # handle gradient residual
                if not self.gradient_residual:
                    # if its first batch
                    self.gradient_residual = [grd.copy() for grd in self.grads]
                else:
                    for idx_grad, grad in enumerate(self.grads):
                        self.gradient_residual[idx_grad] += self.grads[idx_grad]

                initial_local_gradients = pai.pouw.overdrive.clone_gradients(self.grads)

                local_message_map = np.empty([0], dtype=np.uint32)
                local_map = []

                for idx, grad in enumerate(self.grads):
                    # build arrays with ones for values that exceed tau and -tau
                    local_delta_positive = self.gradient_residual[idx] > self.tau
                    local_delta_negative = self.gradient_residual[idx] < -self.tau

                    # build the local weight updates array
                    local_delta = local_delta_positive * self.tau + local_delta_negative * -self.tau

                    # adjust gradient residual accordingly
                    self.gradient_residual[idx] -= local_delta.copy()
                    local_map.append(local_delta.copy())

                    # build message map
                    local_message_map = np.append(local_message_map,
                                                  build_message_map(idx, self.gradients_cumulative,
                                                                    local_delta_positive.asnumpy(),
                                                                    local_delta_negative.asnumpy(),
                                                                    self.zero_setter,
                                                                    self.one_setter))

                # update gradients
                for idx, grad in enumerate(self.grads):
                    grad *= 0
                    grad += local_map[idx].as_in_context(self.ctx)

                # take a gradient step with batch_size equal to data.shape[0]
                self.trainer.update(data.shape[0])

                overdrive_index = pai.pouw.overdrive.calculate_overdrive(initial_local_gradients,
                                                                         local_map,
                                                                         self.tau)

                # update the network to calculate final loss and accuracy
                output = self.net(data)

                # update metrics
                self.comp_metric.update([label], [output.softmax()])

                # report intermediary metrics
                names, metrics = self.comp_metric.get()

                mining_report = ''
                # update redis if there were any updates
                if len(local_message_map) > 0:
                    self.net.save_parameters(model_path + '-0001.params')
                    self.net_hash_end = file_sha256_digest(model_path + '-0001.params')

                    znb_hash = self.miner.announce_new_block()
                    m_key = self.send_gradient_updates(epoch, local_message_map.tolist(),
                                                       metrics, overdrive_index, time_started, znb_hash)

                    # mining takes place after sending the iteration message
                    # first, one must find out how much time is allowed to pick the next message
                    # allowed_time = self.get_allowed_next_message_time()

                    features_filename, features_template = None, None
                    labels_filename, labels_template = None, None
                    model_filename, model_template = None, None
                    bucket_used = None
                    try:
                        block_hex_data, nonce = self.miner.mine_announced_block(m_key.encode('latin1'), b'',
                                                                                self.net_hash_end,
                                                                                local_message_map.tolist())
                        if nonce is not None:
                            mining_report = " nonce=%s" % binascii.hexlify(nonce).decode('latin1')

                        if block_hex_data:
                            features_filename, features_template, labels_filename, labels_template = \
                                save_successful_batch(data, label, self.batch_hash)

                            model_filename, model_template, bucket_used = save_successful_model(
                                os.path.join(self.node_output_directory, 'model'),
                                self.task_id,
                                self.net_hash_start, self.node_id)
                            self.logger.debug('Uploaded {} to bucket {} as key {}'.format(model_filename,
                                                                                          bucket_used,
                                                                                          model_template))
                            self.miner.submit_block(block_hex_data)
                            mining_report += " Mining successful!"
                    except JSONRPCException as e:
                        self.logger.warning('[Epoch %02d Batch %03d] Mining error: %s' % (
                            epoch, i, e.error['message']))
                    except Exception as e:
                        self.logger.warning('[Epoch %02d Batch %03d] Mining error: %s' % (
                            epoch, i, e))

                self.logger.debug(
                    '[Epoch %02d Batch %03d] Training: %s=%f %s=%f' % (
                        epoch, i, names[0], metrics[0],
                        names[1], metrics[1]) + mining_report)

                # END OF ITERATION
                # clean up
                if os.path.isfile(model_path + '-0000.params'):
                    os.remove(model_path + '-0000.params')
                if os.path.isfile(model_path + '-0001.params'):
                    os.remove(model_path + '-0001.params')

            self.end_of_epoch_node_synchronization(data, epoch, self.grads, label)
            self.report_end_of_epoch_data(epoch, epoch_time_started, validation_samples, validation_labels)
            self.logger.info('Completed {} epoch'.format(epoch))
            # mining also takes place after sending the epoch message
            # first, one must find out how much time is allowed to pick the next message
            # allowed_time = self.get_allowed_next_message_time()
            # self.logger.info('[EPOCH %d ] Time to pick next message: %f s' % (epoch, allowed_time))

        model_path = os.path.join(self.node_output_directory, 'model')
        self.logger.info('Started saving model to disk')

        model_path = model_path + '-final.params'
        self.net.save_parameters(model_path)
        self.net_hash_start = file_sha256_hexdigest(model_path)
        self.logger.info('Completed saving model to disk')

        model_key = '{}_{}_{}.params'.format(self.task_id, self.node_id, self.node_id)
        self.logger.info('Started saving model to bucket')

        # here we should save to a path
        shutil.copyfile(model_path, os.path.join(BUCKET, model_key))
        # clean up after model upload
        os.remove(model_path)
        self.logger.info('Uploaded completed model')

        end_result = {
            'worker_signature': self.node_id,
            'j_val': metrics[1],
            'acc_val': metrics[0],
            'worker_id': self.node_id,
            'model_hash': self.net_hash_start,
            'bucket': BUCKET,
            'key': model_key
        }

        self.conn.set('task_done_{}_{}'.format(self.task_id, self.node_id), json.dumps(end_result))

        return end_result

    def report_end_of_epoch_data(self, epoch, epoch_time_started, validation_samples, validation_labels):
        # show epoch metrics
        names, metrics = self.comp_metric.get()
        self.logger.info('[EPOCH %d ] Training: %s=%f %s=%f' % (
            epoch, names[0], metrics[0], names[1], metrics[1]))
        names, metrics = test(self.ctx, self.net, validation_samples, validation_labels)
        self.logger.info('[EPOCH %d ] Validation: %s=%f %s=%f' % (
            epoch, names[0], metrics[0], names[1], metrics[1]))
        # track finish time of epoch
        epoch_time_finished = datetime.datetime.now()

        epoch_metrics = {'miner_id': self.node_id,
                         'cross_entropy': metrics[1],
                         'accuracy': metrics[0]}

        self.conn.rpush('epoch_details_{}_{}'.format(self.task_id, epoch), json.dumps(epoch_metrics))

        if self.is_debug:
            with open(os.path.join(self.node_output_directory,
                                   'epoch_completed_{:03d}.json'.format(epoch)),
                      'w') as outfile:
                results = {'n_version': 0,
                           'task_id': self.task_id,
                           'msg_type': 'EPOCH_COMPLETED',
                           'j_val': metrics[1],
                           'acc_val': metrics[0],
                           'tes': str(epoch_time_started),
                           'tef': str(epoch_time_finished),
                           'epoch': epoch,
                           'signature': self.node_id,
                           'peer_msg_ids': self._peer_msg_ids,
                           'task_data': self.task_data}
                json.dump(results, outfile)

    def send_gradient_updates(self, epoch, weight_indexes, metrics, overdrive_index,
                              time_started, znb_hash):
        self.validate_peer_message_ids()

        serialised_residual = b''
        for g_res in self.gradient_residual:
            serialised_residual += pickle.dumps(g_res.as_in_context(self.ctx))

        serialised_peers = b''
        for peer_msg in self._peer_msg_ids:
            serialised_peers += peer_msg.encode()
            # send local map to redis
        results = {'n_version': 0,
                   'task_id': self.task_id,
                   'msg_type': 'IT_RES',
                   'batch_hash': self.batch_hash,
                   'local_deltas': weight_indexes,
                   'j_tr': metrics[1],
                   'acc_tr': metrics[0],
                   'e_l': self.task_data['ml']['optimizer']['optimizer_initialization_parameters']['learning_rate'],
                   'ts': str(time_started),
                   'tf': str(datetime.datetime.now()),
                   'overdrive': overdrive_index,
                   'epoch': epoch,
                   'tau': self.tau,
                   'znb_hash': znb_hash,
                   'model_hash': self.net_hash_start,
                   'residual_hash': sha256_hexdigest(serialised_residual),
                   'peer_msg_hash': sha256_hexdigest(serialised_peers),
                   'peer_msg_ids': self._peer_msg_ids,
                   'signature': self.node_id, }
        message_key = self.message_key_template.format(self.task_id, epoch,
                                                       self.global_iteration_index)
        results['message_id'] = message_key
        message = json.dumps(results)

        self._consumed_peer_msg_ids.append(message_key)
        self.conn.set(message_key, message)

        if self.is_debug:
            with open(os.path.join(self.node_output_directory,
                                   'it_res.json'),
                      'w') as outfile:
                json.dump(results, outfile)

        return message_key

    def end_of_epoch_node_synchronization(self, data, epoch, grads, label):
        # now we wait for all workers to finish epoch
        epoch_info_key = 'epoch_completed_{}_{}'.format(self.task_id, epoch)
        self.conn.incr(epoch_info_key)
        while True:
            number_of_completed_workers = int(self.conn.get(epoch_info_key))
            are_other_workers_running = number_of_completed_workers < self.get_number_of_registered_nodes()
            if are_other_workers_running:
                time.sleep(1)
            else:
                # we are checking if there were any updates from other workers
                self.receive_and_apply_peer_gradients(data, epoch, grads, label)
                break

    def receive_and_apply_peer_gradients(self, data, epoch, grads, label):
        # we are checking if there were any updates from other workers
        self.global_iteration_index = self.get_global_iteration_index()
        self._peer_msg_ids = []

        if self.last_updated_iteration_index + 1 < self.global_iteration_index:
            # collect other_workers iteration info
            message_index = [x for x in range(self.last_updated_iteration_index + 1,
                                              self.global_iteration_index)]

            additional_iteration_keys = [self.message_key_template.format(self.task_id, epoch,
                                                                          other_index) for
                                         other_index in message_index]

            # here we eliminate already used peer_ids
            additional_iteration_keys = [x for x in additional_iteration_keys
                                         if x not in self._consumed_peer_msg_ids]
            self._consumed_peer_msg_ids.extend(additional_iteration_keys)
            other_workers_data = get_other_workers_local_data(self.conn,
                                                              additional_iteration_keys)
            self._peer_msg_ids = []

            for worker_data in other_workers_data:
                local_map = worker_data['local_deltas']

                deltas = pai.pouw.message_map.decode_message_map(self.ctx, local_map,
                                                                 self.gradients_blueprint,
                                                                 self.gradients_cumulative,
                                                                 worker_data['tau'],
                                                                 self.zero_setter)

                self.trainer.allreduce_grads()
                # do back propagation
                with autograd.record():
                    output = self.net(data)
                    L = self.loss(output, label)
                L.backward()

                for idx, grad in enumerate(grads):
                    grad *= 0
                    grad += deltas[idx].as_in_context(self.ctx)

                # take a gradient step with batch_size equal to data.shape[0]
                self.trainer.update(data.shape[0])
                self._peer_msg_ids.append(worker_data['message_id'])

            self.logger.debug('Applied {} gradient peer updates'.format(len(other_workers_data)))
        self.last_updated_iteration_index = self.global_iteration_index

    def validate_peer_message_ids(self):
        message_ids = [int(message_id.split('_')[-1]) for message_id in self._peer_msg_ids]
        if any(message_id > self.global_iteration_index for message_id in message_ids):
            raise RuntimeError('Collected peer_ids can not be bigger than global index')

    def start_training(self, samples, labels):
        self.tau = self.task_data['ml']['optimizer']['tau']
        validation_size = self.task_data['ml']['validation']['strategy']['size']

        index_split = int(validation_size * len(samples))

        training_samples = samples[:-index_split]
        training_labels = labels[:-index_split]

        validation_samples = samples[-index_split:]
        validation_labels = labels[-index_split:]

        training_result = self.train(training_samples, training_labels, validation_samples, validation_labels)

        self.reset_network_state()
        return training_result

    def initialize_network(self):
        self.net = create_network(self.task_data['ml']['model'])
        initializers = {
            'Xavier': mx.init.Xavier,
            'Bilinear': mx.init.Bilinear,
            'Constant': mx.init.Constant,
            'FusedRNN': mx.init.FusedRNN,
            'LSTMBias': mx.init.LSTMBias,
            'MSRAPrelu': mx.init.MSRAPrelu,
            'Normal': mx.init.Normal,
            'One': mx.init.One,
            'Orthogonal': mx.init.Orthogonal,
            'Uniform': mx.init.Uniform,
            'Zero': mx.init.Zero
        }
        initializer = initializers[self.task_data['ml']['optimizer']['initializer']['name']]
        parameters = self.task_data['ml']['optimizer']['initializer'].get('parameters', {})
        self.net.initialize(initializer(**parameters), ctx=self.ctx)
        # Trainer is for updating parameters with gradient. We use SGD as optimizer.
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd',
                                     self.task_data['ml']['optimizer']['optimizer_initialization_parameters'])


def get_layer_parameters_from_config(layer_parameters):
    layer = copy(layer_parameters)
    del layer['id']
    del layer['type']

    if 'nodes' in layer:
        layer['units'] = layer['nodes']
        del layer['nodes']

    return layer


def create_network(model_data):
    net = nn.Sequential()

    layer_types = {
        'Dense': nn.Dense,
        'Dropout': nn.Dropout,
        'BatchNorm': nn.BatchNorm,
        'InstanceNorm': nn.InstanceNorm,
        'LayerNorm': nn.LayerNorm,
        'Embedding': nn.Embedding,
        'Flatten': nn.Flatten

    }

    with net.name_scope():
        for layer in model_data['hidden-units']:
            layer_class = layer_types[layer['type']]
            layer_parameters = get_layer_parameters_from_config(layer)
            net.add(layer_class(**layer_parameters))
    return net


if __name__ == '__main__':
    main()
