import argparse
import binascii
import datetime
import hashlib
import json
import os.path
import shutil
import time
from copy import copy
from distutils.util import strtobool

import numpy as np
import tensorflow as tf
from bitcoinrpc.proxy import JSONRPCException
from mock import MagicMock
from tensorflow import keras
from tensorflow.keras import layers

from pai.pouw.constants import BUCKET, BLOCK_COMMITMENT_INERATIONS_ANNOUNCED
from pai.pouw.mining.blkmaker.blkmaker import sha256_hexdigest
from pai.pouw.mining.gbtminer import Miner
from pai.pouw.mining.utils import save_successful_batch, \
    get_tensors_hash, get_model_hash
from pai.pouw.nodes.decentralized.committee_candidate import CommitteeCandidate
from pai.pouw.nodes.decentralized.message_map import rebuild_delta_local
from pai.pouw.nodes.decentralized.model_shape import get_shape, get_ranges


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

    # initialize redis connection

    node = WorkerNode(redis_host=opt.redis_host, redis_port=opt.redis_port,
                      is_debug=opt.debug)

    if opt.use_paicoin:
        miner = Miner(BLOCK_COMMITMENT_INERATIONS_ANNOUNCED, opt.server_ip, opt.paicoin_cfg_file)
        node.miner = miner
    else:
        node.miner = MagicMock()
        node.miner.announce_new_block = MagicMock(return_value='')
        node.miner.mine_announced_block = MagicMock(return_value=(False, None))
        node.miner.submit_block = MagicMock(return_value=None)

    node.start_task_listening()


def get_other_workers_local_data(redis_conn, additional_iteration_keys):
    iterations_data = []
    if len(additional_iteration_keys) > 0:
        json_maps = redis_conn.mget(additional_iteration_keys)
        iterations_data = [json.loads(data) for data in json_maps if data is not None]

    # print('Loaded {} iterations data'.format(len(iterations_data)))
    return iterations_data


class WorkerNode(CommitteeCandidate):

    def __init__(self, redis_host, redis_port, is_debug=False):
        CommitteeCandidate.__init__(self, redis_host, redis_port, is_debug)

        # the hash of the current batch
        self.batch_hash = None

        # set the threshold for residual gradient
        self.tau = None

        self._peer_msg_ids = []
        self._consumed_peer_msg_ids = []

        self.last_updated_iteration_index = 0
        self.global_iteration_index = 0

        self.message_key_template = 'it_res_{}_{}_{}'
        self.node_output_directory = None

        self.miner = None

        # NEW CODE --------------------------------------------
        self._residual_gradients = None
        self._tau = 10
        # add the default MNIST model

        inputs = keras.Input(shape=(784,), name="digits")
        x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
        x = layers.Dense(64, activation="relu", name="dense_2")(x)
        outputs = layers.Dense(10, name="predictions")(x)
        self._model = keras.Model(inputs=inputs, outputs=outputs)

        # set the default optimizer
        self._optimizer = keras.optimizers.SGD(learning_rate=1e-3)

        # store structure
        self._structure = get_shape(self._model)

        # store ranges for local map
        self._ranges = get_ranges(self._structure)

        # set the loss function
        self._loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # prepare the metrics
        self._train_metric = keras.metrics.SparseCategoricalAccuracy()
        self._val_metric = keras.metrics.SparseCategoricalAccuracy()

        # set proper compilation for model
        self._model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=[self.train_metric])

        # set the batch size
        self._batch_size = 64

        # datasets
        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        self._structure = value

    @property
    def ranges(self):
        return self._ranges

    @ranges.setter
    def ranges(self, value):
        self._ranges = value

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, value):
        self._train_dataset = value

    @property
    def val_dataset(self):
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, value):
        self._val_dataset = value

    @property
    def test_dataset(self):
        return self._test_dataset

    @test_dataset.setter
    def test_dataset(self, value):
        self._test_dataset = value

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property
    def loss_fn(self):
        return self._loss_fn

    @loss_fn.setter
    def loss_fn(self, value):
        self.loss_fn = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def train_metric(self):
        return self._train_metric

    @train_metric.setter
    def train_metric(self, value):
        self._train_metric = value

    @property
    def val_metric(self):
        return self._val_metric

    @val_metric.setter
    def val_metric(self, value):
        self._val_metric = value

    @property
    def residual_gradients(self):
        return self._residual_gradients

    @residual_gradients.setter
    def residual_gradients(self, value):
        self._residual_gradients = value

    @property
    def tau(self):
        return self._tau

    @tau.setter
    def tau(self, value):
        self._tau = value

    @property
    def peer_msg_ids(self):
        return self._peer_msg_ids

    @peer_msg_ids.setter
    def peer_msg_ids(self, value):
        self._peer_msg_ids = value

    def reset_network_state(self):
        # Collect all parameters from net and its children, then initialize them.
        self.peer_msg_ids = []
        self._consumed_peer_msg_ids = []
        self.last_updated_iteration_index = 0
        self.global_iteration_index = 0
        self.tau = None

    def get_global_iteration_index(self):
        return self.conn.incr('{}_global_iteration_index'.format(self.task_id))

    def compute_delta_local_and_update_residuals(self):
        delta_local = []
        map_local = []
        address = 0
        for idx, res_grad in enumerate(self.residual_gradients):
            upper_grads = tf.math.greater_equal(self.residual_gradients[idx], self.tau)
            lower_grads = tf.math.less_equal(self.residual_gradients[idx], -self.tau)
            map_local.extend(address + (
                    int(up_idx[0] if res_grad.shape.ndims == 1 else up_idx[0] * res_grad.shape[1] + up_idx[1]) | (
                    1 << 31)) for up_idx in tf.where(tf.equal(upper_grads, True)))
            map_local.extend(address + (
                    int(lw_idx[0] if res_grad.shape.ndims == 1 else lw_idx[0] * res_grad.shape[1] + lw_idx[1]) & (
                ~(1 << 31))) for lw_idx in tf.where(tf.equal(lower_grads, True)))
            address += res_grad.shape[0] if res_grad.shape.ndims == 1 else res_grad.shape[0] * res_grad.shape[1]
            delta_local_row = tf.cast(upper_grads, tf.float32) * self.tau + tf.cast(lower_grads, tf.float32) * -self.tau
            self.residual_gradients[idx] = tf.math.subtract(self.residual_gradients[idx], delta_local_row)
            delta_local.append(delta_local_row)

        return delta_local, map_local

    def add_new_gradients_to_residuals(self, grads):
        if not self.residual_gradients:
            self.residual_gradients = [tf.identity(grad) for grad in grads]
        else:
            for idx, grad in enumerate(grads):
                self.residual_gradients[idx] = tf.math.add(self.residual_gradients[idx], grad)

    def train(self):
        # ITERATIONS PHASE
        train_acc = 0.0
        for epoch in range(self.task_data['ml']['optimizer']['epochs']):
            for step, (x_batch_train, y_batch_train) in enumerate(self.train_dataset):
                it_ts = datetime.datetime.now()
                # save the model initial parameters
                model_hash_a = get_model_hash(self.model.trainable_weights)
                self.receive_and_apply_peer_gradients(epoch, step)
                self.batch_hash = self.get_batch_hash_for_tensors(x_batch_train, y_batch_train)

                with tf.GradientTape() as tape:
                    logits = self.model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)
                grads = tape.gradient(loss_value, self.model.trainable_weights)

                self.add_new_gradients_to_residuals(grads)
                delta_local, map_local = self.compute_delta_local_and_update_residuals()
                self.optimizer.apply_gradients(zip(delta_local, self.model.trainable_weights))
                self.train_metric.update_state(y_batch_train, logits)

                mining_report = ''
                if len(map_local) > 0:
                    model_hash_b = get_model_hash(self.model.trainable_weights)
                    znb_hash = self.miner.announce_new_block()
                    m_key = self.send_gradient_updates(epoch, map_local, float(loss_value),
                                                       float(self.train_metric.result()), it_ts, znb_hash, model_hash_a)

                    try:
                        block_hex_data, nonce = self.miner.mine_announced_block(m_key.encode('latin1'), b'',
                                                                                model_hash_b,
                                                                                map_local)
                        if nonce is not None:
                            mining_report = " nonce=%s" % binascii.hexlify(nonce).decode('latin1')

                        if block_hex_data:
                            iteration_id = hashlib.sha256((str(epoch) + self.batch_hash + model_hash_a).encode('utf-8')).hexdigest()
                            save_successful_batch(iteration_id,
                                                  x_batch_train,
                                                  y_batch_train,
                                                  self.task_id, self.node_id,
                                                  self.batch_hash, self.node_output_directory)
                            self.save_successful_model(iteration_id, model_hash_a)
                            self.miner.submit_block(block_hex_data)
                            mining_report += " - Mining successful!"
                    except JSONRPCException as e:
                        self.logger.warning('[epoch %03d batch %03d] Mining error: %s' % (
                            epoch, step, e.error['message']))
                    except Exception as e:
                        self.logger.warning('[epoch %02d batch %03d] Mining error: %s' % (
                            epoch, step, e))

                self.logger.debug(
                    'miner %s - [epoch %03d batch %03d] Training: loss= %f accuracy= %.4f' % (
                        self.node_id, epoch, step, float(loss_value),
                        float(self.train_metric.result())) + mining_report)

            train_acc = self.train_metric.result()

            # Reset training metrics at the end of each epoch
            self.train_metric.reset_states()
            self.end_of_epoch_node_synchronization(epoch)
            self.report_end_of_epoch_data(epoch)

        model_hash_final = get_model_hash(self.model.trainable_weights)
        final_model_path = os.path.join(self.node_output_directory, 'model-{}'.format(model_hash_final))
        self.model.save(final_model_path)
        self.logger.info('miner %s - [saved final model to disk]' % self.node_id)
        model_key = 'task-{}/models/miner-{}/model-{}'.format(self.task_id, self.node_id, model_hash_final)
        shutil.copytree(final_model_path, os.path.join(BUCKET, model_key))
        shutil.rmtree(final_model_path, ignore_errors=True)
        self.logger.info('miner %s - [uploaded final model]' % self.node_id)

        end_result = {
            'worker_signature': self.node_id,
            'j_val': float(loss_value),
            'acc_val': float(train_acc),
            'worker_id': self.node_id,
            'model_hash': model_hash_final,
            'bucket': BUCKET,
            'key': model_key
        }

        self.conn.set('task_done_{}_{}'.format(self.task_id, self.node_id), json.dumps(end_result))

        return end_result

    def report_end_of_epoch_data(self, epoch):
        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in self.val_dataset:
            val_logits = self.model(x_batch_val, training=False)
            # Update val metrics
            self.val_metric.update_state(y_batch_val, val_logits)
        val_acc = self.val_metric.result()
        self.val_metric.reset_states()

        self.logger.info(
            'miner %s - [epoch %02d] Validation: %s=%.4f' % (self.node_id, epoch, 'accuracy', float(val_acc)))

        epoch_metrics = {'miner_id': self.node_id,
                         'accuracy': float(val_acc)}

        self.conn.rpush('epoch_details_{}_{}'.format(self.task_id, epoch), json.dumps(epoch_metrics))

    def send_gradient_updates(self, epoch, weight_indexes, loss, accuracy,
                              start_time, znb_hash, initial_model_hash):
        self.validate_peer_message_ids()

        residuals_hash = get_tensors_hash(self.residual_gradients)

        serialised_peers = b''
        for peer_msg in self.peer_msg_ids:
            serialised_peers += peer_msg.encode()

            # send local map to redis
        results = {'n_version': 0,
                   'task_id': self.task_id,
                   'msg_type': 'IT_RES',
                   'batch_hash': self.batch_hash,
                   'local_deltas': weight_indexes,
                   'j_tr': loss,
                   'acc_tr': accuracy,
                   'ts': str(start_time),
                   'tf': str(datetime.datetime.now()),
                   'epoch': epoch,
                   'tau': self.tau,
                   'znb_hash': znb_hash,
                   'model_hash': initial_model_hash,
                   'residual_hash': residuals_hash,
                   'peer_msg_hash': sha256_hexdigest(serialised_peers),
                   'peer_msg_ids': self.peer_msg_ids,
                   'signature': self.node_id, }
        message_key = self.message_key_template.format(self.task_id, epoch,
                                                       self.global_iteration_index)
        results['message_id'] = message_key
        message = json.dumps(results)

        self._consumed_peer_msg_ids.append(message_key)
        self.conn.set(message_key, message)

        return message_key

    def end_of_epoch_node_synchronization(self, epoch):
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
                self.receive_and_apply_peer_gradients(epoch, -1)
                break

    def receive_and_apply_peer_gradients(self, epoch, step):
        # we are checking if there were any updates from other workers
        self.global_iteration_index = self.get_global_iteration_index()
        self.peer_msg_ids = []

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
            self.peer_msg_ids = []

            for worker_data in other_workers_data:
                local_map = worker_data['local_deltas']
                delta_local = rebuild_delta_local(local_map, self.model.trainable_weights, self.tau,
                                                  self.structure, self.ranges)
                self.optimizer.apply_gradients(zip(delta_local, self.model.trainable_weights))
                self.peer_msg_ids.append(worker_data['message_id'])

            self.logger.debug('miner %s - [epoch %03d batch %03d] Peer updates: %d' %
                              (self.node_id, epoch, step, len(other_workers_data)))
        self.last_updated_iteration_index = self.global_iteration_index

    def validate_peer_message_ids(self):
        message_ids = [int(message_id.split('_')[-1]) for message_id in self.peer_msg_ids]
        if any(message_id > self.global_iteration_index for message_id in message_ids):
            raise RuntimeError('Collected peer_ids can not be bigger than global index')

    def start_training(self, samples, labels):
        self.tau = self.task_data['ml']['optimizer']['tau']
        self.batch_size = self.task_data['ml']['optimizer']['batch-size']
        validation_size = self.task_data['ml']['validation']['strategy']['size']

        index_split = int(validation_size * len(samples))

        training_samples = samples[:-index_split]
        training_labels = labels[:-index_split]

        validation_samples = samples[-index_split:]
        validation_labels = labels[-index_split:]

        val_dataset = tf.data.Dataset.from_tensor_slices((validation_samples, validation_labels))
        self.val_dataset = val_dataset.batch(self.batch_size)

        train_dataset = tf.data.Dataset.from_tensor_slices((training_samples, training_labels))
        self.train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        training_result = self.train()

        self.reset_network_state()
        return training_result

    def initialize_network(self):
        self.model = create_network(self.task_data['ml']['model'])
        initializers = {
            'Xavier': tf.keras.initializers.GlorotNormal,
            'Constant': tf.keras.initializers.Constant,
            'Normal': tf.keras.initializers.RandomNormal,
            'One': tf.keras.initializers.Ones,
            'Orthogonal': tf.keras.initializers.Orthogonal,
            'Uniform': tf.keras.initializers.RandomUniform,
            'Zero': tf.keras.initializers.Zeros
        }
        initializer = initializers[self.task_data['ml']['optimizer']['initializer']['name']]
        parameters = self.task_data['ml']['optimizer']['initializer'].get('parameters', {})
        self.model.initialize(initializer(**parameters))

    def save_successful_model(self, iteration_id, model_hash):
        iteration_model_drop_location = os.path.join(self.node_output_directory, 'iteration-{}'.format(iteration_id),
                                                     'model-{}'.format(model_hash))
        os.makedirs(iteration_model_drop_location, exist_ok=True)
        self.model.save(iteration_model_drop_location)
        dest = os.path.join(BUCKET, 'task-{}'.format(self.task_id), 'miner-{}'.format(self.node_id),
                            'iteration-{}'.format(iteration_id), 'model-{}'.format(model_hash))
        if os.path.isdir(dest):
            shutil.copy2(iteration_model_drop_location, dest)
        else:
            shutil.copytree(iteration_model_drop_location, dest, dirs_exist_ok=True)

    @staticmethod
    def get_batch_hash_for_tensors(data, label):
        return hashlib.sha256(
            str(np.array2string(data.numpy(), formatter={'float_kind': lambda x: "%.4f" % x}) +
                np.array2string(label.numpy(),
                                formatter={'float_kind': lambda x: "%.4f" % x})).encode('latin1')).hexdigest()


def get_layer_parameters_from_config(layer_parameters):
    layer = copy(layer_parameters)
    del layer['id']
    del layer['type']

    if 'nodes' in layer:
        layer['units'] = layer['nodes']
        del layer['nodes']

    return layer


def create_network(model_data):
    net = tf.keras.Sequential()

    layer_types = {
        'Dense': tf.keras.layers.Dense,
        'Dropout': tf.keras.layers.Dropout,
        'BatchNorm': tf.keras.layers.BatchNormalization,
        'LayerNorm': tf.keras.layers.LayerNormalization,
        'Embedding': tf.keras.layers.Embedding,
        'Flatten': tf.keras.layers.Flatten

    }

    with net.name_scope():
        for layer in model_data['hidden-units']:
            layer_class = layer_types[layer['type']]
            layer_parameters = get_layer_parameters_from_config(layer)
            net.add(layer_class(**layer_parameters))
    return net


if __name__ == '__main__':
    main()
