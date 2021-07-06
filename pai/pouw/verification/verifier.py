import binascii
import datetime
import json
import os
import shutil
import struct
import traceback
import uuid

import mxnet as mx
import numpy as np
import yaml

import pai
import redis
from mxnet import gluon, autograd
from pai.pouw import message_map, overdrive
from pai.pouw.constants import TEMP_FOLDER, BUCKET, BLOCK_COMMITMENT_INERATIONS_ANNOUNCED
from pai.pouw.mining.gbtminer import Miner
from pai.pouw.mining.utils import get_batch_hash, file_sha256_hexdigest, file_sha256_digest
from pai.pouw.nodes.decentralized.worker import get_other_workers_local_data, create_network
from pai.pouw.verification import verifier_pb2


# switch endianness
def swap32(i):
    return struct.unpack("<I", struct.pack(">I", i))[0]


# check if messages provided as JSON are ordered in time by tf
def are_messages_ordered(msg_first, msg_second):
    first = datetime.datetime.strptime(msg_first['tf'], '%Y-%m-%d %H:%M:%S.%f')
    second = datetime.datetime.strptime(msg_second['tf'], '%Y-%m-%d %H:%M:%S.%f')
    return first <= second


def verify_block_commitment(conn, msg_id, worker_id, block_header):
    msg_id_components = msg_id.split('_')
    assert len(msg_id_components) == 5 and msg_id_components[0] == 'it' and msg_id_components[1] == 'res'
    it_index = int(msg_id_components[4])

    msg_id_components[4] = '{}'
    template = '_'.join(msg_id_components)
    iterations_counter = BLOCK_COMMITMENT_INERATIONS_ANNOUNCED

    while it_index > 0:
        it_index -= 1
        msg_id = template.format(it_index)
        json_maps = conn.mget(msg_id)
        if json_maps[0] is None:
            continue

        it_data = json.loads(json_maps[0])
        if it_data is None:
            return verifier_pb2.Response.NOT_FOUND, 'Bad previous message.'

        if it_data['signature'] != worker_id:
            continue

        if iterations_counter <= 1:
            break

        iterations_counter -= 1
    else:
        return verifier_pb2.Response.NOT_FOUND, 'Previous i-k message not found.'

    znb_hash = Miner.calculate_zero_nonce_hash(block_header)
    if it_data['znb_hash'] != znb_hash:
        return verifier_pb2.Response.INVALID, 'Block commitment is broken.'

    return None, None


def verify_iteration(msg_history_id, msg_id, nonce, block_header, redis_host='localhost', redis_port=6379):
    print('\nVERIFY ITERATION(msg_history_id = %s, msg_id = %s, nonce = %s)' % (msg_history_id, msg_id, nonce))

    ctx = mx.cpu()
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    json_maps = conn.mget(msg_id)
    iterations_data = [json.loads(data) for data in json_maps if data is not None]
    if len(iterations_data) == 0:
        print('VERIFICATION FAILED -- No message found in Redis.')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Message not found in Redis.")

    it_data = iterations_data[0]
    worker_id = it_data['signature']
    task_id = it_data['task_id']
    batch_location = it_data['batch_hash']
    model_hash = it_data['model_hash']
    learning_rate = float(it_data['e_l'])

    error_code, reason = verify_block_commitment(conn, msg_id, worker_id, block_header)
    if error_code is not None:
        print(f'VERIFICATION FAILED -- {reason}')
        return verifier_pb2.Response(code=error_code,
                                     description=reason)

    model_template = 'models/' + task_id + '/' + worker_id + '/' + model_hash + '/model.params'
    print('Model: %s' % model_template)

    try:
        os.makedirs(TEMP_FOLDER, exist_ok=True)

        model_file_name = str(uuid.uuid4())
        model_location = TEMP_FOLDER + model_file_name

        shutil.copyfile(os.path.join(BUCKET, model_template), model_location)

    except Exception as e:
        print('download_file exception: %s' % e)
        traceback.print_exc()
        raise

    try:
        with open(os.path.join(os.path.dirname(__file__), '../client-task-definition.yaml'), 'r') as request_file:
            request_data = yaml.load(request_file, yaml.UnsafeLoader)

        ver_net = create_network(request_data['ml']['model'])
        ver_net.load_parameters(model_location)
    except Exception as e:
        print('ver_net exception: %s' % e)
        raise

    print('Epoch: %d' % (it_data['epoch']))

    batch_features_template = 'batches/' + batch_location + '/features'
    batch_labels_template = 'batches/' + batch_location + '/labels'
    batch_features_filename = str(uuid.uuid4())
    batch_features_location = TEMP_FOLDER + batch_features_filename
    shutil.copyfile(os.path.join(BUCKET, batch_features_template), batch_features_location)

    batch_labels_filename = str(uuid.uuid4())
    batch_labels_location = TEMP_FOLDER + batch_labels_filename
    shutil.copyfile(os.path.join(BUCKET, batch_labels_template), batch_labels_location)

    # Trainer is for updating parameters with gradient. We use SGD as optimizer.
    trainer = gluon.Trainer(ver_net.collect_params(), 'sgd',
                            {'learning_rate': learning_rate, 'momentum': 0.0})

    # Add metrics: accuracy and cross-entropy loss
    accuracy = mx.metric.Accuracy()
    ce_loss = mx.metric.CrossEntropy()
    comp_metric = mx.metric.CompositeEvalMetric([accuracy, ce_loss])

    loss = gluon.loss.SoftmaxCrossEntropyLoss()
    try:
        data = mx.nd.load(batch_features_location)
        label = mx.nd.load(batch_labels_location)
    except:
        print('VERIFICATION FAILED. Could not load data batch.')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Could not load data batch.")

    data = data[0].as_in_context(ctx)
    label = label[0].as_in_context(ctx)

    comp_metric.reset()

    peer_msg_map = it_data['peer_msg_ids']

    other_workers_data = get_other_workers_local_data(conn, peer_msg_map)

    # perform gradients modifications here
    grads = [g.grad(ctx) for g in ver_net.collect_params().values() if g._grad is not None]

    gradients_blueprint = [g.shape for g in grads]
    gradients_sizes = [g.size for g in grads]
    gradients_cumulative = np.insert(np.cumsum(gradients_sizes), 0, 0)[:-1]
    zero_setter = np.vectorize(lambda int_type: int_type & (~(1 << 31)), otypes=[np.uint32])

    for worker_data in other_workers_data:

        local_map = worker_data['local_deltas']

        deltas = pai.pouw.message_map.decode_message_map(ctx, local_map,
                                                         gradients_blueprint,
                                                         gradients_cumulative,
                                                         worker_data['tau'],
                                                         zero_setter)

        trainer.allreduce_grads()

        # do back propagation
        with autograd.record():
            output = ver_net(data)
            L = loss(output, label)
        L.backward()

        for idx, grad in enumerate(grads):
            grad *= 0
            grad += deltas[idx].as_in_context(ctx)

        # take a gradient step with batch_size equal to data.shape[0]
        trainer.update(data.shape[0])

    trainer.allreduce_grads()
    with autograd.record():
        output = ver_net(data)
        L = loss(output, label)

    # do back propagation
    L.backward()

    initial_local_gradients = overdrive.clone_gradients(grads)

    weight_indices = it_data['local_deltas']
    deltas = message_map.decode_message_map(ctx, weight_indices, gradients_blueprint, gradients_cumulative,
                                            it_data['tau'], zero_setter)

    for idx, grad in enumerate(grads):
        grad *= 0
        grad += deltas[idx].as_in_context(ctx)

    # take a gradient step with batch_size equal to data.shape[0]
    trainer.update(data.shape[0])

    # calculate overdrive
    tau = it_data['tau']
    overdrive_index = overdrive.calculate_overdrive(initial_local_gradients, deltas, tau)

    # re-evaluate network output
    output = ver_net(data)

    # update metrics
    comp_metric.update([label], [output.softmax()])

    names, metrics = comp_metric.get()

    mini_batch_actual = get_batch_hash(data, label)
    mini_batch_provided = it_data['batch_hash']

    model_hash_actual = file_sha256_hexdigest(model_location)

    batches_ok = mini_batch_actual == mini_batch_provided
    print('Mini-batch hashes match: %s -> actual : %s | provided: %s' % (
        'YES' if batches_ok else 'NO',
        mini_batch_actual, mini_batch_provided))

    if not batches_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Batches don't match.")

    models_ok = model_hash_actual == model_hash
    print('Model hashes match: %s -> actual : %s | provided: %s' % (
        'YES' if models_ok else 'NO',
        model_hash_actual, model_hash))

    if not models_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Models don't match.")

    accuracies_ok = metrics[0] == it_data['acc_tr']
    print('Accuracies match: %s -> actual : %s | provided: %s' % (
        'YES' if accuracies_ok else 'NO',
        metrics[0], it_data['acc_tr']))

    if not accuracies_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.INVALID,
                                     description="Invalid accuracy.\nGot {}, expected {}".format(
                                         metrics[0], it_data['acc_tr']))

    loss_ok = metrics[1] == it_data['j_tr']
    print('Loss values match: %s -> actual : %s | provided: %s' % (
        'YES' if loss_ok else 'NO',
        metrics[1], it_data['j_tr']))

    if not loss_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.INVALID,
                                     description="Invalid loss.\nGot {}, expected {}".format(
                                         metrics[1], it_data['j_tr']))

    overdrive_index_ok = overdrive_index == it_data['overdrive']
    print('Overdrive index values match: %s -> actual : %s | provided: %s' % (
        'YES' if overdrive_index_ok else 'NO',
        overdrive_index, it_data['overdrive']))

    if not overdrive_index_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.INVALID,
                                     description="Invalid overdrive index.")

    # verify nonce
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    ver_net.save_parameters(TEMP_FOLDER + 'end_it_model.params')
    end_it_model_hash = file_sha256_digest(TEMP_FOLDER + 'end_it_model.params')
    actual_nonce = Miner.calculate_nonce(end_it_model_hash, weight_indices)
    actual_nonce_int = swap32(int(binascii.hexlify(actual_nonce), 16))
    nonce_ok = actual_nonce_int == nonce
    print('Nonces match: %s -> actual : %s | provided: %s' % (
        'YES' if nonce_ok else 'NO',
        actual_nonce_int, nonce))

    if not nonce_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.INVALID,
                                     description="Nonces don't match.")

    # clean up resources
    if os.path.exists(model_location):
        os.remove(model_location)
    if os.path.exists(batch_features_location):
        os.remove(batch_features_location)
    if os.path.exists(batch_labels_location):
        os.remove(batch_labels_location)

    if os.path.isfile(TEMP_FOLDER + 'end_it_model.params'):
        os.remove(TEMP_FOLDER + 'end_it_model.params')

    print('VERIFICATION SUCCESSFUL')

    return verifier_pb2.Response(code=verifier_pb2.Response.OK,
                                 description="Verification successful.")
