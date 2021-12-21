import binascii
import datetime
import hashlib
import json
import os
import shutil
import struct
import traceback
import uuid

import numpy as np
import redis
import tensorflow as tf

from pai.pouw.constants import TEMP_FOLDER, BUCKET, BLOCK_COMMITMENT_INERATIONS_ANNOUNCED
from pai.pouw.mining.gbtminer import Miner
from pai.pouw.mining.utils import get_model_hash
from pai.pouw.nodes.decentralized.message_map import rebuild_delta_local
from pai.pouw.nodes.decentralized.model_shape import get_shape, get_ranges
from pai.pouw.nodes.decentralized.worker import get_other_workers_local_data
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


def get_batch_hash_for_numpy(data, label):
    return hashlib.sha256(
        str(np.array2string(data, formatter={'float_kind': lambda x: "%.4f" % x}) +
            np.array2string(label,
                            formatter={'float_kind': lambda x: "%.4f" % x})).encode('latin1')).hexdigest()


def verify_iteration(msg_history_id, msg_id, nonce, block_header, redis_host='localhost', redis_port=6379):
    print('\nVERIFY ITERATION(msg_history_id = %s, msg_id = %s, nonce = %s)' % (msg_history_id, msg_id, nonce))

    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    json_maps = conn.mget(msg_id)
    iterations_data = [json.loads(data) for data in json_maps if data is not None]
    if len(iterations_data) == 0:
        print('VERIFICATION FAILED -- No message found in Redis.')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Message not found in Redis.")

    it_data = iterations_data[0]
    miner_id = it_data['signature']
    task_id = it_data['task_id']
    batch_hash = it_data['batch_hash']
    model_hash = it_data['model_hash']
    epoch = it_data['epoch']

    # TO DO: re-enable this once finished with the other stuff
    # error_code, reason = verify_block_commitment(conn, msg_id, worker_id, block_header)
    # if error_code is not None:
    #     print(f'VERIFICATION FAILED -- {reason}')
    #     return verifier_pb2.Response(code=error_code,
    #                                  description=reason)

    iteration_location = 'task-{}/miner-{}/iteration-{}/'.format(task_id, miner_id, hashlib.sha256(
        (str(epoch) + batch_hash + model_hash).encode('utf-8')).hexdigest())
    model_location = iteration_location + 'model-{}'.format(model_hash)
    print('Model: %s' % model_location)

    work_dir = str(uuid.uuid4())
    try:
        os.makedirs(os.path.join(TEMP_FOLDER, work_dir), exist_ok=True)
        local_model_location = os.path.join(TEMP_FOLDER, work_dir, 'model')
        shutil.copytree(os.path.join(BUCKET, model_location), local_model_location)

    except Exception as e:
        print('Download model files exception: %s' % e)
        traceback.print_exc()
        raise

    print('Epoch: %d' % (it_data['epoch']))

    # make locals copies / downloads
    batch_location = iteration_location + 'batch-{}'.format(batch_hash)
    features_file = os.path.join(TEMP_FOLDER, work_dir, 'features')
    labels_files = os.path.join(TEMP_FOLDER, work_dir, 'labels')
    shutil.copyfile(os.path.join(BUCKET, batch_location, 'features'), features_file)
    shutil.copyfile(os.path.join(BUCKET, batch_location, 'labels'), labels_files)

    np_features = np.load(features_file)
    np_labels = np.load(labels_files)

    mini_batch_actual = get_batch_hash_for_numpy(np_features, np_labels)
    mini_batch_provided = it_data['batch_hash']

    batches_ok = mini_batch_actual == mini_batch_provided
    print('Mini-batch hashes match: %s -> actual : %s | provided: %s' % (
        'YES' if batches_ok else 'NO',
        mini_batch_actual, mini_batch_provided))

    if not batches_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Batches don't match.")

    # Load model
    model = tf.keras.models.load_model(local_model_location)
    model_hash_actual = get_model_hash(model.trainable_weights)

    peer_msg_map = it_data['peer_msg_ids']
    other_workers_data = get_other_workers_local_data(conn, peer_msg_map)
    models_ok = model_hash_actual == model_hash
    print('Model hashes match: %s -> actual : %s | provided: %s' % (
        'YES' if models_ok else 'NO',
        model_hash_actual, model_hash))

    if not models_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.NOT_FOUND,
                                     description="Models don't match.")

    structure = get_shape(model)
    ranges = get_ranges(structure)

    local_map = []
    tau = float(it_data['tau'])
    optimizer = model.optimizer
    loss_fn = model.loss

    for worker_data in other_workers_data:
        peer_map = worker_data['local_deltas']
        delta_peer = rebuild_delta_local(peer_map, model.trainable_weights, tau,
                                         structure, ranges)
        optimizer.apply_gradients(zip(delta_peer, model.trainable_weights))

    for x_batch_train, y_batch_train in tf.data.Dataset.from_tensor_slices((np_features, np_labels)).batch(100):
        with tf.GradientTape() as tape:
            logits = model(x_batch_train, training=True)
            loss_value = loss_fn(y_batch_train, logits)
        local_map = it_data['local_deltas']
        delta_local = rebuild_delta_local(local_map, model.trainable_weights, tau,
                                          structure, ranges)
        optimizer.apply_gradients(zip(delta_local, model.trainable_weights))

    loss_ok = float(loss_value) == it_data['j_tr']
    print('Loss values match: %s -> actual : %s | provided: %s' % (
        'YES' if loss_ok else 'NO',
        float(loss_value), it_data['j_tr']))

    if not loss_ok:
        print('VERIFICATION FAILED')
        return verifier_pb2.Response(code=verifier_pb2.Response.INVALID,
                                     description="Invalid loss.\nGot {}, expected {}".format(
                                         float(loss_value), it_data['j_tr']))

    # verify nonce
    os.makedirs(TEMP_FOLDER, exist_ok=True)

    actual_nonce = Miner.calculate_nonce(get_model_hash(model.trainable_weights), local_map)
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
    shutil.rmtree(os.path.join(TEMP_FOLDER, work_dir), ignore_errors=True)
    print('VERIFICATION SUCCESSFUL')

    return verifier_pb2.Response(code=verifier_pb2.Response.OK,
                                 description="Verification successful.")


if __name__ == '__main__':
    response = verify_iteration(0, 'it_res_73fdc50323552e564cb031186747f258d463a230bfd5793eb6737f1165fa8eda_0_384', '',
                                '')
