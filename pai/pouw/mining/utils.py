import hashlib
import json
import os
import pickle
import random
from shutil import copyfile

import mxnet as mx
import numpy as np
from mxnet import ndarray
from pai.pouw.constants import OUTPUT_DIRECTORY, BUCKET

BATCH_DROP_LOCATION = OUTPUT_DIRECTORY + '/drop/batches/{}'
MODEL_DROP_LOCATION = OUTPUT_DIRECTORY + '/drop/models/{}/{}/{}'
BLOCK_DROP_LOCATION = OUTPUT_DIRECTORY + '/blocks'


def reduce_params(net, sym):
    arg_names = set(sym.list_arguments())
    aux_names = set(sym.list_auxiliary_states())
    arg_dict = {}
    for name, param in net.collect_params().items():
        if name in arg_names:
            arg_dict['arg:%s' % name] = param._reduce()
        else:
            assert name in aux_names
            arg_dict['aux:%s' % name] = param._reduce()

    return arg_dict


def save_model(net, path, sym=None, arg_dict=None):
    if not net._cached_graph:
        raise RuntimeError(
            "Please first call block.hybridize() and then run forward with "
            "this block at least once before calling export.")
    if sym is None:
        sym = net._cached_graph[1]
    sym.save('%s.json' % path)

    if arg_dict is None:
        arg_dict = reduce_params(net, sym)
    ndarray.save('%s.params' % path, arg_dict)


def save_successful_model(initial_path, task_id, model_hash, worker_id):
    directory = MODEL_DROP_LOCATION.format(task_id, worker_id, model_hash)
    os.makedirs(directory, exist_ok=True)

    filename = initial_path + '-0000.params'
    copyfile(filename, os.path.join(directory, 'model.params'))
    model_template = 'models/' + task_id + '/' + worker_id + '/' + model_hash + '/model.params'
    os.makedirs(os.path.join(BUCKET, 'models', task_id, worker_id, model_hash), exist_ok=True)
    copyfile(filename, os.path.join(BUCKET, model_template))
    return filename, model_template, BUCKET


def save_batch(name, data, directory, batch_hash):
    filename = os.path.join(directory, name)
    mx.nd.save(filename, data)
    batch_template = 'batches/' + batch_hash + '/' + name
    os.makedirs(os.path.join(BUCKET, 'batches', batch_hash), exist_ok=True)
    copyfile(filename, os.path.join(BUCKET, batch_template))
    return filename, batch_template


def save_successful_batch(data, label, batch_hash):
    directory = BATCH_DROP_LOCATION.format(batch_hash)
    os.makedirs(directory, exist_ok=True)

    features_filename, features_template = save_batch('features', data, directory, batch_hash)
    labels_filename, labels_template = save_batch('labels', label, directory, batch_hash)

    return features_filename, features_template, labels_filename, labels_template


def save_successful_block(nonce, msg_id, model_hash, msg_next_id, task_id, batch_id, worker_id):
    directory = BLOCK_DROP_LOCATION
    os.makedirs(directory, exist_ok=True)

    with open(os.path.join(directory, task_id + '_' + worker_id + '_' + batch_id + '.json'), 'w') as outfile:
        new_block = {
            "nonce": nonce,
            "msg_id": msg_id,
            "model_hash": model_hash,
            "msg_next_id": msg_next_id
        }
        json.dump(new_block, outfile)


def delete_saved(filename, key, bucket_used=None):
    os.remove(filename)

    if bucket_used is not None:
        os.remove(os.path.join(bucket_used, key))


def get_batch_hash(data, label):
    return hashlib.sha256(
        str(np.array2string(data.asnumpy(), formatter={'float_kind': lambda x: "%.4f" % x}) +
            np.array2string(label.asnumpy(),
                            formatter={'float_kind': lambda x: "%.4f" % x})).encode('latin1')).hexdigest()


def file_sha256(file_path):
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        # Read and update hash string value in blocks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash


def file_sha256_hexdigest(file_path):
    return file_sha256(file_path).hexdigest()


def file_sha256_digest(file_path):
    return file_sha256(file_path).digest()


def serialize_local_message_map(local_weights_list):
    return pickle.dumps(local_weights_list)


def get_nonce(msg, model_params_file_path, msg_next):
    msg_hash = hashlib.sha256(msg).hexdigest()
    model_params_hash = file_sha256_hexdigest(model_params_file_path)
    msg_next_hash = hashlib.sha256(msg_next).hexdigest()
    return hashlib.sha256(msg_hash + model_params_hash + msg_next_hash).hexdigest()


def nonce_successful():
    return random.randint(0, 20) < 2  # momentarily do not allow this


def load_rpc_password(password_file_path):
    if password_file_path is None:
        raise RuntimeError('Path toward paicoin.conf must be provided for node to work')

    with open(password_file_path) as password_file:
        for line in password_file:
            if line.startswith('rpcpassword='):
                password = line.split('=')[1].strip()
                return password
