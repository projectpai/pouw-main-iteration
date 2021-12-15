import hashlib
import os
import pickle
import random
from shutil import copyfile

import numpy as np

from pai.pouw.constants import OUTPUT_DIRECTORY, BUCKET

BATCH_DROP_LOCATION = OUTPUT_DIRECTORY + '/drop/batches/batch-{}'
MODEL_DROP_LOCATION = OUTPUT_DIRECTORY + '/drop/models/task-{}/miner-{}/model-{}'
BLOCK_DROP_LOCATION = OUTPUT_DIRECTORY + '/blocks'


def save_batch(name, data, directory, batch_hash):
    filename = os.path.join(directory, name)
    with open(filename, 'wb') as f:
        np.save(f, data.numpy())
    batch_template = 'batches/' + batch_hash + '/' + name
    os.makedirs(os.path.join(BUCKET, 'batches', 'batch-{}'.format(batch_hash)), exist_ok=True)
    copyfile(filename, os.path.join(BUCKET, batch_template))


def save_successful_batch(data, label, batch_hash):
    directory = BATCH_DROP_LOCATION.format(batch_hash)
    os.makedirs(directory, exist_ok=True)

    save_batch('features', data, directory, batch_hash)
    save_batch('labels', label, directory, batch_hash)


def get_batch_hash(data, label):
    return hashlib.sha256(
        str(np.array2string(data.numpy(), formatter={'float_kind': lambda x: "%.4f" % x}) +
            np.array2string(label.numpy(),
                            formatter={'float_kind': lambda x: "%.4f" % x})).encode('latin1')).hexdigest()


def get_tensors_hash(tensors):
    tensors_concat = np.concatenate([w.numpy().ravel() for w in tensors])
    tensors_hash = hashlib.sha256(pickle.dumps(tensors_concat, protocol=0)).hexdigest()
    return tensors_hash


def get_model_hash(weights):
    model_hash_a = get_tensors_hash(weights)
    return model_hash_a


def serialize_local_message_map(local_weights_list):
    return pickle.dumps(local_weights_list, protocol=0)


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
