import hashlib
import os
import pickle
import random
from shutil import copyfile

import numpy as np

from pai.pouw.constants import OUTPUT_DIRECTORY, BUCKET

BLOCK_DROP_LOCATION = OUTPUT_DIRECTORY + '/iterations//blocks'


def save_batch(iteration_id, name, data, base_output_folder, task_id, miner_id, batch_hash):
    drop_location = os.path.join(base_output_folder, 'iteration-{}'.format(iteration_id), 'batch-{}'.format(batch_hash))
    os.makedirs(drop_location, exist_ok=True)
    filename = os.path.join(drop_location, name)
    with open(filename, 'wb') as f:
        np.save(f, data.numpy())
    remote_location = os.path.join(BUCKET, 'task-{}'.format(task_id), 'miner-{}'.format(miner_id),
                                   'iteration-{}'.format(iteration_id), 'batch-{}'.format(batch_hash))
    os.makedirs(remote_location, exist_ok=True)
    copyfile(filename, os.path.join(remote_location, name))


def save_successful_batch(iteration_id, data, label, task_id, miner_id, batch_hash, local_drop_location):
    save_batch(iteration_id, 'features', data, local_drop_location, task_id, miner_id, batch_hash)
    save_batch(iteration_id, 'labels', label, local_drop_location, task_id, miner_id, batch_hash)


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
