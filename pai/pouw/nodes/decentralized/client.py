import argparse
import datetime
import json
import logging
import os.path
import pickle
import shutil
import sys
import time
import uuid
from hashlib import sha256
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
import redis
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from tensorflow import keras

from pai import POUW_ROOT_PATH, DATASETS_DIR
from pai.pouw.constants import OUTPUT_DIRECTORY, CLIENT_TASK_CHANNEL, NUMBER_OF_DATASET_SEGMENTS, BUCKET
from pai.pouw.utils import ColoredConsoleHandler


class Client:
    # in this version we are using redis as a placeholder that will be replaced with blockchain in later versions

    def __init__(self, redis_host='localhost', redis_port=6379, is_debug=False):
        self.conn = None

        self.setup_network_communication(redis_host, redis_port)

        self._client_id = str(uuid.uuid4())
        self._client_listen_address = 'pouw_client_listen_{}'.format(self._client_id)

        self._cluster_request_data = None
        self._cluster_address = None

        self._task_id = None

        self._dataset_segments = None
        self.segment_hashes = None
        self.is_debug = is_debug

        self._logging_formatter = logging.Formatter(
            '%(levelname)8s - %(asctime)s - %(name)s - %(message)s')

        self._file_handler = None
        logging.basicConfig(
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger('client_node')
        # in future we should propagate only ERROR or higher level logs to stderr
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

        console_handler = ColoredConsoleHandler(sys.stdout)
        console_handler.setFormatter(self._logging_formatter)
        self.logger.addHandler(console_handler)

        current_date = datetime.datetime.now().strftime('%Y_%m_%d')
        worker_output_path = os.path.join(OUTPUT_DIRECTORY, current_date, 'client')

        self._worker_output_directory = worker_output_path

        self._batch_size = 64

        if self.is_debug:
            self.set_file_log()

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def worker_output_directory(self):
        os.makedirs(self._worker_output_directory, exist_ok=True)

        return self._worker_output_directory

    def set_file_log(self):
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)

        if self.is_debug:
            self._file_handler = RotatingFileHandler(os.path.join(self.worker_output_directory,
                                                                  '{}_client_node.log'.format(
                                                                      self._client_id)),
                                                     mode='a', maxBytes=10 * 1024 * 1024, backupCount=4
                                                     )
            self._file_handler.setFormatter(self._logging_formatter)
            self.logger.addHandler(self._file_handler)

    def setup_network_communication(self, host, port):
        self.conn = redis.Redis(host=host, port=port)
        self.conn.ping()

    def run_cluster_training(self, client_task_file_path):
        self.send_initial_training_request(client_task_file_path)

        # we are waiting on client listen address, until cluster send us assigned task_id
        # after which we are sending it 10 hashes of dataset segments
        self.obtain_cluster_task_id()
        self.send_dataset_hashes()

        self.obtain_cluster_segment_hash_results()
        self.send_data_segments_to_cluster()
        self.get_trained_models()

    def send_initial_training_request(self, client_task_file_path):
        self.load_training_request_data(client_task_file_path)
        request_data = yaml.dump(self._cluster_request_data)
        self.conn.lpush(CLIENT_TASK_CHANNEL, request_data)
        self.logger.info('Finished sending task request data')

    def load_training_request_data(self, client_task_file_path):
        self.logger.info('Started loading client task definition file')
        with open(client_task_file_path, 'r') as request_file:
            request_data = yaml.load(request_file, Loader=yaml.UnsafeLoader)

        self.validate_training_request_data(request_data)
        request_data['client_id'] = self._client_id
        request_data['client_listen_address'] = self._client_listen_address
        current_time = str(datetime.datetime.now().timestamp())
        request_data['created'] = current_time
        self._cluster_request_data = request_data
        self.logger.info('Completed loading of client task definition file')

    def send_request_data_to_network(self, request_data):
        self.conn.lpush(self._cluster_address, request_data)

    def obtain_cluster_task_id(self):
        # in current version we can set that cluster task id is determined by using sha256 of send message
        # in fully decentralized version this will change
        response = self.get_cluster_response()
        if type(response) != dict:
            raise TypeError('Expected cluster response to be dic, got {}'.format(response))

        if response['client_id'] != self._client_id:
            raise ValueError('Client id is not matching in cluster response')

        self._task_id = response['task_id']
        self._cluster_address = response['cluster_response_address']
        self.logger.info('Collected task id data from cluster')
        self.conn.set('task_submitted_{}_{}'.format(self._task_id, self._client_id),
                      json.dumps(self._cluster_request_data))
        self.logger.info('Added task information in database')

    def get_cluster_response(self):
        self.logger.info('Started waiting for cluster response')
        response_data = None
        while response_data is None:
            response_data = self.conn.rpop(self._client_listen_address)
            time.sleep(0.1)

        self.logger.info('Collected cluster response')
        return yaml.load(response_data, yaml.UnsafeLoader)

    def send_dataset_hashes(self):
        self._dataset_segments = self.get_dataset_hashes()
        request_data = {'hashes': list(self._dataset_segments.keys()),
                        'client_id': self._client_id}
        serialized_request_data = yaml.dump(request_data)
        self.send_request_data_to_network(serialized_request_data)
        self.logger.info('Send dataset hashes for allocating')

    def load_dataset(self):
        self.batch_size = int(self._cluster_request_data['ml']['batch-size'])
        if self._cluster_request_data['ml']['dataset']['format'] == 'CSV':
            train_data = self._load_csv_dataset()
        elif self._cluster_request_data['ml']['dataset']['format'] == 'MNIST':
            train_data = self._load_mnist_dataset()
        else:
            raise Exception('Unknown data format')

        return train_data

    def _load_mnist_dataset(self):
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = np.reshape(x_train, (-1, 784))
        x_test = np.reshape(x_test, (-1, 784))

        # throw away the last unfinished batch if there is one
        x_train = x_train[0: (x_train.shape[0] // self.batch_size) * self.batch_size]
        # x_train = x_train.reshape(x_train.shape[0] // self.batch_size, -1, x_train.shape[1])
        y_train = y_train[0: (y_train.shape[0] // self.batch_size) * self.batch_size]
        # y_train = y_train.reshape(y_train.shape[0] // self.batch_size, -1)

        return np.vstack([x_train, x_test]), np.append(y_train, y_test)

    def get_data_exceptions(self):
        return self._cluster_request_data['ml']['dataset']['remove-features']

    def get_labels(self):
        return self._cluster_request_data['ml']['dataset']['labels']

    def get_features_scaler(self):
        return self.get_scaler(self._cluster_request_data['ml']['dataset']['features-scaler'])


    def get_labels_scaler(self):
        return self.get_scaler(self._cluster_request_data['ml']['dataset']['labels-scaler'])

    def get_scaler(self, scaler):
        scaler_class = None
        if scaler is not None:
            if scaler == 'RobustScaler':
                scaler_class = RobustScaler()
            elif scaler == 'StandardScaler':
                scaler_class = StandardScaler()
            else:
                raise Exception("Unknown scaler")
        return scaler_class

    def _load_csv_dataset(self):
        file_name = self.get_csv_path()
        x_all = pd.read_csv(file_name)

        data_exceptions = self.get_data_exceptions()
        labels = self.get_labels()

        y_all = x_all[labels]
        data_exceptions.extend(labels)
        for data_ex in data_exceptions:
            del x_all[data_ex]

        np_features = x_all.to_numpy()
        features_scaler = self.get_features_scaler()
        features_scaler.fit(np_features)
        np_features = features_scaler.transform(np_features)

        np_labels = y_all.to_numpy()
        labels_scaler = self.get_labels_scaler()
        if labels_scaler is not None:
            labels_scaler.fit(np_labels)
            np_labels = labels_scaler.transform(np_labels)
        x_train, x_test, y_train, y_test = train_test_split(np_features, np_labels, test_size=0.3, random_state=42)

        x_train = np.reshape(x_train, (-1, x_train.shape[1]))
        y_train = np.reshape(y_train, (-1, y_train.shape[1]))

        x_train = x_train[0: (x_train.shape[0] // self.batch_size) * self.batch_size]
        y_train = y_train[0: (y_train.shape[0] // self.batch_size) * self.batch_size]

        return np.vstack([x_train, x_test]), np.vstack([y_train, y_test])

    def get_dataset_hashes(self):
        self.logger.info('Started generating dataset segment hashes')
        data, label = self.load_dataset()

        # split batches into 10 groups and discard the rest
        max_index = int(data.shape[0] / NUMBER_OF_DATASET_SEGMENTS) * NUMBER_OF_DATASET_SEGMENTS
        step = max_index // NUMBER_OF_DATASET_SEGMENTS

        data = data[:max_index]
        label = label[:max_index]

        dataset = dict()

        for index in range(0, max_index, step):
            segment_index_end = index + step
            segment_data = data[index:segment_index_end]
            segment_label = label[index:segment_index_end]

            segment = (segment_data, segment_label)
            serialized_data = pickle.dumps(segment, protocol=0)
            segment_hash = sha256(serialized_data).hexdigest()

            segment_location = os.path.join(self.worker_output_directory,
                                            'segment-{}'.format(segment_hash))
            with open(segment_location, 'wb') as segment_file:
                segment_file.write(serialized_data)

            dataset[segment_hash] = segment_location

        self.logger.info('Completed generation of dataset hashes')
        return dataset

    def obtain_cluster_segment_hash_results(self):
        response = self.get_cluster_response()
        if response['client_id'] != self._client_id:
            raise ValueError('Client id is not matching in cluster response')

        if len(response['hashes']) != NUMBER_OF_DATASET_SEGMENTS:
            raise ValueError('Number of hashes is not matching number of segments')

        self.segment_hashes = response['hashes']
        self.logger.info('Collected segment hashes from cluster')

    def send_data_segments_to_cluster(self):
        test_subset_index = int(
            len(self.segment_hashes) * (1.00 - self._cluster_request_data['ml']['dataset']['train-size']))

        self.logger.info('Started sending dataset segments to cluster')
        if test_subset_index:
            selected_hashes = self.segment_hashes[:-test_subset_index]
        else:
            selected_hashes = self.segment_hashes

        for index, segment_hash in enumerate(selected_hashes):
            segment_key = 'task-{}/segments/segment-{}'.format(self._task_id, segment_hash)

            os.makedirs(os.path.dirname(os.path.join(BUCKET, segment_key)), exist_ok=True)
            shutil.copyfile(self._dataset_segments[segment_hash], os.path.join(BUCKET, segment_key))

            segment_data = {
                'hash': segment_hash,
                'bucket': BUCKET,
                'key': segment_key
            }

            self.conn.rpush(self._cluster_address, yaml.dump(segment_data))

            # clean up segment_data
            os.remove(self._dataset_segments[segment_hash])
            self.logger.info(
                'Send {} out of {} segments'.format(index + 1, len(self.segment_hashes[:-test_subset_index])))

        self.logger.info('Completed sending dataset segments to cluster')

    def validate_training_request_data(self, task_data):
        if type(task_data) != dict:
            raise ValueError("Task definition must be dictionary")

        for key in ['version', 'payment', 'ml']:
            if key not in task_data:
                raise ValueError('Task definition must contain {} key'.format(key))

        for key in ['dataset', 'validation', 'optimizer', 'model', 'metrics', 'loss', 'epochs', 'tau', 'batch-size']:
            if key not in task_data['ml']:
                raise ValueError('ML Task definition configuration must contain {} key'.format(key))

        self.logger.info('Completed validation of client task definition')

    def get_trained_models(self):
        self.logger.info('Started waiting for training results response from cluster')
        response_data = None

        while response_data is None:
            response_data = self.conn.rpop(self._client_listen_address)
            time.sleep(0.1)

        self.logger.info('Collected training results from cluster')
        training_results = yaml.load(response_data, yaml.UnsafeLoader)
        self.get_best_model(training_results)

    def get_best_model(self, worker_training_results):
        best_model_data = max(worker_training_results, key=lambda x: x['metric_val'])
        best_model_path_on_client = os.path.join(self.worker_output_directory, 'models', 'model-{}'.format(best_model_data['model_hash']))

        self.logger.info('Getting best model with best metric value of {}'.format(best_model_data['metric_val']))

        shutil.copytree(os.path.join(best_model_data['bucket'], best_model_data['key']), best_model_path_on_client)

        # TODO cleanup of old model data
        self.logger.info('Downloaded best model at: {}'.format(best_model_path_on_client))
        for segment_hash in self._dataset_segments:
            if os.path.isfile(self._dataset_segments[segment_hash]):
                os.remove(self._dataset_segments[segment_hash])
        self.logger.info('Cleanup test segment data')

    def get_csv_path(self):
        return os.path.join(POUW_ROOT_PATH, DATASETS_DIR, self._cluster_request_data['ml']['dataset']['source'])


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Client for initializing training tasks')
    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='redis host address used for worker synchronization')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port used for connecting to redis database')
    parser.add_argument('--client-task-definition-path', type=str,
                        help='filepath towards file containing initial client task')
    parser.add_argument('--use-continuous-training', default=False, action='store_true',
                        help='Enable running of continuous training on client')

    args = parser.parse_args()

    client = Client(redis_host=args.redis_host, redis_port=args.redis_port)
    if args.use_continuous_training:
        while True:
            client._client_id = str(uuid.uuid4())
            client._client_listen_address = 'pouw_client_listen_{}'.format(client._client_id)
            client.run_cluster_training(args.client_task_definition_path)
    else:
        client.run_cluster_training(args.client_task_definition_path)


if __name__ == '__main__':
    main()
