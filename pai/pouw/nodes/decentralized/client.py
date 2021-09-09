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
import redis
import yaml
from mxnet import gluon
from pai import POUW_ROOT_PATH
from pai.pouw.constants import DATA_DIRECTORY, OUTPUT_DIRECTORY, CLIENT_TASK_CHANNEL, NUMBER_OF_DATASET_SEGMENTS, BUCKET
from pai.pouw.nodes.decentralized.worker import transformer
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

        if self.is_debug:
            self.set_file_log()

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
        if self._cluster_request_data['ml']['dataset']['format'] == 'CSV':
            train_data = self._load_csv_dataset()
        else:
            train_data = self._load_mnist_dataset()

        return train_data

    def _load_mnist_dataset(self):
        train_data = gluon.data.DataLoader(
            gluon.data.vision.MNIST(DATA_DIRECTORY, train=True, transform=transformer),
            batch_size=self._cluster_request_data['ml']['optimizer']['batch-size'],
            shuffle=True, last_batch='discard')
        # TODO merge training and test data
        return train_data

    def _load_csv_dataset(self):
        csv_path = self.get_csv_path()
        # TODO implement paramterization of which column should be transformed into nominal data
        # TODO enable delimiter parametrization
        recorded_values = []

        def str_to_nom(value):
            if value not in recorded_values:
                recorded_values.append(value)
            return float(recorded_values.index(value))

        dataset = np.loadtxt(csv_path,
                             converters={
                                 0: str_to_nom
                             }, delimiter=",", )

        # normalize dataset
        # TODO add normalization to task parametrization
        dataset_normed = dataset / dataset.max(axis=0)

        # we are making assumption that last value is value we need to predict
        # TODO add parametrization of label column
        dataset_normed = gluon.data.ArrayDataset(dataset_normed[:, :-1], dataset_normed[:, -1])

        train_data = gluon.data.DataLoader(
            dataset_normed,
            batch_size=self._cluster_request_data['ml']['optimizer']['batch-size'],
            shuffle=True, last_batch='discard')

        return train_data

    def get_dataset_hashes(self):
        self.logger.info('Started generating dataset segment hashes')
        train_data = self.load_dataset()

        data, label = [], []
        for batch_data, batch_label in train_data:
            data.append(batch_data.asnumpy())
            label.append(batch_label.asnumpy())

        data = np.stack(data)
        label = np.stack(label)
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
            serialized_data = pickle.dumps(segment, protocol=2)
            segment_hash = sha256(serialized_data).hexdigest()

            segment_location = os.path.join(self.worker_output_directory,
                                            '{}.segment'.format(segment_hash))
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
            len(self.segment_hashes) * self._cluster_request_data['ml']['dataset']['test-set-size'])

        self.logger.info('Started sending dataset segments to cluster')
        if test_subset_index:
            selected_hashes = self.segment_hashes[:-test_subset_index]
        else:
            selected_hashes = self.segment_hashes

        for index, segment_hash in enumerate(selected_hashes):
            segment_key = 'segments_{}/{}.segment'.format(self._task_id, segment_hash)

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

        for key in ['dataset', 'validation', 'optimizer', 'model', 'evaluation-metrics']:
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
        best_model_data = max(worker_training_results, key=lambda x: x['acc_val'])
        model_path = os.path.join(self.worker_output_directory, 'model')
        model_path = model_path + '-final.params'

        self.logger.info('Getting best model with accuracy of {}'.format(best_model_data['acc_val']))

        shutil.copyfile(os.path.join(best_model_data['bucket'], best_model_data['key']), model_path)

        # TODO cleanup of old model data
        self.logger.info('Downloaded best model at: {}'.format(model_path))
        for segment_hash in self._dataset_segments:
            if os.path.isfile(self._dataset_segments[segment_hash]):
                os.remove(self._dataset_segments[segment_hash])
        self.logger.info('Cleanup test segment data')

    def get_csv_path(self):
        csv_path = self._cluster_request_data['ml']['dataset']['source']['features']
        if csv_path.startswith('http'):
            # TODO we have to download csv dataset
            pass
        elif csv_path.startswith('/'):
            # path is already absolute
            return csv_path
        else:
            # path is relative
            return os.path.join(POUW_ROOT_PATH, csv_path)


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
