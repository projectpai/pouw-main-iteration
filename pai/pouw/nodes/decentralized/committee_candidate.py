import abc
import argparse
import datetime
import functools
import logging
import math
import os
import pickle
import random
import shutil
import sys
import time
import uuid
from copy import copy
from functools import reduce
from hashlib import sha256
from itertools import cycle

import numpy as np
import redis
import yaml

from pai.pouw.constants import OUTPUT_DIRECTORY, CLIENT_TASK_CHANNEL, \
    MIN_MEMBERS_NUM, WAIT_TIME_AFTER_MINIMAL_NUMBER_OF_NODES_HAS_REGISTERED
from pai.pouw.utils import ColoredConsoleHandler


class CommitteeCandidate:

    def __init__(self, redis_host='localhost', redis_port=6379, is_debug=False):
        self.conn = redis.Redis(host=redis_host, port=redis_port)
        self.conn.ping()

        self.task_data = None
        self._raw_task_data = None

        self.task_id = None

        self.node_output_directory = None

        self.is_debug = False
        self.node_id = str(uuid.uuid4())

        self._task_registration_channel = None
        self._task_moderator_channel = None

        self._client_response_listening_channel = None

        self.segment_hash_table = None

        self._logging_formatter = logging.Formatter(
            '%(levelname)8s - %(asctime)s - %(name)s - %(message)s')

        self._file_handler = None
        logging.basicConfig(
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.logger = logging.getLogger('committee candidate')
        # in future we should propagate only ERROR or higher level logs to stderr
        self.logger.propagate = False
        self.logger.setLevel(logging.DEBUG if is_debug else logging.INFO)

        console_handler = ColoredConsoleHandler(sys.stdout)
        console_handler.setFormatter(self._logging_formatter)
        self.logger.addHandler(console_handler)

    def set_task_id(self, request_data):
        if type(request_data) == str:
            request_data = request_data.encode('utf-8')

        self.task_id = sha256(request_data).hexdigest()
        self._raw_task_data = request_data
        self._task_registration_channel = 'task_registration_{}_{}'.format(self.task_id,
                                                                           self.task_data['client_id'])
        self._task_moderator_channel = 'task_moderator_{}_{}'.format(self.task_id,
                                                                     self.task_data['client_id'])
        self._client_response_listening_channel = 'client_response_channel_{}'.format(self.task_id)

        # we need to create appropriate folders
        current_date = datetime.datetime.now().strftime('%Y_%m_%d')
        worker_output_path = os.path.join(OUTPUT_DIRECTORY, current_date, self.task_id, self.node_id)
        os.makedirs(worker_output_path, exist_ok=True)

        self.node_output_directory = worker_output_path

        if self.is_debug:
            self.set_file_log()

    def set_file_log(self):
        if self._file_handler:
            self.logger.removeHandler(self._file_handler)

        if self.is_debug:
            self._file_handler = logging.FileHandler(os.path.join(self.node_output_directory,
                                                                  'control_node.log'), mode='w')
            self._file_handler.setFormatter(self._logging_formatter)
            self.logger.addHandler(self._file_handler)

    def start_task_listening(self):
        while True:
            self.listen_for_tasks()

    def listen_for_tasks(self):
        # check if there is pending message from client
        self.logger.info('Started listening for tasks')
        while True:
            request = self.get_training_task_request()
            if request is None:
                time.sleep(0.1)
            else:
                break

        self.logger.info('Received client request')
        self.task_data = yaml.load(request, Loader=yaml.UnsafeLoader)
        self.validate_request_data()

        # we need to assign id to current task
        self.set_task_id(request)
        self.set_file_log()

        self.register_for_task()

        self.wait_for_enough_nodes_to_register()
        self.register_for_task(expire=False)
        self.register_moderator()

        if self.is_moderator():
            self.disable_registration_for_client_task()

            self.inform_client_of_task_id()

            self.collect_segment_hash_table()
            self.perform_hash_allocation()
            self.inform_client_of_hash_allocation()

            segments = self.get_training_segments()
            self.distribute_training_segments(segments)

        self.logger.info("Started ml network initialization")
        self.initialize_network()
        samples, labels = self.collect_training_segments()

        self.synchronize_training_nodes()
        self.logger.info("Started training process")
        training_data = self.start_training(samples, labels)
        self.inform_client_of_training_result(training_data)
        # TODO reset node state

    def register_moderator(self):
        self.conn.setnx(self._task_moderator_channel, self.node_id)

    def unregister_moderator(self):
        self.conn.delete(self._task_moderator_channel)

    def is_moderator(self):
        return self.node_id == self.conn.get(self._task_moderator_channel).decode('utf-8')

    def wait_for_enough_nodes_to_register(self,
                                          number=MIN_MEMBERS_NUM,
                                          wait_time=WAIT_TIME_AFTER_MINIMAL_NUMBER_OF_NODES_HAS_REGISTERED):
        # TODO should node try another task if there aren't enough nodes for current task after some time?
        time_elapsed = 0
        self.logger.info('Waiting for enough nodes to register')
        while self.get_number_of_registered_nodes() < number or time_elapsed < wait_time:
            time.sleep(1)
            time_elapsed += 1

    def get_training_task_request(self):
        request = self.conn.lindex(CLIENT_TASK_CHANNEL, -1)
        if request is not None:
            return request.decode("utf-8")

    def validate_request_data(self):
        if type(self.task_data) != dict:
            raise ValueError('Invalid task data structure. It must be dictionary')

        request_root_keys = ['version', 'payment', 'ml']
        for key in request_root_keys:
            if key not in self.task_data:
                raise ValueError('Task data must contain {} root key'.format(key))

        ml_parameter_keys = ['dataset', 'validation', 'optimizer', 'model', 'evaluation-metrics']
        for key in ml_parameter_keys:
            if key not in self.task_data['ml']:
                raise ValueError('Task data must contain machine learning parameters for {}'.format(key))

    def get_number_of_registered_nodes(self):
        return len(self.get_registered_nodes())

    def get_registered_nodes(self):
        nodes = self.conn.keys(f'{self._task_registration_channel}_*')
        if nodes is not None:
            prefix = f'{self._task_registration_channel}_'
            nodes = {node.decode('utf-8')[len(prefix):] for node in nodes}
        else:
            nodes = set()

        if self.node_id not in nodes:
            self.register_for_task()
        return nodes

    def register_for_task(self, expire=True):
        if expire:
            self.conn.set(f'{self._task_registration_channel}_{self.node_id}', '', ex=15)
        else:
            self.conn.set(f'{self._task_registration_channel}_{self.node_id}', '')
        self.logger.info('Successfully registered for client task')

    def get_training_segments(self):
        segment_data = []

        self.logger.info("Started waiting for list of segment data from client")
        test_subset_index = int(
            len(self.segment_hash_table) * self.task_data['ml']['dataset']['test-set-size'])

        while len(segment_data) < len(self.segment_hash_table) - test_subset_index:
            segment_data = self.conn.lrange(self._client_response_listening_channel, 0, -1)
            time.sleep(0.1)

        segment_data = list(map(functools.partial(yaml.load, Loader=yaml.UnsafeLoader), segment_data))
        self.logger.warning('Received list of {} segment data from client'.format(len(segment_data)))
        # here we are checking if client has send segments in same order as votes in hash list
        self.validate_segment_list(segment_data)

        return segment_data

    def perform_hash_allocation(self):
        random.seed(sha256(b"task definition block").digest())
        random.shuffle(self.segment_hash_table)
        random.seed()

    def inform_client_of_task_id(self):
        request = {
            'task_id': self.task_id,
            'client_id': self.task_data['client_id'],
            'cluster_response_address': self._client_response_listening_channel
        }
        request_data = yaml.dump(request)
        self.conn.lpush(self.task_data['client_listen_address'], request_data)
        self.logger.warning('Informed client of task id')

    def collect_segment_hash_table(self):
        while True:
            request = self.conn.lindex(self._client_response_listening_channel, -1)
            if request is None:
                time.sleep(0.1)
            else:
                break

        request_data = yaml.load(request, yaml.UnsafeLoader)
        self.segment_hash_table = request_data['hashes']

    def inform_client_of_hash_allocation(self):
        # removed previous client request
        self.conn.lpop(self._client_response_listening_channel)

        request = {
            'task_id': self.task_id,
            'client_id': self.task_data['client_id'],
            'hashes': self.segment_hash_table
        }
        request_data = yaml.dump(request)
        self.conn.lpush(self.task_data['client_listen_address'], request_data)
        self.logger.warning('Informed client of hash allocation')

    def distribute_training_segments(self, segments):
        selected_workers = list(self.get_registered_nodes())

        self._prepare_segments_for_distribution(segments, selected_workers)

        # above padding algorithm will insure that all nodes get same number of segments
        for index, segment in enumerate(segments):
            worker = selected_workers[index % len(selected_workers)]
            self.assign_segment_to_node(worker, segment)

        self.logger.info("Distributed training segments to workers")

    @staticmethod
    def _prepare_segments_for_distribution(segments, selected_workers):
        # we need to insure that all nodes under the same committee member have same number of segments
        # however nodes under different committee members can have different number of segments

        number_of_nodes_with_more_segments = len(segments) % len(selected_workers)
        padding = segments[-number_of_nodes_with_more_segments:]
        for segment in cycle(padding):
            if len(segments) % len(selected_workers) == 0:
                break
            segments.append(segment)

    def assign_segment_to_node(self, worker_id, segment):
        segment_assignment_channel = self.generate_segment_assignment_channel(worker_id)
        segment_data = yaml.dump(segment)
        self.conn.lpush(segment_assignment_channel, segment_data)

    def generate_segment_assignment_channel(self, worker_id):
        return 'segment_assigment_{}_{}'.format(self.task_id, worker_id)

    def collect_training_segments(self):
        self.logger.info('Started waiting for training data')
        assigned_segments_channel = self.generate_segment_assignment_channel(self.node_id)
        segments = []

        while len(segments) == 0:
            segments = self.conn.lrange(assigned_segments_channel, 0, -1)
            time.sleep(0.1)

        segments = list(map(functools.partial(yaml.load, Loader=yaml.UnsafeLoader), segments))

        samples, labels = [], []

        for segment in segments:
            segment_path = os.path.join(self.node_output_directory, segment['hash'])
            shutil.copyfile(os.path.join(segment['bucket'], segment['key']), segment_path)

            # now we need to validate hash of downloaded segment before continuing
            with open(segment_path, 'rb') as segment_file:
                serialized_data = segment_file.read()

            downloaded_segment_hash = sha256(serialized_data).hexdigest()
            if segment['hash'] != downloaded_segment_hash:
                raise ValueError("Downloaded segment data hash is not matching client send hash")

            sample, label = pickle.loads(serialized_data)

            samples.extend(sample)
            labels.extend(label)

        # now we merge segments
        samples = np.stack(samples)
        labels = np.stack(labels)

        self.logger.info("Collected training data")

        return samples, labels

    @abc.abstractmethod
    def start_training(self, samples, labels):
        pass

    def validate_segment_list(self, segment_list):
        for received_segment, expected_hash in zip(segment_list, self.segment_hash_table):
            if received_segment['hash'] != expected_hash:
                raise ValueError("Segment are not send proper order")

        self.logger.info('Validated received segment list')

    def inform_client_of_training_result(self, training_result):
        training_results_channel = 'training_results_{}_{}'.format(self.task_id,
                                                                   self.task_data['client_id'])
        self.conn.rpush(training_results_channel, yaml.dump(training_result))

        if self.is_moderator():
            results = []

            self.logger.warning('Waiting for all nodes to finish training')
            while len(results) < self.get_number_of_registered_nodes():
                results = self.conn.lrange(training_results_channel, 0, -1)
                time.sleep(0.1)

            results = list(map(functools.partial(yaml.load, Loader=yaml.UnsafeLoader), results))

            self.conn.rpush(self.task_data['client_listen_address'], yaml.dump(results))
            self.unregister_moderator()
            self.logger.warning('Successfully informed client of training results')

    def synchronize_training_nodes(self):
        training_start_key = 'training_start_{}'.format(self.task_id)
        self.conn.incr(training_start_key)

        self.logger.info('Started waiting for other worker to finish pre-training operations')

        while True:
            number_of_completed_workers = int(self.conn.get(training_start_key))
            are_other_workers_running = number_of_completed_workers < self.get_number_of_registered_nodes()

            if are_other_workers_running:
                time.sleep(1)
            else:
                break

        self.logger.info('Initializing training process')

    @abc.abstractmethod
    def initialize_network(self):
        pass

    def disable_registration_for_client_task(self):
        self.conn.lrem(CLIENT_TASK_CHANNEL, -1, self._raw_task_data)
        self.logger.warning("Disabled further registration for this task")


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Worker node for distributed machine learning')
    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='redis host address used for worker synchronization')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port used for connecting to redis database')

    args = parser.parse_args()

    node = CommitteeCandidate(redis_host=args.redis_host, redis_port=args.redis_port)

    node.start_task_listening()


if __name__ == '__main__':
    main()
