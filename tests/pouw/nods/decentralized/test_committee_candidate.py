import random
import time
import uuid

import mock
import pytest
import yaml
from mock import MagicMock, PropertyMock

import pai.pouw.nodes.decentralized.committee_candidate
from pai.pouw.constants import CLIENT_TASK_CHANNEL, WAIT_TIME_AFTER_MINIMAL_NUMBER_OF_NODES_HAS_REGISTERED, \
    MIN_MEMBERS_NUM, NUMBER_OF_DATASET_SEGMENTS
from pai.pouw.nodes.decentralized.client import Client
from pai.pouw.nodes.decentralized.committee_candidate import CommitteeCandidate


def test_get_training_task_request_non_destructive(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb

    redisdb.lpush(CLIENT_TASK_CHANNEL, 'test1')

    assert node.get_training_task_request() == 'test1'
    assert redisdb.llen(CLIENT_TASK_CHANNEL) == 1


def test_get_training_task_request_fifo_behaviour(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb

    redisdb.lpush(CLIENT_TASK_CHANNEL, 'test1')
    redisdb.lpush(CLIENT_TASK_CHANNEL, 'test2')
    redisdb.lpush(CLIENT_TASK_CHANNEL, 'test3')

    assert node.get_training_task_request() == 'test1'
    assert redisdb.llen(CLIENT_TASK_CHANNEL) == 3


def test_validate_request_data_simple(client_task_definition_data):
    node = CommitteeCandidate()
    node.task_data = client_task_definition_data

    node.validate_request_data()


@pytest.mark.parametrize('key', ['version', 'payment', 'ml'])
def test_validate_request_data_missing_root_key(key, client_task_definition_data):
    node = CommitteeCandidate()
    node.task_data = client_task_definition_data

    del node.task_data[key]

    with pytest.raises(ValueError):
        node.validate_request_data()


@pytest.mark.parametrize('key', ['dataset', 'validation', 'optimizer', 'model', 'evaluation-metrics'])
def test_validate_request_data_missing_ml_parameter_key(key, client_task_definition_data):
    node = CommitteeCandidate()
    node.task_data = client_task_definition_data

    del node.task_data['ml'][key]

    with pytest.raises(ValueError):
        node.validate_request_data()


@pytest.mark.parametrize('data', [list(), set(), 'test', 0, False])
def test_validate_request_data_invalid_type(data):
    node = CommitteeCandidate()
    node.task_data = data

    with pytest.raises(ValueError):
        node.validate_request_data()


def test_set_task_id_provides_consistent_hash(client_task_definition_path, client_task_definition_data):
    with open(client_task_definition_path) as request_file:
        request_data = request_file.read()

    node = CommitteeCandidate()
    node.task_data = client_task_definition_data
    node.set_task_id(request_data)

    task_id = node.task_id

    for _ in range(10):
        node.set_task_id(request_data)
        assert task_id == node.task_id


def test_register_for_task_first_node_registration(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb

    registration_channel = 'test_registration'

    node._task_registration_channel = registration_channel
    node.register_for_task()
    assert node.get_number_of_registered_nodes() == 1


def test_register_for_task_same_node_not_counted_twice(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb

    registration_channel = 'test_registration'

    node._task_registration_channel = registration_channel
    node.register_for_task()
    node.register_for_task()
    assert node.get_number_of_registered_nodes() == 1


def test_register_for_task_multiple_nodes(redisdb):
    registration_channel = 'test_registration'

    node_1 = CommitteeCandidate()
    node_1.conn = redisdb

    node_1._task_registration_channel = registration_channel
    node_1.register_for_task()

    node_2 = CommitteeCandidate()
    node_2.conn = redisdb

    node_2._task_registration_channel = registration_channel
    node_2.register_for_task()

    assert node_1.get_number_of_registered_nodes() == 2


def test_get_number_of_registered_nodes(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb

    registration_channel = 'test_registration'

    node._task_registration_channel = registration_channel
    node.register_for_task()
    assert node.get_number_of_registered_nodes() == 1


def test_get_number_of_registered_nodes_same_node_not_counted_twice(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb

    registration_channel = 'test_registration'

    node._task_registration_channel = registration_channel
    node.register_for_task()
    node.register_for_task()
    assert node.get_number_of_registered_nodes() == 1


def test_get_number_of_registered_nodes_multiple(redisdb):
    registration_channel = 'test_registration'

    node_1 = CommitteeCandidate()
    node_1.conn = redisdb

    node_1._task_registration_channel = registration_channel
    node_1.register_for_task()

    node_2 = CommitteeCandidate()
    node_2.conn = redisdb

    node_2._task_registration_channel = registration_channel
    node_2.register_for_task()

    assert node_1.get_number_of_registered_nodes() == 2
    assert node_2.get_number_of_registered_nodes() == 2


def test_wait_for_enough_nodes_to_register_grace_period_waited(mocker):
    mocker.patch('pai.pouw.nodes.decentralized.committee_candidate.time.sleep')
    node = CommitteeCandidate()
    node.get_number_of_registered_nodes = MagicMock(return_value=MIN_MEMBERS_NUM)

    node.wait_for_enough_nodes_to_register()

    assert pai.pouw.nodes.decentralized.committee_candidate.time.sleep.call_count == WAIT_TIME_AFTER_MINIMAL_NUMBER_OF_NODES_HAS_REGISTERED


def test_inform_client_of_task_id(client_task_definition_data):
    node = CommitteeCandidate()
    node.task_data = client_task_definition_data

    node.conn = MagicMock()

    node.inform_client_of_task_id()

    node.conn.lpush.assert_called()


def test_inform_client_of_task_integrated(client_task_definition_data, redisdb):
    client_address = 'test_client'

    node = CommitteeCandidate()
    node.conn = redisdb

    node.task_id = '123'

    node.task_data = client_task_definition_data
    node.task_data['client_listen_address'] = client_address
    node._client_response_listening_channel = 'test_cluster'

    node.inform_client_of_task_id()

    client = Client()
    client._client_id = client_task_definition_data['client_id']
    client.conn = redisdb
    client._client_listen_address = client_address

    client.obtain_cluster_task_id()
    assert client._task_id == '123'
    assert client._cluster_address == 'test_cluster'


def test_collect_segment_hash_table_simple(redisdb):
    node = CommitteeCandidate()
    node.conn = redisdb
    cluster_listen_address = 'test_cluster'
    node._client_response_listening_channel = cluster_listen_address

    client_request = {
        'client_id': '123',
        'hashes': range(NUMBER_OF_DATASET_SEGMENTS)
    }

    redisdb.lpush(cluster_listen_address, yaml.dump(client_request))

    node.collect_segment_hash_table()
    assert len(node.segment_hash_table) == NUMBER_OF_DATASET_SEGMENTS


def test_inform_client_of_hash_voting_results(client_task_definition_data):
    node = CommitteeCandidate()
    node.task_data = client_task_definition_data

    node.conn = MagicMock()

    node.inform_client_of_hash_allocation()

    node.conn.lpush.assert_called()


def test_validate_segment_list_simple():
    node = CommitteeCandidate()

    segment_list = [{'hash': i, 'bucket': '', 'key': ''} for i in range(5)]
    node.segment_hash_table = range(5)

    node.validate_segment_list(segment_list)


def test_validate_segment_list_reverse_order():
    node = CommitteeCandidate()

    segment_list = [{'hash': i, 'bucket': '', 'key': ''} for i in range(5)]
    node.segment_hash_table = range(5)[::-1]

    with pytest.raises(ValueError):
        node.validate_segment_list(segment_list)


def test_prepare_segments_for_distribution():
    node = CommitteeCandidate()

    segments = list(range(5))
    selected_workers = list(range(4))

    node._prepare_segments_for_distribution(segments, selected_workers)
    assert len(segments) == 8
    assert segments == [0, 1, 2, 3, 4, 4, 4, 4]


def test_disable_registration_for_client_task():
    node = CommitteeCandidate()
    node.conn = MagicMock()

    node.disable_registration_for_client_task()
    node.conn.lrem.assert_called()


@pytest.mark.skip('Different api from real redis library')
def test_disable_registration_for_client_task_only_one_task_in_queue(redisdb, client_task_definition_path,
                                                                     client_task_definition_data):
    with open(client_task_definition_path) as request_file:
        request_data = request_file.read()

    node = CommitteeCandidate()
    node.conn = redisdb
    node.task_data = client_task_definition_data
    node.set_task_id(request_data)

    redisdb.lpush(CLIENT_TASK_CHANNEL, request_data)
    node.disable_registration_for_client_task()

    assert redisdb.llen(CLIENT_TASK_CHANNEL) == 0


@pytest.mark.skip('Different api from real redis library')
def test_disable_registration_for_client_task_multiple_tasks_in_queue(redisdb, client_task_definition_path,
                                                                      client_task_definition_data):
    with open(client_task_definition_path) as request_file:
        request_data = request_file.read()

    node = CommitteeCandidate()
    node.conn = redisdb

    node.task_data = client_task_definition_data
    node.set_task_id(request_data)

    redisdb.lpush(CLIENT_TASK_CHANNEL, request_data)
    redisdb.lpush(CLIENT_TASK_CHANNEL, request_data)
    node.disable_registration_for_client_task()

    assert redisdb.llen(CLIENT_TASK_CHANNEL) == 1


def test_committee_gets_disolved_before_completing_training(redisdb):
    registration_channel = 'test_registration'

    node_1 = CommitteeCandidate()
    node_1.conn = redisdb

    node_1._task_registration_channel = registration_channel
    node_1.register_for_task()

    node_2 = CommitteeCandidate()
    node_2.conn = redisdb

    node_2._task_registration_channel = registration_channel
    node_2.register_for_task()

    assert node_1.get_number_of_registered_nodes() == 2
    assert node_2.get_number_of_registered_nodes() == 2

    time.sleep(20)

    assert len(node_1.get_registered_nodes()) == 0
