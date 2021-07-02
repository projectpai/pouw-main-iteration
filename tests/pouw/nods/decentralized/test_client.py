import functools
import os.path
import uuid
from copy import copy, deepcopy

import mxnet
import pytest
import yaml
from mock import MagicMock

import pai.pouw.nodes.decentralized.client
from pai.pouw.constants import CLIENT_TASK_CHANNEL, NUMBER_OF_DATASET_SEGMENTS
from pai.pouw.nodes.decentralized.client import Client


def test_get_dataset_hashes_generated_number(client_task_definition_path, tmpdir):
    client = Client()

    client._worker_output_directory = str(tmpdir)
    client.load_training_request_data(client_task_definition_path)
    assert (len(client.get_dataset_hashes()) == NUMBER_OF_DATASET_SEGMENTS)


def test_get_dataset_hashes_are_segment_files_generated(client_task_definition_path, tmpdir):
    client = Client()

    client._worker_output_directory = str(tmpdir)
    client.load_training_request_data(client_task_definition_path)
    hash_data = client.get_dataset_hashes()
    for segment_path in hash_data.values():
        assert os.path.isfile(segment_path)


def test_debug_mode_initialization(mocker):
    mocker.patch('pai.pouw.nodes.decentralized.client.Client.set_file_log')
    client = Client(is_debug=True)
    client.set_file_log.assert_called()


def test_network_initialization(mocker):
    mocker.patch('pai.pouw.nodes.decentralized.client.Client.setup_network_communication')
    mocker.patch('pai.pouw.nodes.decentralized.client.Client.set_file_log')
    client = Client(is_debug=True)
    client.setup_network_communication.assert_called_once()


def test_load_training_data_invalid_path(tmpdir):
    task_file = os.path.join(str(tmpdir), 'task.yaml')
    client = Client()

    with pytest.raises(IOError):
        client.load_training_request_data(task_file)


def test_load_training_data_is_client_data_added(client_task_definition_path):
    client = Client()

    client.load_training_request_data(client_task_definition_path)
    assert 'client_id' in client._cluster_request_data
    assert 'client_listen_address' in client._cluster_request_data


@pytest.mark.parametrize('task_data', [list(), set(), 1, 1.0, True, 'test'])
def test_validate_training_request_data_invalid_type(task_data):
    client = Client()
    with pytest.raises(ValueError):
        client.validate_training_request_data(task_data)


def test_validate_training_request_valid_data(client_task_definition_data):
    client = Client()
    client.validate_training_request_data(client_task_definition_data)


def test_validate_training_request_missing_key(client_task_definition_data):
    client = Client()

    del client_task_definition_data['client_id']
    del client_task_definition_data['client_listen_address']

    for key in client_task_definition_data:
        task_data = copy(client_task_definition_data)
        del task_data[key]
        with pytest.raises(ValueError):
            client.validate_training_request_data(task_data)


def test_validate_training_request_missing_key_ml(client_task_definition_data):
    client = Client()

    for key in client_task_definition_data['ml']:
        task_data = deepcopy(client_task_definition_data)
        del task_data['ml'][key]

        with pytest.raises(ValueError):
            client.validate_training_request_data(task_data)


def test_send_initial_training_request(client_task_definition_path, redisdb):
    client = Client()
    client.conn = redisdb
    client.send_initial_training_request(client_task_definition_path)
    assert len(redisdb.lrange(CLIENT_TASK_CHANNEL, 0, -1)) == 1


@pytest.mark.timeout(1)
def test_get_cluster_response_valid(redisdb):
    client = Client()
    client.conn = redisdb

    redisdb.lpush(client._client_listen_address, 'test')

    assert client.get_cluster_response() == 'test'


@pytest.mark.timeout(1)
def test_get_cluster_response_waiting(redisdb, monkeypatch):
    client = Client()
    client.conn = redisdb
    counter = [0]

    def send_response(_):
        counter[0] += 1
        if counter[0] > 10:
            redisdb.lpush(client._client_listen_address, 'test')

    monkeypatch.setattr(pai.pouw.nodes.decentralized.client.time, 'sleep', send_response)

    assert client.get_cluster_response() == 'test'


@pytest.mark.parametrize('cluster_response', [list(), set(), 1, 1.0, True, 'test'])
def test_get_cluster_task_id_invalid_response_type(cluster_response):
    client = Client()
    client.get_cluster_response = MagicMock(return_value=cluster_response)

    with pytest.raises(TypeError):
        client.obtain_cluster_task_id()


@pytest.mark.parametrize('key', ['client_id', 'task_id', 'cluster_response_address'])
def test_get_cluster_task_id_invalid_response_structure(key):
    client = Client()
    response = {
        'client_id': client._client_id,
        'task_id': 'test',
        'cluster_response_address': 'test'
    }

    del response[key]
    client.get_cluster_response = MagicMock(return_value=response)

    with pytest.raises(KeyError):
        client.obtain_cluster_task_id()


def test_get_cluster_task_id_invalid_signature():
    client = Client()
    response = {
        'client_id': 'test',
        'task_id': 'test',
        'cluster_response_address': 'test'
    }

    client.get_cluster_response = MagicMock(return_value=response)

    with pytest.raises(ValueError):
        client.obtain_cluster_task_id()


def test_get_cluster_task_id_attributes_properly_set():
    client = Client()
    response = {
        'client_id': client._client_id,
        'task_id': 'test',
        'cluster_response_address': 'test_adress'
    }

    client.get_cluster_response = MagicMock(return_value=response)
    client.obtain_cluster_task_id()

    assert client._task_id == response['task_id']
    assert client._cluster_address == response['cluster_response_address']


def test_send_dataset_hashes_message_format_testing(redisdb, client_task_definition_data, tmpdir):
    client = Client()
    client.conn = redisdb
    client._cluster_request_data = client_task_definition_data
    client._cluster_address = 'test_cluster_address'
    client._worker_output_directory = str(tmpdir)

    client.send_dataset_hashes()

    message = redisdb.lpop('test_cluster_address')
    message_data = yaml.load(message, yaml.UnsafeLoader)

    assert message_data['client_id'] == client._client_id
    assert len(message_data['hashes']) == NUMBER_OF_DATASET_SEGMENTS


@pytest.mark.parametrize('key', ['client_id', 'hashes'])
def test_obtain_cluster_segment_hash_results_missing_key(key):
    client = Client()
    response = {
        'client_id': client._client_id,
        'hashes': ['test']
    }

    del response[key]
    client.get_cluster_response = MagicMock(return_value=response)

    with pytest.raises(KeyError):
        client.obtain_cluster_segment_hash_results()


def test_obtain_cluster_segment_hash_results_invalid_signature():
    client = Client()
    response = {
        'client_id': 'test',
        'hashes': ['test']
    }

    client.get_cluster_response = MagicMock(return_value=response)

    with pytest.raises(ValueError):
        client.obtain_cluster_segment_hash_results()


def test_obtain_cluster_segment_hash_results_invalid_number_of_hashes():
    client = Client()
    response = {
        'client_id': client._client_id,
        'hashes': ['test']
    }

    client.get_cluster_response = MagicMock(return_value=response)

    with pytest.raises(ValueError):
        client.obtain_cluster_segment_hash_results()


def test_obtain_cluster_segment_hash_results_valid():
    client = Client()
    hash_data = {key: 'test' for key in range(NUMBER_OF_DATASET_SEGMENTS)}
    response = {
        'client_id': client._client_id,
        'hashes': hash_data.keys()
    }

    hash_data = {key: 'test' for key in range(NUMBER_OF_DATASET_SEGMENTS)}
    client._dataset_segments = hash_data

    client.get_cluster_response = MagicMock(return_value=response)

    client.obtain_cluster_segment_hash_results()


def test_send_data_segments_to_cluster_without_cluster_response():
    client = Client()

    with pytest.raises(TypeError):
        client.send_data_segments_to_cluster()


def test_send_data_segments_to_client_cleanup_procedure(mocker, tmpdir, redisdb, client_task_definition_data):
    segment = tmpdir.join('test.segment')
    segment.write('content')

    client = Client()
    client.conn = redisdb
    client._cluster_request_data = client_task_definition_data
    client._cluster_address = str(uuid.uuid4())
    client._dataset_segments = {
        'test': str(segment)
    }

    client.segment_hashes = ['test']

    client.send_data_segments_to_cluster()

    assert not os.path.isfile(str(segment))


def test_send_data_segments_to_client_response_message(mocker, tmpdir, redisdb, client_task_definition_data):
    segment = tmpdir.join('test.segment')
    segment.write('content')

    client = Client()
    client.conn = redisdb
    client._cluster_request_data = client_task_definition_data
    client._cluster_address = str(uuid.uuid4())
    client._dataset_segments = {
        'test': str(segment)
    }

    client.segment_hashes = ['test']

    client.send_data_segments_to_cluster()

    message_raw = redisdb.lrange(client._cluster_address, 0, -1)
    assert message_raw is not None

    messages = list(map(functools.partial(yaml.load, Loader=yaml.UnsafeLoader), message_raw))

    message = messages[0]
    assert all(key in message for key in ['hash', 'bucket', 'key'])


def test_send_data_segments_to_client_response_message_order(mocker, tmpdir, redisdb, client_task_definition_data):
    segment0 = tmpdir.join('test0.segment')
    segment0.write('content')

    segment1 = tmpdir.join('test1.segment')
    segment1.write('content')

    segment2 = tmpdir.join('test2.segment')
    segment2.write('content')

    client = Client()
    client.conn = redisdb
    client._cluster_request_data = client_task_definition_data
    client._cluster_address = str(uuid.uuid4())
    client._dataset_segments = {
        'test0': str(segment0),
        'test1': str(segment1),
        'test2': str(segment2)
    }

    client.segment_hashes = ['test1', 'test2', 'test0']

    client.send_data_segments_to_cluster()

    message_raw = redisdb.lrange(client._cluster_address, 0, -1)
    assert message_raw is not None

    messages = list(map(functools.partial(yaml.load, Loader=yaml.UnsafeLoader), message_raw))

    for key, message in zip(client.segment_hashes, messages):
        assert message['hash'] == key


def test_send_data_segments_to_client_response_test_subset_withheld(mocker, tmpdir, redisdb,
                                                                    client_task_definition_data):
    segment0 = tmpdir.join('test0.segment')
    segment0.write('content')

    segment1 = tmpdir.join('test1.segment')
    segment1.write('content')

    segment2 = tmpdir.join('test2.segment')
    segment2.write('content')

    segment3 = tmpdir.join('test3.segment')
    segment3.write('content')

    client = Client()
    client.conn = redisdb
    client._cluster_request_data = client_task_definition_data
    client._cluster_address = str(uuid.uuid4())
    client._dataset_segments = {
        'test0': str(segment0),
        'test1': str(segment1),
        'test2': str(segment2),
        'test3': str(segment3)
    }

    client.segment_hashes = ['test1', 'test2', 'test0', 'test3']

    client.send_data_segments_to_cluster()

    message_raw = redisdb.lrange(client._cluster_address, 0, -1)
    assert message_raw is not None

    messages = list(map(functools.partial(yaml.load, Loader=yaml.UnsafeLoader), message_raw))

    for key, message in zip(client.segment_hashes[:-1], messages):
        assert message['hash'] == key


def test_load_dataset_is_mnist_called(client_task_definition_data):
    client = Client()
    client._load_mnist_dataset = MagicMock()

    client._cluster_request_data = client_task_definition_data
    client.load_dataset()

    client._load_mnist_dataset.assert_called_once()


def test_load_dataset_is_csv_called(client_task_definition_data):
    client = Client()
    client._load_mnist_dataset = MagicMock()
    client._load_csv_dataset = MagicMock()

    client._cluster_request_data = client_task_definition_data
    client._cluster_request_data['ml']['dataset']['format'] = 'CSV'

    client.load_dataset()
    client._load_csv_dataset.assert_called_once()


def test_load_csv_dataset_from_hardrive(client_task_definition_csv_data):
    client = Client()
    client._load_mnist_dataset = MagicMock()

    client._cluster_request_data = client_task_definition_csv_data

    training_data = client._load_csv_dataset()
    assert type(training_data) == mxnet.gluon.data.DataLoader
    for feature_batch, label_batch in training_data:
        assert len(feature_batch) == 100


def test_get_csv_path_test_relative_path(client_task_definition_csv_data, csv_dataset_path):
    client = Client()
    client._cluster_request_data = client_task_definition_csv_data
    client._cluster_request_data['ml']['dataset']['source']['features'] = 'tests/data/abalone.csv'
    assert client.get_csv_path() == csv_dataset_path


def test_get_csv_path_test_absolute_path(client_task_definition_csv_data, csv_dataset_path):
    client = Client()
    client._cluster_request_data = client_task_definition_csv_data

    assert client.get_csv_path() == csv_dataset_path
