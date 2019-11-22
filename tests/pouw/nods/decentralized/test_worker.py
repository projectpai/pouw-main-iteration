from copy import copy

import mxnet as mx
import pytest
from mock import MagicMock
from mxnet.gluon import nn

from pai.pouw.nodes.decentralized.worker import create_network, get_layer_parameters_from_config, WorkerNode


def test_create_network_fc_dnn():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'layer2',
                'type': 'Dense',
                'nodes': 64,
                'activation': 'relu'
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network) == nn.Sequential


@pytest.mark.parametrize('layer_number', range(1, 10))
def test_create_network_hidden_units_number_properly_initialized(layer_number):
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    layer = {
        'id': 'layer',
        'type': 'Dense',
        'nodes': 128,
        'activation': 'relu'
    }

    for index in range(layer_number):
        new_layer = copy(layer)
        new_layer['id'] += str(index)
        model_data['hidden-units'].insert(0, new_layer)

    network = create_network(model_data)
    assert len(network) == layer_number + 1


@pytest.mark.parametrize('node_number', (2 ** n for n in (4, 11)))
def test_create_network_node_number_in_dense_layer(node_number):
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': node_number,
                'activation': 'relu'
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert network[0]._units == node_number


def test_get_layer_parameters_from_config_simple():
    raw_conf = {
        'id': 'layer1',
        'type': 'Dense',
        'nodes': 10,
        'activation': 'relu'
    }

    layer_config = get_layer_parameters_from_config(raw_conf)
    assert layer_config == {
        'units': 10,
        'activation': 'relu'
    }


def test_create_network_dropout_layer():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'dropout1',
                'type': 'Dropout',
                'rate': 0.5,
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network[1]) == nn.Dropout


def test_create_network_batch_normalization_layer():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'batch1',
                'type': 'BatchNorm'
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network[1]) == nn.BatchNorm


def test_create_network_instance_normalization_layer():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'batch1',
                'type': 'InstanceNorm'
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network[1]) == nn.InstanceNorm


def test_create_network_layer_normalization():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'batch1',
                'type': 'LayerNorm'
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network[1]) == nn.LayerNorm


def test_create_network_embedding_layer():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'embedding',
                'type': 'Embedding',
                'input_dim': 64,
                'output_dim': 32
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network[1]) == nn.Embedding


def test_create_network_flatten_layer():
    model_data = {
        'type': 'FC-DNN',
        'hidden-units': [
            {
                'id': 'layer1',
                'type': 'Dense',
                'nodes': 128,
                'activation': 'relu'
            },
            {
                'id': 'embedding',
                'type': 'Flatten',
            },
            {
                'id': 'output',
                'type': 'Dense',
                'nodes': 10
            },
        ],
        'loss': 'SoftmaxCrossEntropyLoss'
    }

    network = create_network(model_data)
    assert type(network[1]) == nn.Flatten


def test_initialize_network_passing_parameters_to_optimizer(client_task_definition_data, mocker):
    mocker.patch('redis.Redis', MagicMock())
    ctx = mx.cpu(0)
    node = WorkerNode(redis_host=None, redis_port=None, context=ctx)
    node.task_data = client_task_definition_data
    node.initialize_network()


@pytest.mark.parametrize('init_settings', [{'name': 'Xavier', 'parameters': {}},
                                           {'name': 'Bilinear', 'parameters': {}},
                                           {'name': 'Constant', 'parameters': {'value': 0}},
                                           {'name': 'FusedRNN',
                                            'parameters': {'init': None, 'num_hidden': 1, 'num_layers': 1,
                                                           'mode': 'test'}},
                                           {'name': 'LSTMBias', 'parameters': {}},
                                           {'name': 'MSRAPrelu', 'parameters': {}},
                                           {'name': 'Normal', 'parameters': {}},
                                           {'name': 'One', 'parameters': {}},
                                           {'name': 'Orthogonal', 'parameters': {}},
                                           {'name': 'Uniform', 'parameters': {}},
                                           {'name': 'Zero', 'parameters': {}}])
def test_initialize_network_passing_parameters_to_optimizer_inicializator(client_task_definition_data, mocker,
                                                                          init_settings):
    mocker.patch('redis.Redis', MagicMock())
    ctx = mx.cpu(0)
    node = WorkerNode(redis_host=None, redis_port=None, context=ctx)
    node.task_data = client_task_definition_data
    node.task_data['ml']['optimizer']['initializer'] = init_settings
    node.initialize_network()


def test_initialize_network_passing_parameters_to_optimizer_no_parameters(client_task_definition_data, mocker):
    mocker.patch('redis.Redis', MagicMock())
    ctx = mx.cpu(0)
    node = WorkerNode(redis_host=None, redis_port=None, context=ctx)
    node.task_data = client_task_definition_data
    del node.task_data['ml']['optimizer']['initializer']['parameters']
    node.initialize_network()
