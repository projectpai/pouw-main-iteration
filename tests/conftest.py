#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Dummy conftest.py for pouw.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    https://pytest.org/latest/plugins.html
"""
from __future__ import print_function, absolute_import, division

import os.path
import uuid

import pytest
import yaml


@pytest.fixture
def client_task_definition_path():
    current_dir = os.path.dirname(__file__)
    yield os.path.join(current_dir, 'data', 'client-task-definition.yaml')


@pytest.fixture
def client_task_definition_data(client_task_definition_path):
    with open(client_task_definition_path) as task_file:
        task_data = yaml.load(task_file, yaml.UnsafeLoader)

    task_data['client_id'] = str(uuid.uuid4())
    task_data['client_listen_address'] = str(uuid.uuid4())
    return task_data


@pytest.fixture
def csv_dataset_path():
    current_dir = os.path.dirname(__file__)
    yield os.path.join(current_dir, 'data', 'abalone.csv')


@pytest.fixture
def client_task_definition_csv_data(client_task_definition_data, csv_dataset_path):
    client_task_definition_data['ml']['dataset']['format'] = 'CSV'
    client_task_definition_data['ml']['dataset']['source'] = {
        'features': csv_dataset_path,
        'labels': csv_dataset_path
    }
    yield client_task_definition_data
