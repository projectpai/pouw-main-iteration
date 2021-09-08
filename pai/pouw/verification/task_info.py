import json

import redis

from pai.pouw.verification.task_info_pb2 import TaskListResponse, TaskRecord, Pagination, HTTPReturnCode
from google.protobuf.timestamp_pb2 import Timestamp


def get_waiting_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    training_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('training_start_*')]))
    result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('task_submitted_*')
              if task.decode("utf-8").split('_')[2] not in training_tasks]

    return prepare_tasks_for_grpc(result, page, per_page)


def get_started_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    done_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('task_done_*')]))

    result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('training_start_*')
              if task.decode("utf-8").split('_')[2] not in done_tasks]

    return prepare_tasks_for_grpc(result, page, per_page)


def get_completed_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    result = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('task_done_*')]))
    return prepare_tasks_for_grpc(result, page, per_page)


def get_task_info(task_id, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    details = None
    for full_task in conn.scan_iter('task_submitted_{}_*'.format(task_id)):
        json_task_details = conn.mget(full_task.decode('utf-8'))
        if json_task_details is not None and json_task_details[0] is not None:
            details = json.loads(json_task_details[0].decode('utf-8'))
        break

    return details


def prepare_tasks_for_grpc(task_ids, page=1, per_page=20, redis_host='localhost', redis_port=6379):
    task_list = []
    timestamp = Timestamp()
    for task_id in task_ids:
        task_details = get_task_info(task_id, redis_host, redis_port)
        timestamp.GetCurrentTime()
        if task_details is not None:
            task_record = TaskRecord(task_id=task_id, model_type=task_details['ml']['model']['type'],
                                     nodes_no=sum([hu['nodes'] for hu in task_details['ml']['model']['hidden-units']]),
                                     batch_size=task_details['ml']['optimizer']['batch-size'],
                                     optimizer=task_details['ml']['optimizer']['type'],
                                     created=timestamp)
        else:
            task_record = TaskRecord(task_id='', model_type='',
                                     nodes_no=0,
                                     batch_size=0,
                                     optimizer='',
                                     created=timestamp)
        task_list.append(task_record)

    # adjust for corner cases
    task_list_len = len(task_list)
    if per_page <= 0 or per_page > task_list_len:
        per_page = task_list_len

    total_pages = 1
    if per_page > 0:
        total_pages = len(task_list) // per_page
        if task_list_len % per_page != 0:
            total_pages += 1

    if page <= 0:
        page = 1
    elif page > total_pages and page != 1:
        page = total_pages

    if total_pages == 0:
        total_pages = 1

    navigation = Pagination.Navigation(self='page={}&per_page={}'.format(page, per_page),
                                       first='page=1&per_page={}'.format(per_page),
                                       previous='page={}&per_page={}'.format(1 if page == 1 else page - 1, per_page),
                                       next='page={}&per_page={}'.format(total_pages if page == total_pages
                                                                         else page + 1, per_page),
                                       last='page={}&per_page={}'.format(total_pages, per_page))

    pagination = Pagination(page=page, per_page=per_page, page_count=total_pages, total_count=task_list_len,
                            navigation=navigation)

    return TaskListResponse(code=HTTPReturnCode.OK, pagination=pagination,
                            tasks=task_list[(page - 1) * per_page: min(page * per_page, task_list_len)])


if __name__ == '__main__':
    response = get_started_tasks()
    print(1)
