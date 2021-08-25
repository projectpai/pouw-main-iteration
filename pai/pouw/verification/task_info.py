import uuid

import redis

from pai.pouw.verification.task_info_pb2 import TaskListResponse, TaskRecord, Pagination, HTTPReturnCode
from google.protobuf.timestamp_pb2 import Timestamp


def get_tasks_info(redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    training_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('training_start_*')]))
    done_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('task_done_*')]))

    final_submitted_tasks = []
    final_training_tasks = []
    final_done_tasks = []
    for task in conn.scan_iter('task_submitted_*'):
        task_id = task.decode("utf-8").split('_')[2]
        if task_id in done_tasks:
            final_done_tasks.append(task_id)
            continue
        if task_id in training_tasks:
            final_training_tasks.append(task_id)
            continue
        final_submitted_tasks.append(task_id)

    return [final_submitted_tasks, final_training_tasks, final_done_tasks]


# def get_waiting_tasks(redis_host='localhost', redis_port=6379):
#     conn = redis.Redis(host=redis_host, port=redis_port)
#     conn.ping()
#
#     training_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('training_start_*')]))
#     result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('task_submitted_*')
#               if task.decode("utf-8").split('_')[2] not in training_tasks]
#     return TaskListResponse(tasks=result)

def get_waiting_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    training_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('training_start_*')]))
    result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('task_submitted_*')
              if task.decode("utf-8").split('_')[2] not in training_tasks]

    timestamp = Timestamp()
    timestamp.GetCurrentTime()
    task_record1 = TaskRecord(task_id=str(uuid.uuid4()), model_type='FC-DNN', nodes_no=128, batch_size=512,
                              optimizer='SGD', created=timestamp)

    timestamp.GetCurrentTime()
    task_record2 = TaskRecord(task_id=str(uuid.uuid4()), model_type='FC-DNN', nodes_no=64, batch_size=32,
                              optimizer='Adam', created=timestamp)

    navigation = Pagination.Navigation(self='page={}&per_page={}'.format(page, per_page), first='page=1&per_page=20',
                                       previous='page=1&per_page=20',
                                       next='page=2&per_page=20', last='page=5&per_page=20')
    pagination = Pagination(page=page, per_page=per_page, page_count=5, total_count=92, navigation=navigation)

    return TaskListResponse(code=HTTPReturnCode.OK, pagination=pagination, tasks=[task_record1, task_record2])


if __name__ == '__main__':
    response = get_waiting_tasks()
    print(1)
