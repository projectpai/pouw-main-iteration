import uuid
import random
import redis

from pai.pouw.verification.task_info_pb2 import TaskListResponse, TaskRecord, Pagination, HTTPReturnCode
from google.protobuf.timestamp_pb2 import Timestamp

MODEL_TYPES = ['FC-DNN', 'CNN', 'LSTM', 'RNN']
BATCH_SIZE = [16, 32, 64, 128, 256, 512, 1024]
OPTIMIZERS = ['SGD', 'Adam', 'Nadam', 'Adagrad', 'Adamax']


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

    task_list = []
    timestamp = Timestamp()
    for i in range(random.randint(1, 30)):
        timestamp.GetCurrentTime()
        task_record = TaskRecord(task_id=str(uuid.uuid4()), model_type=random.choice(MODEL_TYPES),
                                 nodes_no=random.randint(1, 100) * 32,
                                 batch_size=random.choice(BATCH_SIZE),
                                 optimizer=random.choice(OPTIMIZERS), created=timestamp)
        task_list.append(task_record)

    # adjust for corner cases
    task_list_len = len(task_list)
    if per_page <= 0 or per_page > task_list_len:
        per_page = task_list_len

    total_pages = len(task_list) // per_page
    if task_list_len % per_page != 0:
        total_pages += 1
    if page <= 0:
        page = 1
    elif page > total_pages:
        page = total_pages

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
    response = get_waiting_tasks()
    print(1)
