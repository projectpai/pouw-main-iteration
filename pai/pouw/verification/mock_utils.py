import uuid
import random

from pai.pouw.verification.task_info_pb2 import TaskListResponse, TaskRecord, Pagination, HTTPReturnCode
from google.protobuf.timestamp_pb2 import Timestamp

MODEL_TYPES = ['FC-DNN', 'CNN', 'LSTM', 'RNN']
BATCH_SIZE = [16, 32, 64, 128, 256, 512, 1024]
OPTIMIZERS = ['SGD', 'Adam', 'Nadam', 'Adagrad', 'Adamax']


def get_mock_tasks(page, per_page):
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