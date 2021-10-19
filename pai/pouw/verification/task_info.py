import json
import statistics
from datetime import datetime

import redis
from google.protobuf.timestamp_pb2 import Timestamp

from pai.pouw.verification.task_info_pb2 import TaskListResponse, TaskRecord, Pagination, HTTPReturnCode, \
    TaskDetailsResponse, TaskIDResponse, EpochsDetails, EpochInfo, MetricAvgValue

UNAVAILABLE = 'UNAVAILABLE'


def get_task_id(msg_id, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    json_maps = conn.mget(msg_id)
    iterations_data = [json.loads(data) for data in json_maps if data is not None]
    if len(iterations_data) == 0:
        return TaskIDResponse(code=HTTPReturnCode.NOT_FOUND, task_id=UNAVAILABLE)
    it_data = iterations_data[0]
    task_id = it_data['task_id']

    return TaskIDResponse(code=HTTPReturnCode.OK, task_id=task_id)


def get_waiting_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    training_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('training_start_*')]))
    result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('task_submitted_*')
              if task.decode("utf-8").split('_')[2] not in training_tasks]

    return pack_task_list(result, page, per_page, conn)


def get_started_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    done_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('task_done_*')]))

    result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('training_start_*')
              if task.decode("utf-8").split('_')[2] not in done_tasks]

    return pack_task_list(result, page, per_page, conn)


def get_completed_tasks(page=1, per_page=20, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    result = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('task_done_*')]))
    return pack_task_list(result, page, per_page, conn)


def get_completed_epochs(conn, task_id):
    epoch_details_list = []
    scanned_epochs = set()
    for epoch in conn.scan_iter('epoch_details_{}_*'.format(task_id)):
        epoch_no = epoch.decode("utf-8").split('_')[3]
        if epoch_no not in scanned_epochs:
            scanned_epochs.add(int(epoch_no))
            metrics = [json.loads(row.decode("utf-8")) for row in conn.lrange(epoch, 0, -1)]
            metric_avg_list = []
            if len(metrics) == 0:
                continue
            for k in metrics[0]:
                if k == 'miner_id':
                    continue
                metric_avg = MetricAvgValue(metric=k, avg_value=statistics.mean([m[k] for m in metrics]))
                metric_avg_list.append(metric_avg)

            epoch_info = EpochInfo(epoch_no=int(epoch_no), metrics=metric_avg_list)
            epoch_details_list.append(epoch_info)

    return len(scanned_epochs), sorted(epoch_details_list, key=lambda x: x.epoch_no)


def get_task_details(task_id, redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    task_details = get_task_full_info(task_id, conn)

    if task_details is not None:
        completed_epochs_no, epochs_details_list = get_completed_epochs(conn, task_id)
        epoch_details = EpochsDetails(total_epochs=task_details['ml']['optimizer']['epochs'],
                                      completed_epochs=completed_epochs_no, epochs_metrics=epochs_details_list)
        return TaskDetailsResponse(code=HTTPReturnCode.OK,
                                   task_id=task_id, model_type=task_details['ml']['model']['type'],
                                   nodes_no=sum([hu['nodes'] for hu in task_details['ml']['model']['hidden-units']]),
                                   batch_size=task_details['ml']['optimizer']['batch-size'],
                                   optimizer=task_details['ml']['optimizer']['type'],
                                   created=get_grpc_timestamp(task_details),
                                   dataset=task_details['ml']['dataset']['format'],
                                   initializer=task_details['ml']['optimizer']['initializer']['name'],
                                   loss_function=task_details['ml']['model']['loss'],
                                   tau=task_details['ml']['optimizer']['tau'],
                                   evaluation_metrics=task_details['ml']['evaluation-metrics'],
                                   epochs_info=epoch_details)

    # task is not found
    timestamp = Timestamp()
    timestamp.GetCurrentTime()

    epoch_details = EpochsDetails(total_epochs=0, completed_epochs=0, epochs_metrics=[])
    return TaskDetailsResponse(code=HTTPReturnCode.NOT_FOUND,
                               task_id=task_id, model_type=UNAVAILABLE,
                               nodes_no=0,
                               batch_size=0,
                               optimizer=UNAVAILABLE,
                               created=timestamp,
                               dataset=UNAVAILABLE,
                               initializer=UNAVAILABLE,
                               loss_function=UNAVAILABLE,
                               tau=0.0,
                               evaluation_metrics=[],
                               epochs_info=epoch_details)


def get_task_full_info(task_id, conn):
    details = None
    for full_task in conn.scan_iter('task_submitted_{}_*'.format(task_id)):
        json_task_details = conn.mget(full_task.decode('utf-8'))
        if json_task_details is not None and json_task_details[0] is not None:
            details = json.loads(json_task_details[0].decode('utf-8'))
        break

    return details


def pack_task_list(task_ids, page, per_page, conn):
    task_list = []
    for task_id in task_ids:
        task_details = get_task_full_info(task_id, conn)
        if task_details is not None:
            proto_timestamp = get_grpc_timestamp(task_details)

            task_record = TaskRecord(task_id=task_id, model_type=task_details['ml']['model']['type'],
                                     nodes_no=sum([hu['nodes'] for hu in task_details['ml']['model']['hidden-units']]),
                                     batch_size=task_details['ml']['optimizer']['batch-size'],
                                     optimizer=task_details['ml']['optimizer']['type'],
                                     created=proto_timestamp)
        else:
            timestamp = Timestamp()
            timestamp.GetCurrentTime()
            task_record = TaskRecord(task_id=task_id, model_type=UNAVAILABLE,
                                     nodes_no=0,
                                     batch_size=0,
                                     optimizer=UNAVAILABLE,
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


def get_grpc_timestamp(task_details):
    try:
        time_float = float(task_details['created'])
        ts = datetime.utcfromtimestamp(time_float).timestamp()
        seconds = int(ts)
        nanos = int(ts % 1 * 1e9)
        proto_timestamp = Timestamp(seconds=seconds, nanos=nanos)
    except ValueError:
        timestamp = Timestamp()
        timestamp.GetCurrentTime()
        proto_timestamp = timestamp
    return proto_timestamp


# to be used only for debugging purposes
if __name__ == '__main__':
    response = get_task_details('68e292389fc8526e8268fafa0afd7c3b87e285a884f2e7e1ec58c9f1035ae8d1')
    print(1)
