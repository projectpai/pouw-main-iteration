import redis

from pai.pouw.verification.task_info_pb2 import TaskListResponse


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


def get_waiting_tasks(redis_host='localhost', redis_port=6379):
    conn = redis.Redis(host=redis_host, port=redis_port)
    conn.ping()

    training_tasks = list(set([task.decode("utf-8").split('_')[2] for task in conn.keys('training_start_*')]))
    result = [task.decode("utf-8").split('_')[2] for task in conn.scan_iter('task_submitted_*')
              if task.decode("utf-8").split('_')[2] not in training_tasks]
    return TaskListResponse(tasks=result)


if __name__ == '__main__':
    print(get_waiting_tasks())
