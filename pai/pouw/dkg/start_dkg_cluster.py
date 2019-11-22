import argparse
import random
import subprocess
from multiprocessing.pool import Pool

import pai
from pai.pouw.dkg.distributed_joint_feldman import DKGSupervisor
from pai.pouw.dkg.joint_feldman import generate_task_id


def call_worker(index, task_id, leader_index, nodes_number, redis_host='localhost', redis_port=6379):
    worker_script_path = pai.pouw.dkg.distributed_joint_feldman.__file__
    # in order to ensure debugger is working properly
    if worker_script_path.endswith('pyc'):
        worker_script_path = worker_script_path[:-1]

    script_parameters = ['python3',
                         worker_script_path,
                         '--task-id', task_id,
                         '--index',  str(index),
                         '--leader-index', str(leader_index),
                         '--nodes-number', str(nodes_number),
                         '--redis-host', redis_host,
                         '--redis-port', str(redis_port)
                         ]

    subprocess.call(
        script_parameters
    )


def call_wrapper(args):
    return call_worker(*args)


def main():
    parser = argparse.ArgumentParser(description='DKG simulation')
    parser.add_argument('--nodes-number', type=int, default=5,
                        help='number of nodes in cluster')
    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='redis host address used for worker synchronization')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port used for connecting to redis database')

    args = parser.parse_args()

    leader_index = random.randint(0, args.nodes_number)

    print('Leader is node with index: %d' % leader_index)

    task_id = generate_task_id()

    worker_args = [(index, task_id, leader_index, args.nodes_number, args.redis_host, str(args.redis_port))
                   for index in range(args.nodes_number)]

    pool = Pool(processes=args.nodes_number)
    pool.map(call_wrapper, worker_args)


if __name__ == '__main__':
    main()
