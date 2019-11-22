import argparse
import subprocess
from multiprocessing.pool import Pool

import receiver_worker
import sender_worker


def call_worker(index):
    receiver_script_path = receiver_worker.__file__

    if receiver_script_path.endswith('pyc'):
        receiver_script_path = receiver_script_path[:-1]

    sender_script_parameters = ['python3',
                         receiver_script_path,
                         '--index',  str(index)
                         ]

    subprocess.call(
        sender_script_parameters
    )


def call_wrapper(args):
    return call_worker(args)


def main():
    parser = argparse.ArgumentParser(description='Communication simulation')
    parser.add_argument('--nodes-number', type=int, default=2,
                        help='number of nodes in cluster')

    args = parser.parse_args()

    receiver_worker_args = [index for index in range(args.nodes_number)]

    pool = Pool(processes=args.nodes_number)
    pool.map(call_wrapper, receiver_worker_args)


if __name__ == '__main__':
    main()
