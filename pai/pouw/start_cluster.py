import argparse
import subprocess
from distutils.util import strtobool
from multiprocessing import Pool

import pai.pouw.nodes.decentralized.worker


def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser(description='Starting up POUW cluster')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Train on GPU with CUDA')
    parser.add_argument('--nodes-number', type=int, default=3,
                        help='number of nodes in cluster')
    parser.add_argument('--debug', type=strtobool, default=False,
                        help='provide more verbose logging messages')
    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='redis host address used for worker synchronization')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port used for connecting to redis database')
    parser.add_argument('--use-paicoin', type=strtobool, default=True,
                        help='enable/disable usage of paicoin for testing and debugging purposes')
    parser.add_argument('--python-interpreter', type=str, default='python3',
                        help='name of Python executable')

    args = parser.parse_args()

    worker_args = [(args.debug, args.cuda, args.use_paicoin,
                    args.redis_host, args.redis_port, args.python_interpreter)
                   for _ in range(args.nodes_number)]

    pool = Pool(processes=args.nodes_number)
    pool.map(call_wrapper, worker_args)


def call_wrapper(args):
    return call_worker(*args)


def call_worker(is_debug=False, cuda=False, use_paicoin=True, redis_host='localhost', redis_port=6379, python_interpreter='python3'):
    worker_script_path = pai.pouw.nodes.decentralized.worker.__file__
    # in order to ensure debugger is working properly
    if worker_script_path.endswith('pyc'):
        worker_script_path = worker_script_path[:-1]

    script_parameters = [python_interpreter, worker_script_path,
                         '--redis-host', redis_host,
                         '--redis-port', str(redis_port),
                         '--use-paicoin', str(use_paicoin),
                         '--cuda', str(cuda),
                         ]
    if is_debug:
        script_parameters.append('--debug')
        script_parameters.append('True')

    subprocess.call(
        script_parameters
    )


if __name__ == '__main__':
    main()