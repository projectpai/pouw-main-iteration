import argparse
from distutils.util import strtobool
from multiprocessing import Pool

from pai.pouw.cluster import call_wrapper

if __name__ == '__main__':
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