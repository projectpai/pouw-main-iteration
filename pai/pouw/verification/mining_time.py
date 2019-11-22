import argparse
import datetime
import json
import math
import mxnet as mx

import redis
from mxnet import gluon

from pai.pouw.constants import OMEGA
from pai.pouw.mining.allowed_time import LiveMedian, get_network_time_to_pick_next_message

parser = argparse.ArgumentParser(description='Allowed mining time verifier')
parser.add_argument('--cuda', action='store_true', default=False,
                    help='Train on GPU with CUDA')
parser.add_argument('--redis-host', type=str, default='localhost',
                    help='redis host address used for worker synchronization')
parser.add_argument('--redis-port', type=int, default=6379,
                    help='Redis port used for connecting to redis database')
opt = parser.parse_args()


# given a message, this function return the previous messages from its epoch
def get_previous_messages(conn, message_key):
    task_id = message_key.split('_')[2]
    epoch = message_key.split('_')[3]
    message_no = int(message_key.split('_')[4])

    message_key_template = 'iteration_info_{}_{}_*'
    message_key = message_key_template.format(task_id, epoch)
    epoch_messages = conn.keys(message_key)

    previous_messages = sorted([msg for msg in epoch_messages if int(msg.split('_')[4]) < message_no],
                               key=lambda item: int(item.split('_')[4]))
    return previous_messages


def calculate_allowed_mining_time(conn, message_key, network_time):
    previous_messages = get_previous_messages(conn, message_key)
    json_maps = conn.mget(previous_messages)
    messages = [json.loads(data) for data in json_maps if data is not None]
    timedelta = LiveMedian()
    for msg in messages:
        computation_time = (datetime.datetime.strptime(msg['tf'], '%Y-%m-%d %H:%M:%S.%f')
                            - datetime.datetime.strptime(msg['ts'],
                                                         '%Y-%m-%d %H:%M:%S.%f')).total_seconds()
        timedelta.add(computation_time)

    allowed_time = timedelta.get_median()
    if math.isnan(allowed_time):
        allowed_time = 0
    return min(allowed_time / math.exp(1), network_time)


if __name__ == '__main__':
    if opt.cuda:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()

    conn = redis.Redis(host=opt.redis_host, port=opt.redis_port)
    conn.ping()

    # calculate network allowed mining time
    ver_net = gluon.nn.SymbolBlock.imports("it_mnist_000-symbol.json", ['data'], "it_mnist_000-0000.params", ctx)
    grads = [g.grad(ctx) for g in ver_net.collect_params().values() if g._grad is not None]
    gradients_sizes = [g.size for g in grads]
    network_time = get_network_time_to_pick_next_message(OMEGA, gradients_sizes, './it_mini-batch-data_000',
                                            './it_mini-batch-label_000')

    allowed_time = calculate_allowed_mining_time(conn, 'iteration_info_cd207a3d-18cc-4d0b-8490-483c33353699_0_62',
                                                 network_time)

    print(allowed_time)
