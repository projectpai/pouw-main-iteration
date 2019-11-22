import argparse
import json
import pprint
import random
import re
import string
import sys
from time import sleep

import redis
from bls_py.aggregation_info import AggregationInfo
from bls_py.bls import BLS
from bls_py.ec import (default_ec, AffinePoint)
from bls_py.fields import Fq
from bls_py.keys import PrivateKey, PublicKey
from bls_py.threshold import Threshold
from bls_py.signature import Signature

from pai.pouw.mining.blkmaker.blkmaker import sha256_hexdigest

M = 3


def generate_new_threshold_key(t, n):
    return PrivateKey.new_threshold(t, n)


def deserialize_private_share(serialized_request):
    return Fq(default_ec.q, int(json.loads(serialized_request)['prv_share']))


def deserialize_public_polynomial(serialized_request):
    serialized_polynomial = json.loads(serialized_request)['pub_share']
    aff_points = []
    for af in [affine_point[1:-4] for affine_point in serialized_polynomial.split('AffinePoint') if
               len(affine_point) > 5]:
        fqs = re.findall('Fq\((.*?)\)', af)
        x = Fq(default_ec.q, int(fqs[0], 0))
        y = Fq(default_ec.q, int(fqs[1], 0))
        i = re.search('i=(.*)', af).group(1) == True
        aff_points.append(AffinePoint(x, y, i, default_ec))

    return aff_points


def generate_random_message():
    letters = string.ascii_letters
    msg = ''.join(random.choice(letters) for _ in range(8))
    return msg


def deserialize_signature_share(serialized_request):
    serialized_signature_share = json.loads(serialized_request)['signature_share']
    return Signature.from_bytes(serialized_signature_share.encode('latin1'))


def get_supervisor_id(serialized_request):
    return int(json.loads(serialized_request)['supervisor_id'])


def get_global_signature(signatures):
    global_signature = Threshold.aggregate_unit_sigs([sig[1] for sig in signatures], [sig[0]+1 for sig in signatures], M)
    return global_signature


def vote():
    # simulate a situation in which on average 2/3 fo nodes agree
    return random.randint(0, 2) != 0


class DKGSupervisor:
    def __init__(self, task_id, supervisor_id, leader_id, nodes_number, redis_host, redis_port):
        self.message_key_template = None
        self.task_id = task_id
        self.supervisor_id = supervisor_id
        self.leader_id = leader_id
        self.nodes_number = nodes_number
        self.secret_key = None
        self.private_shares = None
        self.public_shares = None
        self.aggregated_private_key = None
        self.master_pubkey = None
        self.conn = redis.Redis(host=redis_host, port=redis_port)
        self.conn.ping()

    def dkg(self):
        self.secret_key, node_public_polynomial, node_private_shares = generate_new_threshold_key(M, self.nodes_number)
        for target, share in enumerate(node_private_shares):
            self.send_private_share(target, share)

        self.send_public_polynomial(node_public_polynomial)

    def send_private_share(self, target, share):
        self.message_key_template = 'dkg_prv_share_{}_{}_{}'
        self.message_key_template = self.message_key_template.format(self.task_id, self.supervisor_id, target)

        # the message key contains: task_id and (from, to) ids
        dkg_request = {
            'supervisor_id': self.supervisor_id,
            'prv_share': share
        }

        self.write_to_db(dkg_request)

    def write_to_db(self, dkg_request):
        serialized_request = json.dumps(dkg_request)
        self.conn.set(self.message_key_template, serialized_request)

    def send_public_polynomial(self, polynomial):
        self.message_key_template = 'dkg_pub_share_{}_{}'
        self.message_key_template = self.message_key_template.format(self.task_id, self.supervisor_id)

        # the message key contains: task_id and (from, to) ids
        dkg_request = {
            'supervisor_id': self.supervisor_id,
            'pub_share': repr(polynomial)
        }

        self.write_to_db(dkg_request)

    def receive_private_shares(self):
        self.message_key_template = 'dkg_prv_share_{}_{}_{}'
        self.message_key_template = self.message_key_template.format(self.task_id, '{}', self.supervisor_id)
        keys = [self.message_key_template.format(index) for index in range(self.nodes_number)]
        json_maps = self.conn.mget(keys)
        self.private_shares = [(get_supervisor_id(data), deserialize_private_share(data)) for data in json_maps if
                               data is not None]

    def receive_public_polynomials(self):
        self.message_key_template = 'dkg_pub_share_{}_{}'
        self.message_key_template = self.message_key_template.format(self.task_id, '{}')
        keys = [self.message_key_template.format(index) for index in range(self.nodes_number)]
        json_maps = self.conn.mget(keys)
        self.public_shares = [(get_supervisor_id(data), deserialize_public_polynomial(data)) for data in json_maps if
                              data is not None]

    def aggregate_private_shares(self):
        self.aggregated_private_key = BLS.aggregate_priv_keys(
            map(PrivateKey, [share[1] for share in self.private_shares]), None, False)

    def is_leader(self):
        return self.supervisor_id == self.leader_id

    def send_proposed_message(self):
        self.message_key_template = 'dkg_message_{}'.format(self.task_id)

        dkg_request = {
            'message': ''.join(random.choice(string.ascii_letters) for _ in range(10))
        }

        self.write_to_db(dkg_request)

    def receive_message(self):
        self.message_key_template = 'dkg_message_{}'.format(self.task_id)
        json_maps = self.conn.mget(self.message_key_template)
        messages = [json.loads(data)['message'] for data in json_maps if data is not None]
        if len(messages) > 0:
            return messages[0]
        return None

    def create_signature_share(self, message):
        return self.aggregated_private_key.sign(message)

    def send_signed_message(self, message, signature_share):
        self.message_key_template = 'dkg_signed_message_{}_{}_{}'.format(self.task_id, sha256_hexdigest(message.encode('latin1')),
                                                                         self.supervisor_id)

        dkg_request = {
            'supervisor_id': self.supervisor_id,
            'message': message,
            'signature_share': signature_share.serialize().decode('latin1')
        }

        self.write_to_db(dkg_request)

    def receive_signature_shares(self, message):
        self.message_key_template = 'dkg_signed_message_{}_{}_{}'
        self.message_key_template = self.message_key_template.format(self.task_id, sha256_hexdigest(message.encode('latin1')), '{}')
        keys = [self.message_key_template.format(index) for index in range(self.nodes_number)]
        json_maps = self.conn.mget(keys)
        return [(get_supervisor_id(data), deserialize_signature_share(data)) for data in json_maps if data is not None]

    def create_master_pub_key(self):
        self.master_pubkey = BLS.aggregate_pub_keys(
            [PublicKey.from_g1(cpoly[0].to_jacobian()) for cpoly in [pub[1] for pub in self.public_shares]], False)

    def verify_signature(self, message, signature):
        agg_info = AggregationInfo.from_msg(self.master_pubkey, message)
        signature.set_aggregation_info(agg_info)
        return BLS.verify(signature)

    def verify_secret_fragment(self, sender, private_share):
        return Threshold.verify_secret_fragment(M, private_share, self.supervisor_id + 1, self.get_pub_poly(sender))

    def verify_all_secret_fragments(self):
        faulty_senders = []
        for (sender, share) in self.private_shares:
            if not self.verify_secret_fragment(sender, share):
                faulty_senders.append(sender)
        return faulty_senders

    def get_pub_poly(self, index):
        found_item = [item[1] for item in self.public_shares if item[0] == index]
        if found_item is not None:
            return found_item[0]
        return found_item


def wait_some_time():
    sleep(3)


def main():
    parser = argparse.ArgumentParser(description='DKG simulation')
    parser.add_argument('--task-id', type=str, default='default_task',
                        help='The task id')
    parser.add_argument('--index', type=int, default=0,
                        help='Id/index of the current node')
    parser.add_argument('--leader-index', type=int, default=0,
                        help='Id/index of the leader')
    parser.add_argument('--nodes-number', type=int, default=5,
                        help='The total number of nodes')
    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='redis host address used for worker synchronization')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port used for connecting to redis database')

    args = parser.parse_args()

    # DKG
    supervisor = DKGSupervisor(args.task_id, args.index, args.leader_index, args.nodes_number, args.redis_host,
                               args.redis_port)
    supervisor.dkg()

    # wait some seconds to finish the initial DKG procedure
    wait_some_time()

    supervisor.receive_private_shares()
    supervisor.receive_public_polynomials()

    # nodes will verify what they received
    wait_some_time()
    faulty_nodes = supervisor.verify_all_secret_fragments()
    if len(faulty_nodes) > 0:
        print('These are the faulty nodes:')
        pprint.pprint(faulty_nodes)
        sys.exit(1)

    supervisor.aggregate_private_shares()
    supervisor.create_master_pub_key()

    # the supervisor proposes a message only if he's the epoch leader
    if supervisor.is_leader():
        supervisor.send_proposed_message()

    wait_some_time()

    message = supervisor.receive_message()
    if message is None:
        sys.exit(1)

    # the supervisor inspects the message and votes
    # sends the signed message only if agrees with it
    if vote() is True or supervisor.is_leader():
        signature_share = supervisor.create_signature_share(message)
        supervisor.send_signed_message(message, signature_share)

    # the leader gathers the results and validates them
    if supervisor.is_leader():
        wait_some_time()
        signatures = supervisor.receive_signature_shares(message)
        if len(signatures) < M:
            print('Message did not get enough votes!')
        else:
            global_signature = get_global_signature(signatures)
            if supervisor.verify_signature(message, global_signature):
                print('Message is valid!')
            else:
                print('Message is invalid!')
    else:
        print('Node finished job {}'.format(args.index))


if __name__ == '__main__':
    main()
