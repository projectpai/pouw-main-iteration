# utilised code from https://github.com/zebra-lucky/python-bls/blob/master/bls_py/tests.py

import random
import uuid
from itertools import combinations

from bls_py.bls import BLS
from bls_py.keys import PrivateKey
from bls_py.threshold import Threshold

# the message we want to sign (in this case, the solution to the most famous challenge ciphertext of the year 1977)
message = "The Magic Words are Squeamish Ossifrage"

# how many supervisors; supervisors are indexed (0, 1 ..)
N = 5

# the minimum number required to sign a transaction
M = 3

# the private shares; a supervisor can see only his/her row (supervisor 0 has access to
# private_shares[0], supervisor 1 has only access to private_shares[1] and so on
private_shares = [[None] * N for _ in range(N)]

# the public shares are seen by everybody
public_shares = []


def generate_task_id():
    return uuid.uuid4().hex


def generate_new_threshold_key(t, n):
    return PrivateKey.new_threshold(t, n)


def send_private_share(target, sender, share):
    private_shares[target][sender] = share


def send_public_polynomial(polynomial):
    public_shares.append(polynomial)


def dkg(node):
    node_secret_key, node_public_polynomial, node_private_shares = generate_new_threshold_key(M, N)
    for target, share in enumerate(node_private_shares):
        send_private_share(target, node, share)
    send_public_polynomial(node_public_polynomial)


def verify_received_private_share(node_target, node_source):
    return Threshold.verify_secret_fragment(
        M, private_shares[node_target - 1][node_source - 1],
        node_target, public_shares[node_source - 1])


def verify_private_shares_from_all_nodes():
    # O(n^2) verification
    for node_source in range(1, N + 1):
        for node_target in range(1, N + 1):
            if not verify_received_private_share(node_target, node_source):
                return False
    return True


def aggregate_private_shares(shares):
    return BLS.aggregate_priv_keys(map(PrivateKey, shares), None, False)


def sign(private_share, msg, node, quorum):
    return private_share.sign_threshold(msg, node, quorum)


def random_quorum():
    all_quora = list(combinations(range(1, N + 1), M))
    return random.choice(all_quora)


def generate_signature_shares(pks):
    return [sign(pks[member - 1], message, member, quorum) for member in quorum]


if __name__ == "__main__":
    # Run DKG for each node
    for node in range(N):
        dkg(node)

    # Verify all nodes were honest
    if not verify_private_shares_from_all_nodes():
        print('Verification of private shares failed. Protocol aborted!')
        exit(1)

    # Each node will aggregate his/her private shares
    # We convene that only the corresponding row is visible to the supervisor (e.g. supervisor 3 sees the third row etc)
    aggregated_secret_shares = [aggregate_private_shares(share) for share in private_shares]

    # Elect an ad-hoc quorum of M nodes (supervisors)
    quorum = random_quorum()

    # Generate signature shares for the message;
    # Signature shares can be visible to everyone (they are not secret)
    signature_shares = [sign(aggregated_secret_shares[member - 1], message, member, quorum)
                        for member in quorum]

    # The leader gathers the signatures from the quorum and
    # builds a signature that is appended with the message.
    # This threshold signature is the same as the global signature on the same message.
    signature = BLS.aggregate_sigs_simple(signature_shares)

    print('Message: %s' % message)
    print('Signature: %s' % signature.serialize())
