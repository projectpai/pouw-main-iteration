from hashlib import sha256

from charm.schemes.pk_vrf import VRF10
from charm.toolbox.pairinggroup import PairingGroup

from pai.pouw.ticket_selection.similarity import cosine_similarity

# ticket preferences vector
ticket_prefs = [0, 7, 100]

# corresponding task properties vector
task_props = [0, 2, 1]


def get_bit(data, num):
    base = int(num // 8)
    shift = int(num % 8)
    return (data[base] & (1 << shift)) >> shift


def bytearray_to_bits(data):
    return [get_bit(data, i) for i in range(len(data) * 8)]


def setup(message):
    message_hash = sha256(message).digest()
    bit_message = bytearray_to_bits(message_hash)
    # block of bits
    n = len(bit_message)
    grp = PairingGroup('MNT224')
    vrf = VRF10(grp)
    (pk, sk) = vrf.setup(n)
    return bit_message, pk, sk, vrf


def get_hash_threshold(st):
    hash_obj = sha256(repr(st['y']).encode())
    actual_hash = int.from_bytes(hash_obj.digest(), 'big')
    max_hash = 2 ** 256 - 1
    hash_threshold = actual_hash / max_hash
    return hash_threshold


def main():

    # we pass the role: miner, supervisor etc and the
    message = bytearray("miner9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08".encode())

    bit_message, pk, sk, vrf = setup(message)

    # generate proof over block x (using sk)
    st = vrf.prove(sk, bit_message)

    hash_threshold = get_hash_threshold(st)
    print(f'Ratio: {hash_threshold:.2f}')

    # verify bits using pk and proof
    if vrf.verify(pk, bit_message, st):
        print('Verification succeeded.')
    else:
        print('Verification failed.')

    cos_sim = cosine_similarity(ticket_prefs, task_props)
    print(f'Similarity: {cos_sim:.4f}')

    if cos_sim >= hash_threshold:
        print('Ticket selected to work.')
    else:
        print('Ticket should wait for another task.')


if __name__ == "__main__":
    main()