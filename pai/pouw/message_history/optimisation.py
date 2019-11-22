from copy import deepcopy

from merkletools import MerkleTools
from suffix_trees import STree
from hashlib import sha256
from pprint import pprint

SHA_SIZE = 64
COMPRESSION_THRESHOLD = 1


def get_lcs(chunk_hashes):
    chunk_hashes_input = ['$'.join(chunk_hash) for chunk_hash in chunk_hashes]
    st = STree.STree(chunk_hashes_input)
    return st.lcs()


def get_common_messages(chunk_hashes):
    common_segment = list(filter(None, get_lcs(chunk_hashes).split('$')))
    if common_segment is not None and len(common_segment) > COMPRESSION_THRESHOLD and\
            len(common_segment[0]) == SHA_SIZE and len(common_segment[1]) == SHA_SIZE:
        return common_segment
    else:
        return None


def get_merkle_tree(message_hashes):
    mt = MerkleTools()
    mt.add_leaf(message_hashes)
    mt.make_tree()
    return mt


def compress_message_history(chunk_hashes):
    local_chunk_hashes = deepcopy(chunk_hashes)

    # determine the longest common sub-sequence
    common_messages = get_common_messages(local_chunk_hashes)
    while common_messages:
        mt = get_merkle_tree(common_messages)
        # replace the common messages with their Merkle tree root
        for i, _ in enumerate(local_chunk_hashes):
            repl_index = local_chunk_hashes[i].index(common_messages[0])
            local_chunk_hashes[i][repl_index] = mt.get_merkle_root()
            for j in range(1, len(common_messages)):
                local_chunk_hashes[i].remove(common_messages[j])

        common_messages = get_common_messages(local_chunk_hashes)

    return local_chunk_hashes


if __name__ == "__main__":

    s1 = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    s2 = ["1", "2", "x", "4", "5", "x", "7", "8", "9"]
    s3 = ["x", "2", "3", "4", "5", "6", "7", "8", "9"]

    chunk = [s1, s2, s3]
    chunk_hashes = [[sha256(msg.encode()).hexdigest() for msg in sup] for sup in chunk]
    print('Original messages:')
    pprint(chunk_hashes)

    compressed_messages = compress_message_history(chunk_hashes)
    print('Compressed messages:')
    pprint(compressed_messages)
