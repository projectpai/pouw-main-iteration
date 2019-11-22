import hashlib
import multiprocessing as mp
import pprint

import numpy as np

NUM_BATCHES = 5
NUM_COMMITTEE = 3


def vote(q, batches_list):
    randst = np.random.mtrand.RandomState()
    arr = randst.permutation(len(batches_list))
    q.put(zip(batches_list, arr.tolist()))


def dump_queue(queue):
    ret_queue = []
    while not queue.empty():
        ret_queue.append(queue.get())

    return ret_queue


def count_votes(q):
    borda_votes = {}
    q_list = dump_queue(q)
    for vote_list in q_list:
        for vote_tuple in vote_list:
            borda_votes.setdefault(vote_tuple[0], []).append(vote_tuple[1])

    borda_count = {}
    for k, v in borda_votes.iteritems():
        borda_count[k] = sum(v)

    return [(k, borda_count[k]) for k in sorted(borda_count, key=borda_count.get, reverse=True)]


if __name__ == '__main__':

    batches = []
    for i in range(NUM_BATCHES):
        batches.append(hashlib.sha256(str(i)).hexdigest())

    q = mp.Queue()

    procs = []
    for i in range(NUM_COMMITTEE):
        p = mp.Process(target=vote, args=(q, batches, ))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    counts = count_votes(q)
    pp = pprint.PrettyPrinter(depth=6)
    pp.pprint(counts)
