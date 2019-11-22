import random
import simpy
import uuid
import pprint
from py3votecore.schulze_npr import SchulzeNPR
from timeit import default_timer as timer


class BroadcastPipe(object):
    def __init__(self, env, capacity=simpy.core.Infinity):
        self.env = env
        self.capacity = capacity
        self.pipes = []

    def put(self, value):
        if not self.pipes:
            raise RuntimeError('There are no output pipes.')
        events = [store.put(value) for store in self.pipes]
        return self.env.all_of(events)  # Condition event for all "events"

    def get_output_conn(self):
        pipe = simpy.Store(self.env, capacity=self.capacity)
        self.pipes.append(pipe)
        return pipe


class Witness:
    def __init__(self, env, id):
        self.messages = []
        self.env = env
        self.id = id
        self.in_pipes = []
        self.action = env.process(self.run())

    def add_pipe(self, in_pipe):
        self.in_pipes.append(in_pipe)

    def run(self):
        while True:
            positions = [[i] for i in range(len(self.in_pipes))]
            # random.shuffle(positions)
            for p in positions:
                msg = yield self.in_pipes[p[0]].get()

                self.messages.append((self.env.now, msg[1]))

                print('Time %d, witness %d received from miner %d the message \"%s\".' %
                  (self.env.now, self.id, msg[0], msg[1]))

                yield self.env.timeout(random.randint(1, 2))


class Miner:
    def __init__(self, env, id):
        self.env = env
        self.id = id
        self.out_pipes = []
        self.action = env.process(self.run())

    def add_pipe(self, out_pipe):
        self.out_pipes.append(out_pipe)

    def run(self):
        while True:
            msg = (self.id, uuid.uuid4().hex)
            for pipe in self.out_pipes:
                yield self.env.timeout(random.randint(1, 1))
                pipe.put(msg)


def main():
    env = simpy.Environment()

    miners = []
    witnesses = []
    for m in range(9):
        miners.append(Miner(env, m))

    for w in range(3):
        witnesses.append(Witness(env, w))

    for m in miners:
        for w in witnesses:
            bc_pipe = BroadcastPipe(env)
            m.add_pipe(bc_pipe)
            w.add_pipe(bc_pipe.get_output_conn())

    env.run(until=100)

    print('Statistics:')
    stats = {}
    for w in witnesses:
        for m in w.messages:
            if m[1] in stats:
                stats[m[1]].append(m[0])
            else:
                stats[m[1]] = [m[0]]

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(stats)

    print('Order of messages:')
    for w in witnesses:
        msg = 'Witness ' + str(w.id) + ": "
        for m in w.messages:
            msg += m[1] + " "
        msg += '\n'
        print(msg)

    print('Voting')
    input = []
    for w in witnesses:
        input.append({"count": 1, "ballot": [[msg[1]] for msg in w.messages]})

    start = timer()
    output = SchulzeNPR(input, winner_threshold=len(stats), ballot_notation=SchulzeNPR.BALLOT_NOTATION_GROUPING).as_dict()
    end = timer()
    pp.pprint(output['order'])
    print(end - start)


if __name__ == "__main__":
    main()
