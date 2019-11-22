import argparse
import zmq

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Receiver')
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

    print('Started worker {}'.format(args.index))

    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")

    socket.setsockopt_string(zmq.SUBSCRIBE, "")

    while True:
        string = socket.recv_string()
        print("Worker {} received message: {}\n".format(args.index, string))
