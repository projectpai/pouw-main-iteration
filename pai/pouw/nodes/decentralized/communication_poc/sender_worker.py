import zmq
import time

if __name__ == '__main__':
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    for i in range(10):
        socket.send_string("message id %d" % i)
        time.sleep(1)

    print('Finished sending all messages.')
