import json
import os.path

import grpc
from pai.pouw.mining.utils import BLOCK_DROP_LOCATION
# import the generated classes
from pai.pouw.verification.verifier_pb2 import Request
from pai.pouw.verification.verifier_pb2_grpc import VerifierStub

if __name__ == '__main__':
    # open a gRPC channel
    channel = grpc.insecure_channel('localhost:50011')

    for block_id in os.listdir(BLOCK_DROP_LOCATION):
        # create a stub (client)
        stub = VerifierStub(channel)
        # load the block
        block_id = str(block_id)
        block = json.load(open(os.path.join(BLOCK_DROP_LOCATION, block_id), "r"))

        # extract the relevant fields
        msg_id = block['msg_id']
        model_hash = block['model_hash']
        msg_next_id = block['msg_next_id']

        # build the request
        req = Request(msg_id=msg_id, model_hash=model_hash,
                      msg_next_id=msg_next_id)

        # make the call
        response = stub.Verify(req)
        print('********\nBlock: {}\nMessage id: {}'.format(block_id, msg_id))
        print(response.code)
        print(response.description)
