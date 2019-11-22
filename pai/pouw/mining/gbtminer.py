import argparse
import os.path
import sys
import time

from binascii import hexlify, unhexlify
from bitcoinrpc.authproxy import AuthServiceProxy, JSONRPCException
from bitcoinrpc.config import read_default_config
from pai.pouw.mining.blkmaker import blktemplate
from pai.pouw.mining.blkmaker.blkmaker import double_sha256, sha256_hexdigest, sha256_digest
from pai.pouw.mining.utils import serialize_local_message_map


def create_miner_based_on_cmd_parameters():
    parser = argparse.ArgumentParser(description='Real PAICoin blockchain miner code stub')
    parser.add_argument('--server-ip', type=str, default='127.0.0.1',
                        help='PAICoin server IP (default is 127.0.0.1)')
    parser.add_argument('--paicoin-cfg-file', type=str, default=None,
                        help='Filepath to PAICoin configuration file containing rpcport, rpcuser and rpcpassword')
    opt = parser.parse_args()
    miner = Miner(5, opt.server_ip, opt.paicoin_cfg_file)
    return miner


class Block:
    def __init__(self, template):
        (data, dataid) = template.get_data()
        self.header = data[:76]
        self.hexdata = template.get_blkhex(b'', dataid)
        self.target = template.target


class Miner:
    def __init__(self, iterations_announced, server_ip, paicoin_cfg_file):
        self._pai_address = None
        self._server_ip = server_ip

        if paicoin_cfg_file is None:
            if sys.platform.startswith('darwin'):
                paicoin_cfg_file = os.path.expanduser('~/Library/Application Support/PAIcoin/paicoin.conf')
            else:
                paicoin_cfg_file = os.path.join(os.path.expanduser("~"), '.paicoin', 'paicoin.conf')
        if not os.path.isfile(paicoin_cfg_file):
            raise RuntimeError('Path toward paicoin.conf must be provided for node to work')
        cfg = read_default_config(paicoin_cfg_file)

        self._server_port = int(cfg.get('rpcport', 4002))
        self._rpc_user = cfg.get('rpcuser', 'paicoin')
        self._rpc_password = cfg.get('rpcpassword', '')
        self._template = None
        self._template_refresh_interval = 5
        self._iterations_announced = iterations_announced
        self._blocks = []

    @property
    def _rpc_connection(self):
        return AuthServiceProxy("http://%s:%s@%s:%d"
                                % (self._rpc_user, self._rpc_password,
                                   self._server_ip, self._server_port))

    @staticmethod
    def _check_nonce(blkhash, target):
        for i in range(32):
            if blkhash[31 - i] > target[i]:
                return False
            if blkhash[31 - i] < target[i]:
                return True
        return True

    def _get_block_template(self):
        if self._pai_address is None:
            self._pai_address = self._rpc_connection.getaccountaddress("miner")

        height = self._template.height if self._template is not None else 0
        self._template = blktemplate.Template()
        gbt_params = self._template.request(self._pai_address)['params'][0]
        gbt_resp = self._rpc_connection.getblocktemplate(gbt_params)
        self._template.add(gbt_resp)

        # invalidate old announcements
        if self._template.height > height:
            self._blocks = []

    # calculate nonce from pouw values
    @staticmethod
    def calculate_nonce(end_it_model_hash, local_message_map):
        serialized_local_weights = serialize_local_message_map(local_message_map)
        nonce_input = end_it_model_hash + serialized_local_weights
        nonce_precursor = sha256_digest(nonce_input)
        return sha256_digest(nonce_precursor)[:4]

    @staticmethod
    def calculate_zero_nonce_hash(block_header_hex):
        block_header = unhexlify(block_header_hex)[:76]
        return hexlify(double_sha256(block_header)).decode('ascii')

    # build zero-nonce block and return its hash
    def announce_new_block(self):
        if (self._template is None) or (self._template.version is None) or (
                time.time() - self._template.curtime > self._template_refresh_interval):
            self._get_block_template()

        block = Block(self._template)
        self._blocks.append(block)
        return hexlify(double_sha256(block.header)).decode('ascii')

    # check nonce for announced block and return block hex data and calculated nonce
    def mine_announced_block(self, msg_id, message_history_id, end_it_model_hash, local_message_map):
        if len(self._blocks) <= self._iterations_announced:
            return False, None

        while len(self._blocks) > self._iterations_announced + 1:
            self._blocks.pop(0)

        block = self._blocks.pop(0)

        nonce = self.calculate_nonce(end_it_model_hash, local_message_map)
        pouw = (len(message_history_id)).to_bytes(1, 'little') + message_history_id + \
               (len(msg_id)).to_bytes(1, 'little') + msg_id

        # add nonce and pouw fields
        header = block.header + nonce + pouw

        # test block hash
        blkhash = double_sha256(header)
        if not self._check_nonce(blkhash, block.target):
            return False, nonce

        hexdata = hexlify(header).decode('ascii')
        hexdata += block.hexdata
        return hexdata, nonce

    def submit_block(self, hexdata):
        err = self._rpc_connection.submitblock(hexdata, "dummy")
        self._template = None
        if err:
            raise RuntimeError("submitblock error: %s" % err)


def main():
    miner = create_miner_based_on_cmd_parameters()

    pow_msg_id = b'1234'
    pow_message_history_id = b''
    pow_end_it_model_hash = double_sha256(b'5678')
    pow_local_message_map = [1, 2, 3, 4]

    try:
        for it in range(0, miner._iterations_announced):
            miner.announce_new_block()
            hexdata, nonce = miner.mine_announced_block(pow_msg_id, pow_message_history_id, pow_end_it_model_hash,
                                                        pow_local_message_map)
            assert hexdata is False
            assert nonce is None

        for it in range(0, 1000):
            miner.announce_new_block()
            hexdata, nonce = miner.mine_announced_block(pow_msg_id, pow_message_history_id, pow_end_it_model_hash,
                                                        pow_local_message_map)
            if hexdata:
                miner.submit_block(hexdata)
                print(f"submit_block: nonce = {hexlify(nonce).decode('latin1')}")

    except JSONRPCException as e:
        print(e.error['message'])
    except Exception as e:
        print(e)
    return


if __name__ == '__main__':
    main()
