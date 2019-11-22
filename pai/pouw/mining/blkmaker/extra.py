from binascii import a2b_hex as __a2b_hex

MAX_BLOCK_VERSION = 2
SIZEOF_WORKID = 8

try:
    __a2b_hex('aa')
    _a2b_hex = __a2b_hex
except TypeError:
    def _a2b_hex(a):
        return __a2b_hex(a.encode('ascii'))


class _Transaction:
    def __init__(self, txnj={}):
        if txnj is None:
            return
        if 'data' not in txnj:
            raise ValueError("Missing or invalid type for transaction data")
        self.data = _a2b_hex(txnj['data'])


def _request(jcaps, address=None, lpid=None):
    params = {
        'capabilities': jcaps,
        'maxversion': MAX_BLOCK_VERSION,
    }
    if (address):
        params['address'] = address
    if lpid:
        params['longpollid'] = lpid
    req = {
        'id': 0,
        'method': 'getblocktemplate',
        'params': [params],
    }
    return req
