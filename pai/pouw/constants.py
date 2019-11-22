import os.path

PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
DATA_DIRECTORY = os.path.join(PACKAGE_ROOT, 'data')
OUTPUT_DIRECTORY = os.path.join(PACKAGE_ROOT, 'output')
OMEGA = (10 ** (-12))
TEMP_FOLDER = '/tmp/pai-pouw/'

BUCKET = None
BUCKET_ENV_VAR = "POUW_BUCKET"
if BUCKET_ENV_VAR in os.environ:
    BUCKET = os.environ[BUCKET_ENV_VAR]
else:
    BUCKET = os.path.join(PACKAGE_ROOT, 'bucket')

GENESIS_NONCE = 2083236893
GENESIS_TARGET = 0x207fffff  # add this instead of the original 0x1d00ffff in order to get easily new blocks

# Ensure existence of output directory
os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)

FREE_NODES_LISTENING_CHANNEL = 'pouw_free_nodes_v2'
CLIENT_TASK_CHANNEL = 'pouw_task_requests_v2'

MIN_MEMBERS_NUM = 3

WAIT_TIME_AFTER_MINIMAL_NUMBER_OF_NODES_HAS_REGISTERED = 10
MAX_NODE_VOTE_RESULTS_WAIT_TIME_IN_SECONDS = 60
NUMBER_OF_DATASET_SEGMENTS = 10

SUPERVISORY_COMMITTEE_SIZE = 0

BLOCK_COMMITMENT_INERATIONS_ANNOUNCED = 5
