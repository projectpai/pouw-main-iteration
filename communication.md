# ML Training algorithm

All messages send over network will be string which can be deserialized
 using yaml format

- client loads task definition. At the moment training task is defined at
client-task-definition.yaml . There also support for loading other task
definitions by specifying filepath towards them

- Client serializes request to string which then sends it to P2P messaging system
NEW_ML_TASK message type
- After this client starts listening to P2P messaging system for cluster response
CLUSTER_READY message type

## Cluster creation
- Free nodes are listening on P2P messaging system, once they get NEW_ML_TASK,
they send P2P INIT_CLUSTER message type, which contains information if their redis port is open to public and their initial vote
- We have a waiting period for all nodes to respond (10-20 minutes?)
- All nodes with open ports are considered for committee
- If there are more committee candidates then max committee size, we select
all nodes whose voices are in minority (votes can be 1 or 0). If number of candidates is still too big, we send P2P COMMITTEE_VOTING message which contains voting cycle information and vote value, this process is repeated until number of candidates is less or equal to max committee size
- Committee members then redis ping one another and send ping time values from
them to other members using P2P CLUSTER_DATABASE_SELECTION message type
- using boarda algorithm we select two highest ranked nodes. First is setup as master
and second as slave
- Cluster sends to client P2P message CLUSTER_READY

## Dataset selection
- After client receives CLUSTER_READY message, it sends P2P DATASET_SELECTION message
which contains ten hashes of dataset segments
- Cluster then performs boarda voting algorithm on hashes and returns them Client
using P2P HASH_VOTES message
- Client then sends P2P DATASET_SEGMENT message which contains hash and a bucket link for
segment download

## training
- Cluster then distributes segments to nodes and starts training using newly created
redis cluster
- Upon completion cluster sends to client P2P message TRAINING_COMPLETED which contains
model accuracy and links on the bucket for their download
- Cluster is then disbanded and rewards is distributed to nodes 
