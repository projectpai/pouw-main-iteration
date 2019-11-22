- Client loads a request task definition file and sends it to redis task listening channel
- Interested nodes create task_id by doing sha256 hash and then inform client about it
- All interested nodes join on new channel (training_task_TASK_ID)
- After waiting for predetermined amount of time and if there are enough workers,
task request is removed from queue
- Worker nodes start committee selection procedure using training_task_TASK_ID
- After committee is selected, they inform client using client redis listening channel
from task request
- Client sends list of 10 hashes to cluster listening address which was provided in
committee response
- Committee votes for hashes using Borda algorithm
- List of hashes is send to Client
- Client transmits dataset to committee using segment order based on hash HASH_VOTES
- Committee distributes segments between workers
- Training starts
- each node reports to client training results and goes back to waiting for work
- after all nodes have completed training, client downloads model with highest accuracy
