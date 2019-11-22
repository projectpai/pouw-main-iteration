## How to run the mini-POC for ZMQ

1. Start the cluster of receivers (the default is 2), but here is an example with 3:
````
python3 start_cluster.py --nodes-number 3
````

2. Run the sender script, tha sends 10 messages to everyone:
````
python3 sender_worker.py
````

3. Look in the console output of the `start_cluster.py` script for results.