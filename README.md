## How to run PoUW locally on your computer

This setup assumes you will be running a local paicoin node and two ML miners (on the same machine).
This is only for testing and debugging purposes.

1. If you already have the sources, please update to the latest version. 
    
    Otherwise, if you start from scratch, please clone the `pouw-adjustable-difficulty` branch of paicoin and the `distributed branch` of the Python project:
    ~~~~
    git clone -b "pouw-q4" --single-branch https://github.com/projectpai/paicoin.git
    git clone -b "master" --single-branch https://github.com/projectpai/pouw-main-iteration
    ~~~~

2. You can skip this step if you have `protobuf 3.6` installed. This is how it is installed using Homebrew:

    ~~~
    brew install protobuf@3.6
    brew link --force protobuf@3.6
    ~~~

3. If paicoin is already setup, delete the `testnet3` directory from `~/Application Support/PAIcoin/` if you are on a Mac (or the corresponding folder from Linux).

4. If this is the first time you set up the local testing environment, please copy the configuration files from `http://gitlab.int.oben.me/devops/pouw-docker/tree/master/configs` to `~/Application Support/PAIcoin/`.
The file paicoin.conf should have the following content:
    ~~~
    server=1
    bantime=1
    daemon=0
    rpcuser=paicoin
    rpcpassword=10050021
    rpcport=4002
    testnet=1
    rpcallowip=0.0.0.0/0
    txindex=1
    onlynet=ipv4
    listenonion=0
    maxtipage=31104000
    listen=1
    rpcbind=0.0.0.0
    verificationserver=0.0.0.0:50011
    printtoconsole=1
    connect=0
    ignore-not-connected=1
    dnsseed=0

    ~~~

5. Configure the PAIcoin build by running:
    ~~~~
    cd paicoin/
    ./autogen.sh
    ./configure --with-gui=no --disable-tests --disable-bench --enable-chainparams-conf
    ~~~~
    
6. Build the code:

    ~~~~
    make -j5
    ~~~~

7. Create the genesis block.
    ~~~~
    cd src/
    ./paicoind -mine-genesis-block
    ~~~~

8. Switch to the Python code and install the ML trainer:
    ~~~~
    cd ../../main-iteration/
    python3 setup.py develop
    ~~~~

9. Start the verification server:
    ~~~~
    python3 pai/pouw/verification/server.py
    ~~~~

10. Start the *paicoind* process:
    ~~~~
    cd ../../paicoin/src/
    ./paicoind -ignore-not-connected
    ~~~~

11. Start the cluster with 3 nodes:
    ~~~~
    cd ../../../main-iteration/
    python3 pai/pouw/start_cluster.py --nodes-number 3
    ~~~~

12. Run the client that starts the training process:
    ~~~~
    python3 pai/pouw/nodes/client.py --client-task-definition-path=../../client-task-definition.yaml
    ~~~~
    
13. You can open a new Terminal and check with `paicoin-cli` the blockchain status. E.g.:
    ~~~~
    cd ../../paicoin/src/
    ./paicoin-cli --rpcuser=paicoin --rpcpassword=10050021 --rpcport=4002 getmininginfo
    ~~~~
    
14. While the mining and training is taking place, you can see the verification process in the output window of `server.py`,
while the training/mining status can be checked in the output window of `start_cluster.py`.

