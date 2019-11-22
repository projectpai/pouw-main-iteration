# Verifier client for RPC server

This is a C++ demo project to get started with the RPC client for the verification server.

These are Mac OS instructions. For Linux, please consult the official documentation.

### Prerequisites

To build the project you must have gRPC prerequisites installed.

```
[sudo] xcode-select --install
brew install autoconf automake libtool shtool
brew install gflags
```

### Installing

```
git clone -b $(curl -L http://grpc.io/release) https://github.com/grpc/grpc
cd grpc
git submodule update --init
mkdir bin
cd bin

cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DgRPC_PROTOBUF_PROVIDER=package -DgRPC_ZLIB_PROVIDER=package -DgRPC_CARES_PROVIDER=package -DgRPC_SSL_PROVIDER=package -DCMAKE_BUILD_TYPE=Release -DOPENSSL_ROOT_DIR=/usr/local/Cellar/openssl/1.0.2p -DOPENSSL_LIBRARIES=/usr/local/Cellar/openssl/1.0.2p/lib ..

make -j4
sudo make install
```

### Development

After installing gRPC, you can build the project in the IDE of your choice (e.g. CLion).
`verifier_client.cc` contains the minimum code to start using the verification server. The interface is found in `../verifier.proto`.


### Test
You must first start `../server.py` from the parent directory, then test with `./verifier_client`.