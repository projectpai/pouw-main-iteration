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

First we install a local version of gRPC (this is for Ubuntu systems):
```
export MY_INSTALL_DIR=/Volumes/blackbox/Archive/uar/pouw-main-iteration/pai/pouw/verification/verifier_client/local
git clone --recurse-submodules -b v1.30.2 https://github.com/grpc/grpc
cd grpc
mkdir -p cmake/build
pushd cmake/build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$MY_INSTALL_DIR \
      ../..
make -j
make install
popd
```

### Development
Go to the verifier_client folder and run the following commands:
```
mkdir -p cmake/build
$ pushd cmake/build
$ cmake -DCMAKE_PREFIX_PATH=$MY_INSTALL_DIR ../..
$ make -j
```

After installing gRPC, you can build the project in the IDE of your choice (e.g. CLion).
`verifier_client.cc` contains the minimum code to start using the verification server. The interface is found in `../verifier.proto`.


### Test
You must first start `../server.py` from the parent directory, then test with `./verifier_client`.