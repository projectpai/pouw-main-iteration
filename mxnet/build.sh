#!/bin/bash

if [ "$#" -ne 1 ]
then
    echo "Usage: build.sh <version>"
    exit -1
fi
cd `dirname $0` && \
docker image build --tag mxnet-build --file Dockerfile.build . && \
docker container run -e VERSION=$1 --rm -v $PWD:/mxnet-dist -it mxnet-build
