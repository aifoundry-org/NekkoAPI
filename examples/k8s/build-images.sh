#!/bin/sh
#

docker build -t nekko-api:latest -f examples/k8s/Dockerfile .

mkdir -p tmp && \
cd tmp && \
git clone --depth 1 https://github.com/aifoundry-org/storage-manager.git && \
cd storage-manager && \
docker build -t nekko-sm:latest .


