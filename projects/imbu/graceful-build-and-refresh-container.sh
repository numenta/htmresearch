#!/bin/bash

mkdir -p `pwd`/cache
pushd ../../
docker ${IMBU_DOCKER_OPTIONS} stop imbu
docker ${IMBU_DOCKER_OPTIONS} rm imbu
docker ${IMBU_DOCKER_OPTIONS} build -t imbu:latest -f projects/imbu/Dockerfile .
docker ${IMBU_DOCKER_OPTIONS} run \
  --name imbu \
  -d \
  -p 8080:80 \
  -e IMBU_LOAD_PATH_PREFIX=${IMBU_LOAD_PATH_PREFIX} \
  -e CORTICAL_API_KEY=${CORTICAL_API_KEY} \
  -e IMBU_RETINA_ID=${IMBU_RETINA_ID} \
  ${IMBU_DOCKER_EXTRAS} \
  imbu:latest
popd
