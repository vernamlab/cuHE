#!/usr/bin/env bash

CUHE_DOCKER_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CUHE_HOME_PATH=$CUHE_DOCKER_PATH/../

case "$1" in
  build)
    nvidia-docker -D build -t="centos-cuhe:latest" .
  ;;
  run)
    #remember to start nvidia-docker-plugin
    nvidia-docker run --rm -i -v $CUHE_HOME_PATH:/home/sources:rw -t "centos-cuhe:latest"
  ;;
  *)
    echo "Usage [ build, run ] "
    exit 0
  ;;
esac