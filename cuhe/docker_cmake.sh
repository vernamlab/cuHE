#!/usr/bin/env bash

case $1 in
    ''|*[!0-9]*)
      echo "Specify your GPU Compute Capabilities (check: https://developer.nvidia.com/cuda-gpus)"
      exit 0
    ;;
    *)
      cmake ./ -DGCC_CUDA_VERSION:STRING=gcc -DGPU_ARCH:STRING=$1
    ;;
esac
