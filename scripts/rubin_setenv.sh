#!/bin/bash

# Specific locations for Tufts Rubin machine

# ROOT
#source /usr/local/root/build/bin/thisroot.sh

# CUDA
# typical location of cuda in ubuntu
export CUDA_HOME=/usr/local/cuda-10.0
[[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# OPENCV
export OPENCV_INCDIR=/usr/local/include
export OPENCV_LIBDIR=/usr/local/lib

# LIBTORCH
# location below is typically where running `pip install torch` will put pytorch
export LIBTORCH_DIR="/usr/local/lib/python2.7/dist-packages/torch"
export LIBTORCH_LIBDIR=${LIBTORCH_DIR}/lib
export LIBTORCH_INCDIR=${LIBTORCH_DIR}/lib/include

