#!/bin/bash

# OPENCV
export OPENCV_INCDIR=/usr/local/include
export OPENCV_LIBDIR=/usr/local/lib

# LIBTORCH
export LIBTORCH_DIR="/usr/local/lib/python2.7/dist-packages/torch"
export LIBTORCH_LIBDIR="/usr/local/lib/python2.7/dist-packages/torch/lib"
export LIBTORCH_INCDIR="/usr/local/lib/python2.7/dist-packages/torch/lib/include"
[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}:"* ]] && \
    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"
