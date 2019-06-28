#!/bin/bash

# LOCATIONS FOR UBDL CONTAINER (FOR USE ON TUFTS TYPICALLY)

# ROOT
source /usr/local/root/build/bin/thisroot.sh

# CUDA
# typical location of cuda in ubuntu
export CUDA_HOME=/usr/local/cuda-10.0
[[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# OPENCV
export OPENCV_INCDIR=/usr/local/include
export OPENCV_LIBDIR=/usr/local/lib

# LIBTORCH
PYTHON_VERSION=`python -c 'import sys; print(sys.version_info[0])'`
echo "SETUP FOR PYTHON ${PYTHON_VERSION}"
if [ $PYTHON_VERSION='3' ]
then
    # location below is typically where running `pip install torch` will put pytorch
    export LIBTORCH_DIR="/usr/local/lib/python3.5/dist-packages/torch"
    export LIBTORCH_LIBDIR=${LIBTORCH_DIR}/lib
    export LIBTORCH_INCDIR=${LIBTORCH_DIR}/lib/include
else
    export LIBTORCH_DIR="/usr/local/lib/python3.5/dist-packages/torch"
    export LIBTORCH_LIBDIR=${LIBTORCH_DIR}/lib
    export LIBTORCH_INCDIR=${LIBTORCH_DIR}/lib/include
fi
       

[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}:"* ]] && \
    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"
