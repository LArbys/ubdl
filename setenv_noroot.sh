#!/bin/bash

# WE NEED TO SETUP ENV VARIABLES FOR ROOT, CUDA, OPENCV
alias python=python3
MACHINE=`uname --nodename`
if [ -z ${SINGULARITY_NAME} ]
then
    echo "not inside singularity container"
else
    echo "inside singularity container: ${SINGULARITY_CONTAINER}"
    MACHINE=${SINGULARITY_NAME}
fi


export CUDA_HOME=/usr/local/cuda/
[[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export OPENCV_INCDIR=/usr/include/opencv4/
export OPENCV_LIBDIR=/usr/lib/x86_64-linux-gnu/
export JUPYTER_CONFIG_DIR=~/.local/etc/jupyter
    
cd lardly
source setenv.sh
cd ..

export UBDL_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
export LARFLOW_BASEDIR=$UBDL_BASEDIR/larflow

# LIBTORCH
# location below is typically where running `pip install torch` will put pytorch
# export LIBTORCH_DIR="/usr/local/lib/python2.7/dist-packages/torch"
#export LIBTORCH_DIR="/home/jmills/.local/lib/python3.5/site-packages/torch"
#export LIBTORCH_LIBDIR=${LIBTORCH_DIR}/lib
#export LIBTORCH_INCDIR=${LIBTORCH_DIR}/lib/include
#export LIBTORCH_DIR="/usr/local/torchlib"
#export LIBTORCH_LIBDIR=${LIBTORCH_DIR}/lib
#export LIBTORCH_INCDIR=${LIBTORCH_DIR}/include
#[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBDIR}:"* ]] && \
#    export LD_LIBRARY_PATH="${LIBTORCH_LIBDIR}:${LD_LIBRARY_PATH}"
