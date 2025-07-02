#!/bin/bash

# WE NEED TO SETUP ENV VARIABLES FOR ROOT, CUDA, OPENCV
alias python=python3

#Force use of container options
#(needed if building ubdl in container recipe on one of the named machines in setenv_py3.sh)

echo "DEFAULT SETUP (COMPAT WITH SINGULARITY CONTAINER)"
source /usr/local/root/bin/thisroot.sh

export CUDA_HOME=/usr/local/cuda/
[[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

export OPENCV_INCDIR=/usr/include
export OPENCV_LIBDIR=/usr/local/lib

# Build against libtorch built with c++11 ABI standard
# This is different than the version in python3.8
# Careful about mixing these up.
export LIBTORCH_DIR=/usr/local/libtorch1.9.0_cxx11abi/libtorch
export LIBTORCH_LIBRARY_DIR=${LIBTORCH_DIR}/lib
export LIBTORCH_CMAKE_DIR=${LIBTORCH_DIR}/share/cmake/Torch
export LIBTORCH_BIN_DIR=${LIBTORCH_DIR}/bin

[[ ":$LD_LIBRARY_PATH:" != *":${LIBTORCH_LIBRARY_DIR}:"* ]] && LD_LIBRARY_PATH="${LIBTORCH_LIBRARY_DIR}:${LD_LIBRARY_PATH}"
[[ ":$PATH:" != *":${LIBTORCH_BIN_DIR}:"* ]] && PATH="${LIBTORCH_BIN_DIR}:${PATH}"

# Add prongCNN folder
export PRONGCNN_DIR=/home/twongjirad/working/larbys/gen2/container_u20_env/work/prongCNN/
export LARPID_DIR=/home/twongjirad/working/larbys/gen2/container_u20_env/work/prongCNN/larpid/build/installed/
export LARPID_LIBDIR=${LARPID_DIR}/lib
export LARPID_INCDIR=${LARPID_DIR}/include

[[ ":$LD_LIBRARY_PATH:" != *":${LARPID_LIBDIR}:"* ]] && export LD_LIBRARY_PATH="${LARPID_LIBDIR}:${LD_LIBRARY_PATH}"
[[ ":$LD_LIBRARY_PATH:" != *":${LARPID_LIBDIR}:"* ]] && export LD_LIBRARY_PATH="${LARPID_LIBDIR}:${LD_LIBRARY_PATH}"
[[ ":$PATH:" != *":${LARPID_BINDIR}:"* ]] && export PATH="${LARFLOW_BINDIR}:${PATH}"
[[ ":${PYTHONPATH}:" != *":${PRONGCNN_DIR}:"* ]] && export PYTHONPATH="${PRONGCNN_DIR}:${PYTHONPATH}"
[[ ":${PYTHONPATH}:" != *":${PRONGCNN_DIR}/models:"* ]] && export PYTHONPATH="${PRONGCNN_DIR}/models:${PYTHONPATH}"


