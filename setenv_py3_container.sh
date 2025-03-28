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

export TORCH_CMAKE_DIR=`python3 -c "import os,torch; print(os.path.dirname(torch.__file__))"`/share/cmake/Torch
[[ ":${CMAKE_MODULE_PATH}:" != *":${TORCH_CMAKE_DIR}:"* ]] && export CMAKE_MODULE_PATH="${TORCH_CMAKE_DIR}:${CMAKE_MODULE_PATH}"

