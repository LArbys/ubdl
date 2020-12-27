#!/bin/bash

# WE NEED TO SETUP ENV VARIABLES FOR ROOT, CUDA, OPENCV

MACHINE=`uname --nodename`

if [ $MACHINE == "trex" ]
then
    echo "SETUP TREX"
    source /usr/local/root/6.16.00_py3/bin/thisroot.sh

    export CUDA_HOME=/usr/local/cuda-10.0
    [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    export OPENCV_INCDIR=/usr/local/opencv/opencv-3.4.6/include
    export OPENCV_LIBDIR=/usr/local/opencv/opencv-3.4.6/lib

elif [ $MACHINE == "meitner" ]
then
    echo "SETUP MEITNER"

    source /usr/local/root6/bin/thisroot.sh

    export CUDA_HOME=/usr/local/cuda-10.0
    [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    export OPENCV_INCDIR=/usr/include
    export OPENCV_LIBDIR=/usr/local/lib    

elif [ $MACHINE == "goeppert" ]
then
    echo "SETUP GOEPPERT"
    source /usr/local/root/6.16.00_py3/bin/thisroot.sh

    export CUDA_HOME=/usr/local/cuda-10.0
    [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    export OPENCV_INCDIR=/usr/local/include
    export OPENCV_LIBDIR=/usr/local/lib    

elif [ $MACHINE == "mayer" ]
then
    echo "SETUP MAYER"
    source /usr/local/root/6.16.00_python3/bin/thisroot.sh

    export CUDA_HOME=/usr/local/cuda-10.0
    [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    export OPENCV_INCDIR=/usr/local/include
    export OPENCV_LIBDIR=/usr/local/lib    

elif [ $MACHINE == "blade" ]
then
    echo "SETUP TARITREE's RAZER BLADE"

    source /home/twongjirad/software/root6/6.14.02/bin/thisroot.sh

    export CUDA_HOME=/usr/local/cuda
    [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    export OPENCV_INCDIR=/usr/include
    export OPENCV_LIBDIR=/usr/local/lib    
    
else
    echo "DEFAULT SETUP (COMPAT WITH SINGULARITY CONTAINER)"
    source /usr/local/root/root-6.16.00/bin/thisroot.sh

    export CUDA_HOME=/usr/local/cuda-10.0
    [[ ":$LD_LIBRARY_PATH:" != *":${CUDA_HOME}/lib64:"* ]] && export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

    export OPENCV_INCDIR=/usr/include
    export OPENCV_LIBDIR=/usr/local/lib
    
fi

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
