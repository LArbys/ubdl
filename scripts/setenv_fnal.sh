#!/bin/bash

# Note locations here are some defaults that typically work for ubuntu.
# they are also the locations used in the [ubdl container](https://github.com/larbys/larbys-containers)
source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone_mcc9.sh

# ROOT
# SETUP BY UPS

setup root v6_28_12 -q e26:p3915:prof
setup gcc v12_1_0


# CUDA
# NO CUDA

# OPENCV
setup python v3_9_15
setup opencv v3_4_16_nogui -q e26:p3915:prof
setup numpy v1_24_3 -q e26:p3915
setup tbb v2021_9_0 -q e26
setup libtorch v2_1_1b -q e26
setup eigen v23_08_01_66e8f
setup boost v1_82_0 -q e26:prof

# SETUP BY UPS, need slightly different version of variables for ubdl repos
export OPENCV_INCDIR=${OPENCV_INC}
export OPENCV_LIBDIR=${OPENCV_LIB}

# LIBTORCH
# setup by UPS
export LIBTORCH_LIBDIR=${LIBTORCH_LIB}
export LIBTORCH_INCDIR=${LIBTORCH_INC}

