#!/bin/bash

# Note locations here are some defaults that typically work for ubuntu.
# they are also the locations used in the [ubdl container](https://github.com/larbys/larbys-containers)

# ROOT
# SETUP BY UPS

setup root v6_12_06a -q e17:prof


# CUDA
# NO CUDA

# OPENCV
setup opencv v3_1_0 -q e17
setup python v2_7_14b
setup cmake v3_13_2
setup numpy v1_14_3 -q e17:p2714b:openblas:prof
setup libtorch v1_0_1 -q e17:prof
setup eigen v3_3_4a
setup boost v1_66_0a -q e17:prof

# SETUP BY UPS, need slightly different version of variables for ubdl repos
export OPENCV_INCDIR=${OPENCV_INC}
export OPENCV_LIBDIR=${OPENCV_LIB}

# LIBTORCH
# setup by UPS
export LIBTORCH_LIBDIR=${LIBTORCH_LIB}
export LIBTORCH_INCDIR=${LIBTORCH_INC}

