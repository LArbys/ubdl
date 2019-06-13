#!/bin/bash

# Note locations here are some defaults that typically work for ubuntu.
# they are also the locations used in the [ubdl container](https://github.com/larbys/larbys-containers)

# ROOT
# SETUP BY UPS

# CUDA
# NO CUDA

# OPENCV
# SETUP BY UPS, need slightly different version of variables for ubdl repos
export OPENCV_INCDIR=${OPENCV_INC}
export OPENCV_LIBDIR=${OPENCV_LIB}

# LIBTORCH
# setup by UPS
export LIBTORCH_LIBDIR=${LIBTORCH_LIB}
export LIBTORCH_INCDIR=${LIBTORCH_INC}

