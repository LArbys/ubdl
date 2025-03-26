#!/bin/bash

# This is assumed to run inside an SL7 container
#LARSOFT_VERSION=v10_04_07
#QUALIFIER=e26:debug
#echo "QUALIFIER: ${QUALIFIER}"

# Setup cvmfs for uboone
source /cvmfs/uboone.opensciencegrid.org/products/setup_uboone.sh

__ubdl_buildall_workdir__="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
#echo "THIS SCRIPT LOCATED AT: ${__ubdl_buildall_workdir__}"

# NOTE: currently setup to build with dependencies of uboonecode v10_04_07

# Setup products and evironment variables
setup cmake v3_27_4
setup gcc v12_1_0
setup root v6_28_12 -q e26:p3915:debug
setup python v3_9_15
setup opencv v3_4_16_nogui -q e26:p3915:prof
setup numpy v1_24_3 -q e26:p3915
setup tbb v2021_9_0 -q e26
setup libtorch v2_1_1b -q e26
setup eigen v23_08_01_66e8f
setup boost v1_82_0 -q e26:debug

# SETUP BY UPS, need slightly different version of variables for ubdl repos
export OPENCV_INCDIR=${OPENCV_INC}
export OPENCV_LIBDIR=${OPENCV_LIB}

# LIBTORCH
# setup by UPS
export LIBTORCH_LIBDIR=${LIBTORCH_LIB}
export LIBTORCH_INCDIR=${LIBTORCH_INC}

cd ${__ubdl_buildall_workdir__}/../
source configure.sh

cd ${__ubdl_buildall_workdir__}
