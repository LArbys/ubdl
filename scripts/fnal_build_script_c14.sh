#!/bin/bash

# This is assumed to run inside an SL7 container
#LARSOFT_VERSION=v10_04_06
QUALIFIER=c14:prof
#OUTPUT_DIR=/exp/uboone/app/users/tmw/ups_dev/products/ubdl/latest/Linux64bit+3.10-2.17_c14_prof

__ubdl_buildall_workdir__="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "QUALIFIER: ${QUALIFIER}"
echo "UBDL LOCATED AT: ${__ubdl_buildall_workdir__}"

# Setup products and evironment variables
setup cmake v3_27_4
setup clang v14_0_6c
setup root v6_28_12 -q c14:p3915:prof
setup python v3_9_15
setup opencv v3_4_16_nogui -q e26:p3915:prof
setup numpy v1_24_3 -q c14:p3915
setup tbb v2021_9_0 -q c14
setup libtorch v2_1_1b -q c14
setup eigen v23_08_01_66e8f
setup boost v1_82_0 -q c14:prof

# OPENCV ENVIRONMENT
# setup by ups, need slightly different version of variables for ubdl repos
export OPENCV_INCDIR=${OPENCV_INC}
export OPENCV_LIBDIR=${OPENCV_LIB}

# LIBTORCH
# setup by UPS
export LIBTORCH_LIBDIR=${LIBTORCH_LIB}
export LIBTORCH_INCDIR=${LIBTORCH_INC}

cd ${__ubdl_buildall_workdir__}/../
source configure.sh

# larlite
echo "<<< BUILD LARLITE >>>"
cd ${UBDL_BASEDIR}/larlite/
mkdir build
cd build
cmake -DUSE_PYTHON3=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ../
make install -j4
cd ${UBDL_BASEDIR}/larlite

# Geo2D
echo "<<< BUILD GEO2D >>>"
cd ${UBDL_BASEDIR}/Geo2D
source config/setup.sh
make
cd $__ubdl_buildall_workdir__

# LArOpenCV
echo "<<< BUILD LAROPENCV >>>"
cd ${UBDL_BASEDIR}/LArOpenCV
make
cd $__ubdl_buildall_workdir__

# LArCV
echo "<<< BUILD LARCV >>>"
cd ${UBDL_BASEDIR}/larcv
mkdir -p build
cd build
cmake -DUSE_PYTHON3=ON -DUSE_OPENCV=ON -DON_FNAL=ON -DUSE_TORCH=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ../
make install
cd $__ubdl_buildall_workdir__

# CILANTRO
echo "<<< BUILD CILANTRO >>>"
cd ${UBDL_BASEDIR}/cilantro
mkdir -p build
cd build
cmake -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ../
make
cd $__ubdl_buildall_workdir__

# UBLARCVAPP
echo "<<< BUILD UBLARCVAPP >>>"
cd ${UBDL_BASEDIR}/ublarcvapp
mkdir -p build
source configure.sh
cd build
cmake -DUSE_OPENCV=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ../
make install
cd $__ubdl_buildall_workdir__

# LARFLOW
echo "<<< BUILD LARFLOW >>>"
cd ${UBDL_BASEDIR}/larflow
mkdir -p build
source configure.sh
cd build
cmake -DUSE_PYTHON3=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ ../
make install
cd $__ubdl_buildall_workdir__

#mkdir -p ${OUTPUT_DIR}/larlite/python/
#rsync -av --progress --exclude='*.git' python/larlite ${OUTPUT_DIR}/larlite/python/

#rsync -av --progress --exclude='*.git' python/colored_msg ${OUTPUT_DIR}/larlite/python/

# -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++

#echo "<<< copy table file >>>"
#cp -r ${UBDL_BASEDIR}/ups ${OUTPUT_DIR}/





