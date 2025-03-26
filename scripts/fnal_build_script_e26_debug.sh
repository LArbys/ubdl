#!/bin/bash

# This is assumed to run inside an SL7 container
#LARSOFT_VERSION=v10_04_06
QUALIFIER=e26:debug
#OUTPUT_DIR=/exp/uboone/app/users/tmw/ups_dev/products/ubdl/latest/Linux64bit+3.10-2.17_c14_prof

__ubdl_buildall_workdir__="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "QUALIFIER: ${QUALIFIER}"
echo "UBDL LOCATED AT: ${__ubdl_buildall_workdir__}"

source fnal_setenv_e26_debug.sh

# larlite
echo "<<< BUILD LARLITE >>>"
cd ${UBDL_BASEDIR}/larlite/
mkdir build
cd build
cmake -DUSE_PYTHON3=ON -DCMAKE_BUILD_TYPE=Debug ../
make install -j4
cd ${UBDL_BASEDIR}/larlite

# Geo2D
echo "<<< BUILD GEO2D >>>"
cd ${UBDL_BASEDIR}/Geo2D
source config/setup.sh
export GEO2D_CXX="${GEO2D_CXX} -g"
make
cd $__ubdl_buildall_workdir__

# LArOpenCV
echo "<<< BUILD LAROPENCV >>>"
cd ${UBDL_BASEDIR}/LArOpenCV
export LARLITE_CXX="${LARLITE_CXX} -g"
make
cd $__ubdl_buildall_workdir__

# LArCV
echo "<<< BUILD LARCV >>>"
cd ${UBDL_BASEDIR}/larcv
mkdir -p build
cd build
cmake -DUSE_PYTHON3=ON -DUSE_OPENCV=ON -DON_FNAL=ON -DUSE_TORCH=ON -DCMAKE_BUILD_TYPE=Debug ../
make install
cd $__ubdl_buildall_workdir__

# CILANTRO
echo "<<< BUILD CILANTRO >>>"
cd ${UBDL_BASEDIR}/cilantro
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Debug ../
make
cd $__ubdl_buildall_workdir__

# UBLARCVAPP
echo "<<< BUILD UBLARCVAPP >>>"
cd ${UBDL_BASEDIR}/ublarcvapp
mkdir -p build
source configure.sh
cd build
cmake -DUSE_OPENCV=ON -DCMAKE_BUILD_TYPE=Debug ../
make install
cd $__ubdl_buildall_workdir__

# LARFLOW
echo "<<< BUILD LARFLOW >>>"
cd ${UBDL_BASEDIR}/larflow
mkdir -p build
source configure.sh
cd build
cmake -DUSE_PYTHON3=ON -DCMAKE_BUILD_TYPE=Debug ../
make install
cd $__ubdl_buildall_workdir__


