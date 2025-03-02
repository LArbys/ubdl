#!/bin/bash

alias python=python3
alias python-config=python3-config

__ubdl_buildall_py3_workdir__=$PWD
build_log=${__ubdl_buildall_py3_workdir__}/build.log

#BUILD_FLAG=$1

echo "<<< BUILD LARLITE >>>"
cd larlite
mkdir build
cd build
cmake -DUSE_PYTHON3=ON ../  || { echo "larlite cmake setup failed"; exit 1; }
#make install -j4 >> ${build_log} 2>&1 || { echo "larlite build failed"; exit 1; }
make install -j4 || { echo "larlite build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "<<< BUILD GEO2D >>>"
cd Geo2D
source config/setup.sh
#make -j4 >> ${build_log} 2>&1 || { echo "Geo2D build failed"; exit 1; }
make -j4 || { echo "Geo2D build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "<<< BUILD LAROPENCV >>>"
cd LArOpenCV
#make -j4 >> ${build_log} 2>&1 || { echo "LArOpenCV build failed"; exit 1; }
make -j4 || { echo "LArOpenCV build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "<<< BUILD LARCV >>>"
cd larcv
mkdir -p build
cd build
cmake -DUSE_PYTHON3=ON -DUSE_OPENCV=ON -DUSE_FNAL=ON -DUSE_TORCH=OFF ../ || { echo "larcv cmake setup failed"; exit 1; }
make install -j4 || { echo "larcv build failed"; exit 1; }
#make install -j4 >> ${build_log} 2>&1 || { echo "larcv build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "<<< BUILD CILANTRO >>>"
cd cilantro
mkdir -p build
cd build
cmake ../
make -j4 || { echo "cilantro build failed"; exit 1; }
#make -j4 >> ${build_log} 2>&1 || { echo "cilantro build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "<<< BUILD UBLARCVAPP >>>"
mkdir -p ublarcvapp/build
cd ublarcvapp
source configure.sh
cd build
cmake -DUSE_OPENCV=ON ../
make install -j4 >> ${build_log} 2>&1 || { echo "ublarcvapp build failed"; exit 1; }
make install -j4 || { echo "ublarcvapp build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "<<< BUILD LARFLOW >>>"
mkdir -p larflow/build
cd larflow
source configure.sh
cd build
cmake -DUSE_PYTHON3=ON ../
make install -j4 || { echo "larflow build failed"; exit 1; }
#make install -j4 >> ${build_log} 2>&1 || { echo "larflow build failed"; exit 1; }
cd $__ubdl_buildall_py3_workdir__

echo "built ubdl modules"
