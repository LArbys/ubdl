#!/bin/bash

__ubdl_buildall_py2_startdir__=$PWD
__ubdl_buildall_py2_workdir__="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

build_log=${__ubdl_buildall_py2_workdir__}/build.log

echo "<<< BUILD LARLITE >>>"
cd larlite
make >& $build_log
echo "<<< BUILD LARLITE/UserDev/BasicTool >>>"
cd UserDev/BasicTool
make -j4 >> $build_log 2>&1
echo "<<< BUILD LARLITE/UserDev/SelectionTool/LEEPreCuts >>>"
cd ../SelectionTool/LEEPreCuts
git submodule init
git submodule update
make -j4 >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "<<< BUILD GEO2D >>>"
cd Geo2D
make -j4 >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "<<< BUILD LAROPENCV >>>"
cd LArOpenCV
make -j4 >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "<<< BUILD LARCV >>>"
cd larcv
mkdir -p build
cd build
cmake -DUSE_PYTHON2=ON -DUSE_OPENCV=ON -DON_FNAL=ON -DUSE_TORCH=ON ../
make install >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "<<< BUILD CILANTRO >>>"
cd cilantro
mkdir -p build
cd build
cmake ../
make >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "<<< BUILD UBLARCVAPP >>>"
mkdir -p ublarcvapp/build
cd ublarcvapp
source configure.sh
cd build
cmake -DUSE_OPENCV=ON ../
make install >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "<<< BUILD LARFLOW >>>"
mkdir -p larflow/build
cd larflow
source configure.sh
cd build
cmake ../
make install >> $build_log 2>&1
cd $__ubdl_buildall_py2_workdir__

echo "built ubdl modules"
cd $__ubdl_buildall_py2_startdir__
