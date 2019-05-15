#!/bin/bash

workdir=$PWD
cd larlite
make
cd UserDev/BasicTool
make -j4
cd $workdir

cd Geo2D
make -j4
cd $workdir

cd LArOpenCV
make -j4
cd $workdir

cd larcv
mkdir -p build
cd build
cmake -DUSE_PYTHON2=ON -DUSE_OPENCV=ON ../
make install
cd $workdir

cd cilantro
mkdir -p build
cd build
cmake ../
make
cd $workdir

mkdir -p ublarcvapp/build
cd ublarcvapp
source configure.sh
cd build
cmake -DUSE_OPENCV=ON ../
make install
cd $workdir

echo "built ubdl modules"

