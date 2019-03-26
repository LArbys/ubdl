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

cd larcv/build
cmake -DUSE_PYTHON3=ON -DUSE_OPENCV=OFF ../
make install
cd $workdir

mkdir -p ublarcvapp/build
cd ublarcvapp
source configure.sh
cd build
cmake ../
make install
cd $workdir

echo "built ubdl modules"
