#!/bin/bash

alias python=python3
alias python-config=python3-config

__ubdl_buildall_py3_workdir__=$PWD
build_log=${__ubdl_buildall_py3_workdir__}/build.log

cd larlite
make >>& ${build_log}
cd UserDev/BasicTool
make -j4 >>& ${build_log}
cd $workdir

cd Geo2D
make -j4 >>& ${build_log}
cd $workdir

cd LArOpenCV
make -j4 >>& ${build_log}
cd $workdir

cd larcv/build
cmake -DUSE_PYTHON3=ON -DUSE_OPENCV=ON -DUSE_FNAL=OFF -DUSE_TORCH=OFF ../
make install >>& ${build_log}
cd $workdir

mkdir -p ublarcvapp/build
cd ublarcvapp
source configure.sh
cd build
cmake ../
make install >>& ${build_log}
cd $workdir

echo "built ubdl modules"
