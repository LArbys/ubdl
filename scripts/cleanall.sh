#!/bin/bash

start_dir=$PWD

cd $UBDL_BASEDIR

cd larlite
make clean
cd UserDev/BasicTool
make clean
cd $UBDL_BASEDIR

cd Geo2D
make clean
cd $UBDL_BASEDIR

cd LArOpenCV
make clean
cd $UBDL_BASEDIR

cd larcv/build && make clean
cd $UBDL_BASEDIR

cd ublarcvapp/build && make clean
cd $UBDL_BASEDIR

echo "cleaned ubdl repositories"
cd $start_dir
