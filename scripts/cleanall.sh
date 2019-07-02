#!/bin/bash

__ubdl_cleanall_start_dir__=$PWD

cd $UBDL_BASEDIR

echo "<<< CLEAN LARLITE >>>"
cd larlite
make clean
cd UserDev/BasicTool
make clean
cd $UBDL_BASEDIR

echo "<<< CLEAN GEO2D >>>"
cd Geo2D
make clean
cd $UBDL_BASEDIR

echo "<<< CLEAN LAROPENCV >>>"
cd LArOpenCV
make clean
cd $UBDL_BASEDIR

echo "<<< CLEAN LARCV >>>"
cd larcv/build && make clean
cd $UBDL_BASEDIR

echo "<<< CLEAN UBLARCVAPP >>>"
cd ublarcvapp/build && make clean
cd $UBDL_BASEDIR

echo "<<< CLEAN LARFLOW >>>"
cd larflow/build && make clean
cd $UBDL_BASEDIR

echo "cleaned ubdl repositories"
cd $__ubdl_cleanall_start_dir__
