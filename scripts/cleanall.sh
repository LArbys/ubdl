#!/bin/bash

__ubdl_cleanall_start_dir__=$PWD

echo "<<< CLEAN LARLITE >>>"
cd $UBDL_BASEDIR
cd larlite/build/ && rm -rf *

echo "<<< CLEAN GEO2D >>>"
cd $UBDL_BASEDIR
cd Geo2D
make clean

echo "<<< CLEAN LAROPENCV >>>"
cd $UBDL_BASEDIR
cd LArOpenCV
make clean
cd $UBDL_BASEDIR

echo "<<< CLEAN CILANTRO >>>"
cd $UBDL_BASEDIR
cd cilantro/build
make clean
rm -r *

echo "<<< CLEAN LARCV >>>"
cd $UBDL_BASEDIR
cd larcv/build && rm -rf *

echo "<<< CLEAN UBLARCVAPP >>>"
cd $UBDL_BASEDIR
cd ublarcvapp/build && rm -rf *


echo "<<< CLEAN LARFLOW >>>"
cd $UBDL_BASEDIR
cd larflow/build && rm -rf *


echo "cleaned ubdl repositories"
cd $__ubdl_cleanall_start_dir__
