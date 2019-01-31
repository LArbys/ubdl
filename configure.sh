#!/bin/bash

# setup the environment variables for all the components
workdir=$PWD

export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include

# larlite
cd larlite
source config/setup.sh
cd $workdir

# Geo2D
cd Geo2D
source config/setup.sh
cd $workdir

# LArOpenCV
cd LArOpenCV
source setup_laropencv.sh
cd $workdir

# LArCV
cd larcv
source configure.sh
cd $workdir

# UB LArCV app
#cd ublarcvapp
#source configure.sh
