#!/bin/bash

# setup the environment variables for all the components
workdir=$PWD

# set the basedir
export UBDL_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

export OPENCV_LIBDIR=/usr/local/lib
export OPENCV_INCDIR=/usr/local/include

cd $UBDL_BASEDIR

# larlite
cd larlite
source config/setup.sh
cd $UBDL_BASEDIR

# Geo2D
cd Geo2D
source config/setup.sh
cd $UBDL_BASEDIR

# LArOpenCV
cd LArOpenCV
source setup_laropencv.sh
cd $UBDL_BASEDIR

# LArCV
cd larcv
source configure.sh
cd $UBDL_BASEDIR

# Cilantro (3rd party)
cd cilantro
cd $UBDL_BASEDIR

# UB LArCV app
cd ublarcvapp
source configure.sh


cd $workdir
