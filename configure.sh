#!/bin/bash

# setup the environment variables for all the components

# save the folder where script is called
__ubdl_configure_workdir__=$PWD

# set the basedir
export UBDL_BASEDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

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
source dllee_setup.sh
cd $UBDL_BASEDIR

# UB LArCV app
cd ublarcvapp
source configure.sh
cd $UBDL_BASEDIR

# LArFlow
cd larflow
source configure.sh
cd $UBDL_BASEDIR

# LArdly viewing tools
cd lardly
source setenv.sh
cd $UBDL_BASEDIR

cd ${__ubdl_configure_workdir__}


export ROOT_INCLUDE_PATH=${LARLITE_INCDIR}
export ROOT_INCLUDE_PATH=${LARLITE_INCDIR}/larlite/Base:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARLITE_INCDIR}/larlite/BasicTool:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARLITE_INCDIR}/larlite/Analysis:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARLITE_INCDIR}/larlite/DataFormat:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARLITE_INCDIR}/larlite/LArUtil:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/Base:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/CPPUtil:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/CVUtil:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/DataFormat:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/Processor:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/PyUtil:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/ROOTUtil:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/core/json:${ROOT_INCLUDE_PATH}
export ROOT_INCLUDE_PATH=${LARCV_INCDIR}/larcv/app/ImageMod:${ROOT_INCLUDE_PATH}
