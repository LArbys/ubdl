#!/bin/bash
#This container seems to be gone -josh
#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_091319.simg
container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_10022019.simg
module load singularity
singularity shell --nv $container
