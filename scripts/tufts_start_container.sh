#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_091319.simg

module load singularity
singularity shell --nv $container
