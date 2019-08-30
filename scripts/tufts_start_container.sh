#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_deps_py2_082619.simg

module load singularity
singularity shell --nv $container
