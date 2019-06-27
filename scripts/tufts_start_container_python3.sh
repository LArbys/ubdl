#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_python3_041619.img

module load singularity
singularity shell --nv $container
