#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_py3_deponly_070119.img

module load singularity
singularity shell --nv $container
