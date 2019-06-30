#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_051919.img

module load singularity
singularity shell --nv $container
