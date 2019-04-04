#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_032919.img

module load singularity
singularity shell $container
