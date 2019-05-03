#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_041519.img

module load singularity
singularity shell $container
