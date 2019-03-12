#!/bin/bash

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img

module load singularity
singularity shell $container
