#!/bin/bash
container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdldeps_u20.02_py3.simg
module load singularity
singularity shell --nv $container
