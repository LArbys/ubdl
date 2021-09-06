#!/bin/bash
#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_py3deps_u18.04_cu11_pytorch1.7.1.simg
container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdldeps_u20.02_pytorch1.9_py3.simg
module load singularity/3.5.3
singularity shell --nv --bind /cluster/tufts/:/cluster/tufts/ $container
