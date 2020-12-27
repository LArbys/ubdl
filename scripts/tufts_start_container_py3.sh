#!/bin/bash
container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_py3deps_u18.04_cu11_pytorch1.7.1.simg
module load singularity
singularity shell --nv $container
