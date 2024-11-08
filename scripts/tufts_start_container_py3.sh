#!/bin/bash
#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_py3deps_u18.04_cu11_pytorch1.7.1.simg
#container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdldeps_u20.02_pytorch1.9_py3.simg
#container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0.sif
#container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_comput8.sif
#container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/singularity_minkowskiengine_u20.04.cu111.torch1.9.0_compute8_wjupyternotebook.sif
container=/cluster/tufts/wongjiradlabnu//larbys/larbys-container/singularity_minkowski_u20.04.cu111.torch1.9.0_jupyter_xgboost.sif
module load singularity/3.5.3
singularity shell --nv --bind /cluster/tufts/:/cluster/tufts/,/tmp:/tmp $container
