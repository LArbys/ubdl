#!/bin/bash

# slurm submission script to build ubdl

#SBATCH --job-name=py3_ubdl_build
#SBATCH --output=py3_ubdl_build.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=3
#SBATCH --partition wongjiradlab

container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_depsonly_py3.6.11_u16.04_cu11_pytorch1.7.1.simg

# get dir where we called script
workdir=$PWD

# in case we called it in the folder above
workdir=`echo ${workdir} | sed 's|scripts||'`

# get directory where script lives:
echo "workdir: ${workdir}"

module load singularity
#srun singularity exec ${container} bash -c "cd ${workdir} && source setenv_py3.sh && source configure.sh && source scripts/cleanall.sh && source buildall_py3.sh"
srun singularity exec ${container} bash -c "cd ${workdir} && source setenv_py3.sh && source configure.sh && source buildall_py3.sh"

