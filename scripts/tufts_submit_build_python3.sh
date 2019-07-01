#!/bin/bash

# slurm submission script to build ubdl

#SBATCH --job-name=py3_ubdl_build
#SBATCH --output=py3_ubdl_build.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=30:00
#SBATCH --cpus-per-task=3


container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_py3_deponly_070119.img

# get dir where we called script
workdir=$PWD

# in case we called it in the folder above
workdir=`echo ${workdir} | sed 's|scripts||'`

# get directory where script lives:
echo "workdir: ${workdir}"

module load singularity
srun singularity exec ${container} bash -c "cd ${workdir} && source setenv.sh && source configure.sh && source buildall_py3.sh"

