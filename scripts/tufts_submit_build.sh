#!/bin/bash

# slurm submission script to build ubdl

#SBATCH --job-name=ubdl_build
#SBATCH --output=ubdl_build.log
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8000
#SBATCH --time=30:00
#SBATCH --cpus-per-task=3


container=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img

# get dir where we called script
workdir=$PWD

# in case we called it in the folder above
workdir=`echo ${workdir} | sed 's|scripts||'`

# get directory where script lives:
echo "workdir: ${workdir}"

# change name for dir when inside container
workdir_ic=`echo ${workdir} | sed 's|kappa/90-days-archive|kappa|'`
workdir_ic=`echo ${workdir_ic} | sed 's|tufts|kappa|'`
echo "workdir_ic: ${workdir_ic}"

module load singularity
srun singularity exec ${container} bash -c "cd ${workdir_ic} && source setenv.sh && source configure.sh && source buildall.sh >& build.log"
