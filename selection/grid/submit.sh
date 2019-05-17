#!/bin/bash
#
#SBATCH --job-name=cosmictag
#SBATCH --output=log_cosmictag
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
#SBATCH --array=0000-263

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/ubdl_singularity_031219.img
WORKDIR=/cluster/tufts/wongjiradlab/rshara01/ubdl/selection/

# MCC9 Jan EXTBNB
TAG=mcc9v12_intrinsicoverlay
SUPERALIST_IC=${WORKDIR}/grid/flist_larcvtruth_${TAG}.list
RECO2DLIST_IC=${WORKDIR}/grid/flist_ublarcv_${TAG}.list
OUTDIR=${WORKDIR}/output/${TAG}/

mkdir -p ${OUTDIR}

module load singularity

# grid running
singularity exec ${CONTAINER} bash -c "cd ${WORKDIR} && ls && source ${WORKDIR}/grid/run_cosmictag.sh ${SUPERALIST_IC} ${RECO2DLIST_IC} ${OUTDIR} ${TAG}"

# local running for tests
#singularity exec ${CONTAINER} bash -c "cd ${WORKDIR} && ls && SLURM_ARRAY_TASK_ID=1 source ${WORKDIR}/grid/run_cosmictag.sh ${SUPERALIST_IC} ${RECO2DLIST_IC} ${OUTDIR} ${TAG}"
#singularity exec ${CONTAINER} bash -c "cd ${WORKDIR} && ls && SLURM_ARRAY_TASK_ID=1 source ${WORKDIR}/grid/test.sh ${SUPERALIST_IC} ${RECO2DLIST_IC} ${OUTDIR} ${TAG}"
