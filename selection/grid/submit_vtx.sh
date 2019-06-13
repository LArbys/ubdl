#!/bin/bash
#
#SBATCH --job-name=cosmictag
#SBATCH --output=log_cosmictag
#SBATCH --mem-per-cpu=2000
#SBATCH --time=1:00:00
#SBATCH --array=0001-200

CONTAINER=/cluster/tufts/wongjiradlab/larbys/larbys-containers/singularity_ubdl_041519.img
WORKDIR=/cluster/tufts/wongjiradlab/rshara01/ubdl/selection/

# MCC9 Jan EXTBNB
TAG=mcc9v12_intrinsicoverlay
VTX_WOTAGGER=${WORKDIR}/grid/flist_vtx_woTagger_${TAG}.list
VTX_WITAGGER=${WORKDIR}/grid/flist_vtx_wiTagger_${TAG}.list
#CLUSTLIST=${WORKDIR}/grid/flist_clust_${TAG}.list
SUPERALIST=${WORKDIR}/grid/flist_larcvtruth_${TAG}.list
#OUTDIR=${WORKDIR}/output/${TAG}/

#mkdir -p ${OUTDIR}

module load singularity

# grid running
singularity exec ${CONTAINER} bash -c "cd ${WORKDIR} && ls && source ${WORKDIR}/grid/run_vtxana.sh ${VTX_WOTAGGER} ${VTX_WITAGGER} ${CLUSTLIST} ${SUPERALIST} ${TAG}"

# local running for tests
#singularity exec ${CONTAINER} bash -c "cd ${WORKDIR} && ls && SLURM_ARRAY_TASK_ID=1 source ${WORKDIR}/grid/run_vtxana.sh ${VTX_WOTAGGER} ${VTX_WITAGGER} ${SUPERALIST} ${TAG}"
#singularity exec ${CONTAINER} bash -c "cd ${WORKDIR} && ls && SLURM_ARRAY_TASK_ID=1 source ${WORKDIR}/grid/test.sh ${SUPERALIST_IC} ${RECO2DLIST_IC} ${OUTDIR} ${TAG}"
