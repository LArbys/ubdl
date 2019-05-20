#!/bin/bash

# note, we are inside the container
flist_vtx1=$1
flist_vtx2=$2
flist_clust=$3
flist_larcvtruth=$4
tag=$5
arrayid=$SLURM_ARRAY_TASK_ID

jobdir=`printf jobdirs/vtx/job%d ${arrayid}`

# parse flist 
let line=${arrayid}+1
vtx1file=`sed -n ${line}p ${flist_vtx1}`
vtx2file=`sed -n ${line}p ${flist_vtx2}`
ubclustfile=`sed -n ${line}p ${flist_clust}`
#larcvtruthfile=`sed -n ${line}p ${flist_ublarcv} | sed 's%ubdlserver%stage1%g' | sed 's%ubpost-noinfill-larcv-mcc9v12_intrinsicoverlay%larcvtruth%g'`
larcvtruthfile=`sed -n ${line}p ${flist_larcvtruth}`
superafile=`echo "$larcvtruthfile" | sed 's%larcvtruth%supera%g'`
mcinfofile=`echo "$larcvtruthfile" | sed 's%larcvtruth%mcinfo%g'`

echo "${vtx1file}"
echo "${vtx2file}"
echo "${larcvtruthfile}"
echo "${superafile}"
echo "${ubclustfile}"
echo "${mcinfofile}"

source /usr/local/root/build/bin/thisroot.sh

# UBDL REPO TO USE

# inside container
cd /usr/local/ubdl

source setenv.sh
source configure.sh

workdir=/cluster/tufts/wongjiradlab/rshara01/ubdl/selection/
echo ${workdir}
cd $workdir

mkdir -p $jobdir
cd $jobdir/


cp $workdir/vertex_eval .
#echo "./dev_cosmictag ${ublarcvfile} ${ubclustfile} ${larcvtruthfile} ${ssnetfile} ${larliteout}"
./vertex_eval $vtx1file $vtx2file $ubclustfile $superafile $larcvtruthfile $mcinfofile >& log_job${arrayid}.txt || exit
 
# finally copy output
#mv $larcvout $outdir/
#mv $larliteout $outdir/

# clean up
cd $workdir
#rm -r $jobdir
