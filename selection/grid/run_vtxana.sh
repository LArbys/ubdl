#!/bin/bash

# note, we are inside the container
flist_vtx1=$1
flist_vtx2=$2
#flist_clust=$3
flist_larcvtruth=$3
tag=$4
arrayid=$SLURM_ARRAY_TASK_ID

jobdir=`printf jobdirs/vtx/job%d ${arrayid}`

# parse flist 
let line=${arrayid}+1
vtx1file=`sed -n ${line}p ${flist_vtx1}`
vtx2file=`sed -n ${line}p ${flist_vtx2}`
larcvtruthfile=`sed -n ${line}p ${flist_larcvtruth}`
#ubclustfile=`sed -n ${line}p ${flist_clust}`
ubclustfile=`sed -n ${line}p ${flist_larcvtruth} | sed 's%stage1%ubdlserver%g' | sed 's%larcvtruth%ubclust-noinfill-larlite-mcc9v12_intrinsicoverlay%g'`
ubmrcnnfile=`echo "$ubclustfile" | sed 's%ubclust%ubmrcnn%g' | sed 's%larlite%larcv%g'`
superafile=`echo "$larcvtruthfile" | sed 's%larcvtruth%supera%g'`
mcinfofile=`echo "$larcvtruthfile" | sed 's%larcvtruth%mcinfo%g'`
ssnetfile=`echo "$larcvtruthfile" | sed 's%larcvtruth%ssnetserveroutv2-larcv%g'`

echo "${superafile}"
echo "${mcinfofile}"
echo "${ubmrcnnfile}"
echo "${ubclustfile}"
echo "${larcvtruthfile}"
echo "${ssnetfile}"
echo "${vtx1file}"
echo "${vtx2file}"


source /usr/local/root/build/bin/thisroot.sh

# UBDL REPO TO USE

# inside container
#cd /usr/local/ubdl
cd /cluster/tufts/wongjiradlab/rshara01/ubdl/

source setenv.sh
source configure.sh

workdir=/cluster/tufts/wongjiradlab/rshara01/ubdl/selection/
echo ${workdir}
cd $workdir

mkdir -p $jobdir
cd $jobdir/

saveme="save"

cp $workdir/dev_cosmictag .
#echo "./dev_cosmictag ${ublarcvfile} ${ubclustfile} ${larcvtruthfile} ${ssnetfile} ${larliteout}"
./dev_cosmictag $superafile $mcinfofile $ubmrcnnfile $ubclustfile $larcvtruthfile $ssnetfile $vtx1file $vtx2file $saveme  >& log_job${arrayid}.txt || exit
 
# finally copy output
#mv $larcvout $outdir/
#mv $larliteout $outdir/

# clean up
cd $workdir
#rm -r $jobdir
