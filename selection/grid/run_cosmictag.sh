#!/bin/bash

# note, we are inside the container

flist_larcvtruth=$1
flist_ublarcv=$2
outdir=$3
tag=$4
arrayid=$SLURM_ARRAY_TASK_ID

jobdir=`printf jobdirs/job%d ${arrayid}`

# parse flist 
let line=${arrayid}+1
#larcvtruthfile=`sed -n ${line}p ${flist_larcvtruth}`
larcvtruthfile=`sed -n ${line}p ${flist_ublarcv} | sed 's%ubdlserver%stage1%g' | sed 's%ubpost-noinfill-larcv-mcc9v12_intrinsicoverlay%larcvtruth%g'`
ssnetfile=`echo "$larcvtruthfile" | sed 's%larcvtruth%ssnetserveroutv2-larcv%g'`
ublarcvfile=`sed -n ${line}p ${flist_ublarcv}`
ubclustfile=`echo "${ublarcvfile}" | sed 's%ubpost%ubclust%g' | sed 's%larcv%larlite%g'`

echo "${larcvtruthfile}"
echo "${ssnetfile}"
echo "${ubclustfile}"
echo "${ublarcvfile}"
larcvout="cosmictag-noinfill-larcv"
larcvout+=`echo "${larcvtruth:(-28)}"`
larliteout="cosmictag-noinfill-larlite"
larliteout+=`echo "${ubclustfile:(-28)}"`

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

doeff="eff"

cp $workdir/dev_cosmictag .
#echo "./dev_cosmictag ${ublarcvfile} ${ubclustfile} ${larcvtruthfile} ${ssnetfile} ${larliteout}"
./dev_cosmictag $ublarcvfile $ubclustfile $larcvtruthfile $ssnetfile $larliteout $doeff >& log_job${arrayid}.txt || exit
 
# finally copy output
#mv $larcvout $outdir/
mv $larliteout $outdir/

# clean up
cd $workdir
#rm -r $jobdir
