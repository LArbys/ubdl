#!/bin/bash

INSTALL_PATH=/exp/uboone/app/users/tmw/ups_dev/products/ubdl/latest/Linux64bit+3.10-2.17_c14_prof/

rsync_cmd="rsync -av --progress --exclude-from=./rsync_exclude_patterns.txt ${UBDL_BASEDIR}/ups ${INSTALL_PATH}/"
echo $rsync_cmd
$rsync_cmd

rsync_cmd="rsync -av --progress --exclude-from=./rsync_exclude_patterns.txt ${UBDL_BASEDIR}/* ${INSTALL_PATH}/"
echo $rsync_cmd
$rsync_cmd


