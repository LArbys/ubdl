#!/bin/bash

export UBDL_DIR=/cluster/home/ubdl/
export RECO_TEST_DIR=/cluster/home/ubdl/larflow/larflow/Reco/test/
export LARMATCH_DIR=/cluster/home/ubdl/larflow/larmatchnet/larmatch/
export SSNET_DIR=/cluster/home/uresnet_pytorch/
export NTMAKER_DIR=/cluster/home/gen2ntuple/
export LARPID_DIR=/cluster/home/prongCNN/models/checkpoints/
export LANTERN_SCRIPTS=/cluster/home/lantern_scripts/

input_rootfile=$1
lm_outfile=$2
FLAGS=$3

CONFIG_FILE="${LANTERN_SCRIPTS}/config_larmatchme_deploycpu.yaml"
WEIGHT_FILE="larmatch_ckpt78k.pt"

CMD="python3 $LARMATCH_DIR/deploy_larmatchme.py --config-file ${CONFIG_FILE} --supera $input_rootfile --weights ${LARMATCH_DIR}/${WEIGHT_FILE} --output $lm_outfile --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu --use-skip-limit ${FLAGS}"
echo $CMD
$CMD