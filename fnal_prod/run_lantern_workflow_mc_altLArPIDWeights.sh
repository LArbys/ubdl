#!/bin/bash

echo "file contents at job start:"
ls

echo "setting up LANTERN environment"

WORKDIR="$(pwd)"

shopt -s expand_aliases
alias python=python3

export UBDL_DIR=/cluster/home/ubdl/
export RECO_TEST_DIR=/cluster/home/ubdl/larflow/larflow/Reco/test/
export LARMATCH_DIR=/cluster/home/ubdl/larflow/larmatchnet/larmatch/
export SSNET_DIR=/cluster/home/uresnet_pytorch/
export NTMAKER_DIR=/cluster/home/gen2ntuple/
export LARPID_DIR=/cluster/home/prongCNN/models/checkpoints/
export LANTERN_SCRIPTS=/cluster/home/lantern_scripts/

source ${UBDL_DIR}/setenv_py3_container.sh
source ${UBDL_DIR}/configure.sh

cd ${UBDL_DIR}/larflow/larmatchnet
source set_pythonpath.sh
export PYTHONPATH=${LARMATCH_DIR}:${PYTHONPATH}
export PYTHONPATH=${PYTHONPATH}:${NTMAKER_DIR}
export PYTHONPATH=${PYTHONPATH}:${SSNET_DIR}

cd $WORKDIR

echo "Environment after container setup"
env

echo "running LANTERN workflow"

larsoftout=$(find . -maxdepth 1 -type f -name 'merged_dlreco*.root' -printf '%f\n' -quit)
baseinput="merged_dlreco_with_ssnet.root"
CONFIG_FILE="${LANTERN_SCRIPTS}/config_larmatchme_deploycpu.yaml"
WEIGHT_FILE="larmatch_ckpt78k.pt"
LARPID_MODEL="${LARPID_DIR}/LArPID_alternate_network_weights.pt"

echo "larsoftout: $larsoftout"
echo "baseinput: $baseinput"

# local outfile
INPUTSTEM="merged_dlreco_with_ssnet"
lm_outfile=$(echo $baseinput  | sed 's|'${INPUTSTEM}'|larmatchme|g')
baselm=$(echo $baseinput | sed 's|'${INPUTSTEM}'|larmatchme|g' | sed 's|.root|_larlite.root|g')
reco_outfile=$(echo $baseinput  | sed 's|'${INPUTSTEM}'|larflowreco|g')
reco_basename=$(echo $baseinput | sed 's|'${INPUTSTEM}'|larflowreco|g' | sed 's|.root||g')
ntuple_outfile=$(echo $baseinput  | sed 's|'${INPUTSTEM}'|flat_ntuple|g')
echo "larmatch outfile: $lm_outfile"
echo "baselm: $baselm"
echo "reco outfile: $reco_outfile"
echo "reco basename: $reco_basename"
echo "ntuple outfile: $ntuple_outfile"


echo "----------------------------------------------"

CMD="python3 $SSNET_DIR/inference_sparse_ssnet_uboone.py -i $larsoftout -w $SSNET_DIR/weights -o ssnet_output.root"
echo $CMD
$CMD

echo "merging ssnet output with merged_dlreco..."
cp ${larsoftout} ${baseinput}
rootcp ssnet_output.root:sparseimg_sparsessnet_tree ${baseinput}
python3 ${SSNET_DIR}/recreate_ubspurn.py -i ${baseinput} -o ssnet_ubspurn_output.root
inputTrees=$(rootls ssnet_ubspurn_output.root)
for tree in $inputTrees; do
  rootcp ssnet_ubspurn_output.root:${tree} ${baseinput}
done

CMD="python3 $LARMATCH_DIR/deploy_larmatchme.py --config-file ${CONFIG_FILE} --supera $baseinput --weights ${LARMATCH_DIR}/${WEIGHT_FILE} --output $lm_outfile --min-score 0.5 --adc-name wire --chstatus-name wire --device-name cpu"
echo $CMD
$CMD

CMD="python3 $RECO_TEST_DIR/run_kpsrecoman.py --input-dlmerged ${baseinput} --input-larflow ${baselm} --output ${reco_outfile} -mc --products min --save-all-keypoints --loglevel 3"
echo $CMD
$CMD

CMD="python3 $NTMAKER_DIR/make_dlgen2_flat_ntuples.py -f ${reco_basename}_kpsrecomanagerana.root -t $baseinput -o $ntuple_outfile -m $LARPID_MODEL -mc --ignoreWeights --noEvtSkipping"
echo $CMD
$CMD

echo "compress cosmic reco tracks and save in separate file"
CMD="python3 ${NTMAKER_DIR}/extract_and_compress_cosmicreco.py larflowreco_larlite.root"
echo $CMD
$CMD

echo "extract crt and opflash info from merged_dlreco"
rootcp merged_dlreco_with_ssnet.root:crthit_*_tree merged_dlreco_with_ssnet.root:crttrack_*_tree merged_dlreco_with_ssnet.root:opflash_*_tree merged_dlreco_with_ssnet.root:opdigit_*_tree extract_from_merged.root

echo "add compressed cosmic tracks to kpsrecomanagerana file"
hadd larflowreco_kpsrecomanagerana_wcosmic.root compressed_cosmic_components.root larflowreco_kpsrecomanagerana.root extract_from_merged.root
mv larflowreco_kpsrecomanagerana_wcosmic.root larflowreco_kpsrecomanagerana.root

echo "cleaning up intermediate files"
rm ssnet_output.root ssnet_ubspurn_output.root merged_dlreco_with_ssnet.root
rm larmatchme_larlite.root larmatchme_larcv.root larflowreco_larlite.root larflowreco_larcv.root
rm compressed_cosmic_components.root
rm temp.root
rm extract_from_merged.root

echo "done!"

