#!/bin/bash

startdir=$PWD
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

if [[ -z ${UBDL_BASEDIR} ]]; then
  # need to setup ubdl
  source ../configure.sh
fi

if [ ! -f ../ublarcvserver/configure.sh ]; then
  # need to clone the ublarcvserver submodule
  git submodule init
  git submodule update
fi

if [[ -z ${UBLARCVSERVER_BASEDIR} ]]; then
  # need to setup ublarcvserver
  source ../ublarcvserver/configure.sh
fi

if [ ! -f ../ublarcvserver/networks/pytorch-uresnet/.git ]; then
  cd ${here}/../ublarcvserver/
  git submodule init
  git submdoule update
  cd ${startdir}
fi

# PYTHON PATHS

# ubsset app dir
UBSSNETAPP_DIR=${UBLARCVSERVER_BASEDIR}/app/ubssnet
[[ ":$PYTHONPATH:" != *":${UBSSNETAPP_DIR}:"* ]] && export PYTHONPATH="${UBSSNETAPP_DIR}:${PYTHONPATH}"

# pytorch-uresnet model dir
PYTORCH_URESNET_MODEL_DIR=${UBLARCVSERVER_BASEDIR}/networks/pytorch-uresnet/models
[[ ":$PYTHONPATH:" != *":${PYTORCH_URESNET_MODEL_DIR=}:"* ]] && export PYTHONPATH="${PYTORCH_URESNET_MODEL_DIR=}:${PYTHONPATH}"

if [ ! -f ../ublarcvserver/networks/pytorch-uresnet/weights/mcc8_caffe_ubssnet_plane0.tar ]; then
  echo "--------------------------------------------------------"
  echo "missing the weight files for ubssnet."
  echo "go to ublarcvserver/networks/pytorch-uresnet/weights"
  echo "and run get_weights.sh if you are running the workers"
  echo "--------------------------------------------------------"
fi

echo ""
echo "Ready to go."
echo " Run ./start_ublarcvserver_broker.py to start the server"
echo " Run ./start_ubssnet_worker.py to start a worker"
echo " Run ./start_ubssnet_client.py to start a client"
