#!/bin/bash

startdir=$PWD
here="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $here

if [[ ! -z ${UBDL_BASEDIR} ]]; then
  # need to setup ubdl
  source ../setenv.sh
  source ../configure.sh
fi

echo "back to start dir"
cd $startdir
