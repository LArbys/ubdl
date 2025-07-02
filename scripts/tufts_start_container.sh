#!/bin/bash

module load apptainer/1.2.4-suid

container=/cluster/tufts/wongjiradlabnu/larbys/larbys-container/u20.04_cu111_cudnn8_torch1.9.0_minkowski_npm.sif

ls /cluster/tufts/wongjiradlab > /dev/null
ls /cluster/tufts/wongjiradlabnu > /dev/null
apptainer shell --nv -B /cluster:/cluster $container
