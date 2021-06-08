#!/bin/env python

## IMPORT

# python,numpy
import platform
print(platform.python_version())
import os,sys
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F


from networkmodel import SparseClassifier


# ===================================================
# TOP-LEVEL PARAMETERS
RUNPROFILER=False
IMAGE_WIDTH=3456
IMAGE_HEIGHT=1002
# ===================================================


def main():

    DEVICE = torch.device("cpu")

    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 16
    noutput_features = 16
    nplanes = 5
    reps = 1    
        
    # self, inputshape, reps, nin_features, nout_features, nplanes,show_sizes
    model = SparseClassifier( (IMAGE_HEIGHT,IMAGE_WIDTH), reps,
                           ninput_features, noutput_features,
                           nplanes, show_sizes=True).to(DEVICE)

    # uncomment to dump model
    if True:
        print ("Loaded model: ",model)
        return




    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        print("PROFILER")
        if RUNPROFILER:
            print (prof)


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
