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

import dataLoader as dl

# ===================================================
# TOP-LEVEL PARAMETERS
RUNPROFILER=False
IMAGE_WIDTH=3456
IMAGE_HEIGHT=1024
# ===================================================
# image height is really 1008

def main():

    DEVICE = torch.device("cpu")

    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 64
    noutput_features = 64
    nplanes = 5
   
        
    # self, inputshape, nin_features, nout_features, nplanes,show_sizes
    model = SparseClassifier( (IMAGE_HEIGHT,IMAGE_WIDTH),
                           ninput_features, noutput_features,
                           nplanes, show_sizes=True).to(DEVICE)

    # uncomment to dump model
    # if True:
    #     print ("Loaded model: ",model)
    #     return

    #put the network in train mode
    model.train()
    
    # TODO: make sure this all works for multiple inputs
    
    all_data = dl.load_rootfile_training("/home/ebengh01/ubdl/Elias_project/networks/data/output_10001.root", 0, 1)
    img_list = dl.split_into_planes(all_data[0], 0, 1)
    coords_np = img_list[0]
    inputs_np = img_list[1]
    
    ncoords = [coords_np[0].shape[0], coords_np[1].shape[0], coords_np[2].shape[0]]
    
    
    
    # ncoords = [3789000, 2890472, 8931000]
    batchsize = 1
    # torch.set_default_dtype(torch.float32)
    coord_t = [torch.Tensor(), torch.Tensor(), torch.Tensor()]
    input_t = [torch.Tensor(), torch.Tensor(), torch.Tensor()]
    print("coord_t type:",coord_t[0].dtype)
    print("input_t type:",input_t[0].dtype)
    for i in range(3):
        print("ncoords[i]:",ncoords[i])
        coord_t[i] = torch.from_numpy(coords_np[i])
        input_t[i] = torch.from_numpy(inputs_np[i])
        
        # input_t[i] = torch.zeros( (ncoords[i],1), dtype=torch.float)
        # input_t[i].set_default_dtype(torch.float64)
        print("coord_t type:",coord_t[i].dtype)
        print("input_t type:",input_t[i].dtype)
    
    # send the prediction through the model
    planes = dl.get_truth_planes(all_data[4], 0, 1)
    if planes == 0:
        predict_t = model(coord_t, input_t,batchsize)
    else:
        print("Empty planes")
        predict_t = -1
    

    

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:

        print("PROFILER")
        if RUNPROFILER:
            print (prof)


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()
