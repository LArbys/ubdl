import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
# sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larflow/larcvdataset")
# from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata
import dataLoader as dl

def load_classifier_larcvdata( name, inputfile, batchsize, nworkers,
                            input_producer_name,true_producer_name, plane,
                            tickbackward=False, readonly_products=None):
    feeder = SparseClassifierPyTorchDataset(inputfile, batchsize,
                                         input_producer_name=input_producer_name,
                                         true_producer_name=true_producer_name,
                                         plane=plane,
                                         tickbackward=tickbackward, nworkers=nworkers,
                                         readonly_products=readonly_products,
                                         feedername=name)
    return feeder


class SparseClassifierPyTorchDataset(torchdata.Dataset):
        idCounter = 0
        def __init__(self,inputfile,batchsize,input_producer_name, true_producer_name,
                     plane, tickbackward=False,nworkers=4,
                     readonly_products=None,
                     feedername=None):
            super(SparseClassifierPyTorchDataset,self).__init__()

            if type(inputfile) is str:
                self.inputfiles = [inputfile]
            elif type(inputfile) is list:
                self.inputfiles = inputfile

            if type(input_producer_name) is not str:
                raise ValueError("producer_name type must be str")

            # get length by querying the tree
            self.nentries  = 0
            tchain = rt.TChain("sparseimg_nbkrnd_sparse_tree".format(input_producer_name))
            print("TCHAIN:",tchain)
            for finput in self.inputfiles:
                tchain.Add(finput)
            self.nentries = tchain.GetEntries()
            #print "nentries: ",self.nentries
            del tchain

            if feedername is None:
                self.feedername = "SparseClassifierImagePyTorchDataset_%d"%\
                                    (SparseImagePyTorchDataset.idCounter)
            else:
                self.feedername = feedername
            self.batchsize = batchsize
            self.nworkers  = nworkers
            readonly_products = None
            # params = {"inputproducer":input_producer_name, "trueproducer":true_producer_name,"plane":plane}

            # note, with way LArCVServer workers, must always use batch size of 1
            #   because larcvserver expects entries in each batch to be same size,
            #   but in sparse representations this is not true
            # we must put batches together ourselves for sparseconv operations
            # TODO: this gets the data, make it from dl probs
            # self.feeder = LArCVServer(1,self.feedername,
            #                           load_cropped_sparse_infill,
            #                           self.inputfiles,self.nworkers,
            #                           server_verbosity=-1,worker_verbosity=-1,
            #                           io_tickbackward=tickbackward,
            #                           func_params=params)
            print("SETTING FEEDER")
            self.feeder = dl.load_rootfile_training(inputfile)
            
            SparseClassifierPyTorchDataset.idCounter += 1

        def __len__(self):
            #print "return length of sample:",self.nentries
            return self.nentries

        def __getitem__(self,index):
            data = [self.feeder[0][index],self.feeder[4][index]]
            return data