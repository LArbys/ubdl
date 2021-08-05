import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
# sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larflow/larcvdataset")
# from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata
import dataLoader as dl # TODO: Change this back

def load_classifier_larcvdata( name, inputfile, batchsize, nworkers, nbatches, verbosity,
                            input_producer_name,true_producer_name,
                            tickbackward=False, readonly_products=None,
                            start_entry=0, end_entry=-1, rand=True):
    feeder = SparseClassifierPyTorchDataset(inputfile, batchsize, nbatches, verbosity,
                                         input_producer_name=input_producer_name,
                                         true_producer_name=true_producer_name,
                                         tickbackward=tickbackward, nworkers=nworkers,
                                         readonly_products=readonly_products,
                                         feedername=name,
                                         start_entry=start_entry, end_entry=end_entry, rand=rand)
    return feeder


class SparseClassifierPyTorchDataset(torchdata.Dataset):
        idCounter = 0
        def __init__(self,inputfile,batchsize,nbatches,verbosity,input_producer_name, true_producer_name,
                     tickbackward=False,nworkers=4,
                     readonly_products=None,
                     feedername=None,
                     start_entry = 0, end_entry = -1, rand=True):
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
            self.nbatches = nbatches
            self.nworkers  = nworkers
            self.verbosity = verbosity
            self.start_entry = start_entry
            self.end_entry = end_entry
            self.rand = rand
            readonly_products = None
            print("SETTING FEEDER")
            self.feeder = dl.load_rootfile_training(inputfile, self.batchsize, self.nbatches, self.verbosity, start_entry=self.start_entry, end_entry=self.end_entry, rand=self.rand)
            
            SparseClassifierPyTorchDataset.idCounter += 1

        def __len__(self):
            return self.nentries

        def __getitem__(self,index):
            data = [self.feeder[0][index],self.feeder[4][index]]
            return data
        
        def set_nentries(self,nentries):
            self.nentries = nentries
        
        def get_len_eff(self):
            len_eff = self.feeder[8]
            return len_eff
