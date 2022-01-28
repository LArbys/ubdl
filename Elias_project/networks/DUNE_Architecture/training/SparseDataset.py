import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
# sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larflow/larcvdataset")
# from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata
import DataLoader

class SparseClassifierDataset(torchdata.Dataset):
    def __init__(self, dataloader, start_entry=0, end_entry=-1):
        self.start_entry = start_entry
        self.end_entry = end_entry
        self.nentries = dataloader.nentries_cv
        print("start entry:",self.start_entry)
        self.feeder = dataloader.load_rootfile(self.start_entry, self.end_entry)

    def __len__(self):
        return self.nentries

    def __getitem__(self,index):
        data = [self.feeder[0][index],self.feeder[4][index],self.feeder[6][index],[self.feeder[1][index],self.feeder[2][index],self.feeder[3][index]]]
        return data
    
    def set_nentries(self,nentries):
        self.nentries = nentries
    
    def get_len_eff(self):
        len_eff = self.feeder[8]
        return len_eff
