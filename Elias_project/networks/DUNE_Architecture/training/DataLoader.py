import random
import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp

import torch

class RootFileDataLoader(object):
    def __init__(self, infile, device, batchsize, nbatches, verbosity, rand=True):
        self.infile = infile
        self.device = device
        self.batchsize = batchsize
        self.nbatches = nbatches
        self.verbosity = verbosity
        self.rand = rand
        
        print("Initializing IOManager for",self.infile)
        self.iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickForward)
        self.iocv.reverse_all_products() # Do I need this?
        self.iocv.add_in_file(self.infile)
        self.iocv.initialize()
        
        self.nentries_cv = self.iocv.get_n_entries()
        
    def __del__(self):
        print("Deleting IOManager for",self.infile)
        self.iocv.finalize()
        
    def load_rootfile(self, start_entry = 0, end_entry = -1):
        
        # set up empty lists for saving data
        full_image_list = []
        runs = []
        subruns   = []
        events    = []
        truth = []
        filepaths = []
        entries   = []

        # entry number checks
        if end_entry > self.nentries_cv or end_entry == -1:
            end_entry = self.nentries_cv
        if start_entry > end_entry or start_entry < 0:
            start_entry = 0
        max_len = 0

        # determine how to pull entries
        if self.rand:
            entry_list = random.sample(range(start_entry,end_entry),self.nbatches*self.batchsize)
        else:
            if (start_entry + self.nbatches*self.batchsize) < end_entry:
                entry_list = [*range(start_entry,start_entry + self.nbatches*self.batchsize)]
            else:
                entry_list = [*range(start_entry,end_entry)]
        # begin iterating over entries
        for i in entry_list:
            print("Loading Entry:", i, "of range", start_entry, end_entry)
            self.iocv.read_entry(i)

            # ev_wire
            ev_sparse    = self.iocv.get_data(larcv.kProductSparseImage,"nbkrnd_sparse")
            ev_sparse_v = ev_sparse.SparseImageArray()
            
            # Get Sparse Image to a Numpy Array
            ev_sparse_np = larcv.as_sparseimg_ndarray(ev_sparse_v[0])
            # print("ev_sparse_np:",ev_sparse_np)
            
            sparse_ev_shape = ev_sparse_np.shape
            print("sparse_ev_shape:",sparse_ev_shape)
            if sparse_ev_shape[0] > max_len:
                max_len = sparse_ev_shape[0]

            # Extracting subrun truth informtion:
            subrun = ev_sparse.subrun()

            nc_cc = (subrun%10000)//1000
            flavors = (subrun%1000)//100
            if nc_cc == 1:
                if flavors == 0 or flavors == 1:
                    c_flavors = 0
                elif flavors == 2 or flavors == 3:
                    c_flavors = 1
            else: # nc_cc = 0
                c_flavors = 2
                # if flavors == 0 or flavors == 1:
                #     c_flavors = 2
                # elif flavors == 2 or flavors == 3:
                #     c_flavors = 3

            interactionType = (subrun%100)//10

            planes = subrun%10

            subrun = subrun//10000

            # Extracting event truth information
            event = ev_sparse.event()

            num_protons = (event%10000)//1000
            if num_protons > 2:
                num_protons = 3

            num_neutrons = (event%1000)//100
            if num_neutrons > 2:
                num_neutrons = 3

            num_pion_charged = (event%100)//10
            if num_pion_charged > 2:
                num_pion_charged = 3

            num_pion_neutral = event%10
            if num_pion_neutral > 2:
                num_pion_neutral = 3

            event = event//10000

            truth_list = [c_flavors, interactionType, num_protons, num_pion_charged, num_pion_neutral, num_neutrons, planes]
            
            if self.verbosity:
                print("shape test")
                print("raw subrun:",subrun)
                print("nc_cc:",nc_cc)
                print("flavors:",flavors)
                print("c_flavors:",c_flavors)
                print("interactionType:",interactionType)
                print("planes:",planes)
                print("subrun:",subrun)
                print("raw event:", event)
                print("num_protons:",num_protons)
                print("num_neutrons:", num_neutrons)
                print("num_pion_charged:", num_pion_charged)
                print("num_pion_neutral:",num_pion_neutral)
                print("event:",event)

            
            coords_u = np.empty((0,2))
            coords_v = np.empty((0,2))
            coords_y = np.empty((0,2))
            inputs_u = np.empty((0,1))
            inputs_v = np.empty((0,1))
            inputs_y = np.empty((0,1))
            coords = [coords_u.astype('long'), coords_v.astype('long'), coords_y.astype('long')]
            inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]
            pts = sparse_ev_shape[0]
            for j in range (0,pts):
                for k in range(2,5):
                    if ev_sparse_np[j,k] != 0:
                        coord_pair = [[ev_sparse_np[j,1],ev_sparse_np[j,0]]]
                        coords[k-2] = np.append(coords[k-2], coord_pair, axis=0).astype("long")
                        inputs[k-2] = np.append(inputs[k-2], [[ev_sparse_np[j,k]]], axis=0)
            # print(len(coords[0]))
            # print(len(coords[1]))
            # print(len(coords[2]))
            
            # print("coords[0].shape:",coords[0].shape)
            # print("coords[1].shape:",coords[1].shape)
            # print("coords[2].shape:",coords[2].shape)
            
            # print("HERE COMES THE MONEEEYYY")
            # inputs[0] = np.random.uniform(low=10, high=100, size=inputs[0].shape).astype('float32')
            # inputs[1] = np.random.uniform(low=10, high=100, size=inputs[1].shape).astype('float32')
            # inputs[2] = np.random.uniform(low=10, high=100, size=inputs[2].shape).astype('float32')
            
            # print("inputs[0]:",inputs[0])
            # print("inputs[1]:",inputs[1])
            # print("inputs[2]:",inputs[2])
            # coords[0] = np.random.random_integers(low=0, high=1008, size=coords[0].shape).astype('long')
            # coords[1] = np.random.random_integers(low=0, high=1008, size=coords[1].shape).astype('long')
            # coords[2] = np.random.random_integers(low=0, high=1008, size=coords[2].shape).astype('long')
            # 
            shape = [coords[0].shape[0],coords[1].shape[0],coords[2].shape[0]]
            print("shape:",shape)
            # coord = [coords_u.astype('long'), coords_v.astype('long'), coords_y.astype('long')]
            # input = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]
            # for i in range(3):
            #     for j in range(shape[i]):
            #         sampl = [[np.random.random_integers(low=0, high=1008),np.random.random_integers(low=0, high=3456)]]#, size=(1,2)
            #         coord[i] = np.append(coord[i], sampl ,axis=0).astype("long")
            
            # print("coord[0].shape:",coord[0].shape)
            # print("coord[1].shape:",coord[1].shape)
            # print("coord[2].shape:",coord[2].shape)
            # 
            # print("coords[0]:",coords[0])
            # print("coord[0]:",coord[0])
            # 
            # print("coords[1]:",coords[1])
            # print("coord[1]:",coord[1])
            # 
            # print("coords[2]:",coords[2])
            # print("coord[2]:",coord[2])
            
            # print("truth_list:",truth_list)
            
            data = [coords,inputs]
            full_image_list.append(data)
            runs.append(ev_sparse.run())
            subruns.append(subrun)
            events.append(event)
            truth.append(truth_list)
            filepaths.append(self.infile)
            entries.append(i)
        print()
        print("MAX LEN:",max_len)
        len_eff = len(full_image_list)
        for i in range(len_eff):
            # print("padding entry ",i)
            for j in range(0,3):
                full_image_list[i][0][j] = np.append(full_image_list[i][0][j],np.zeros((max_len-full_image_list[i][0][j].shape[0],2),dtype=np.float32),0)
                full_image_list[i][1][j] = np.append(full_image_list[i][1][j],np.zeros((max_len-full_image_list[i][1][j].shape[0],1),dtype=np.float32),0)
        return full_image_list, runs, subruns, events, truth, filepaths, entries, max_len, len_eff
    
    def split_batch(self, batch):
        print("DEVICE IN SPLITBATCH:",self.device)
        
        entries = batch[2]
        rse = batch[3]
        
        truth_list_temp = batch[1]
        truth_list = []
        for truth in truth_list_temp:
            truth = truth.to(self.device)
            truth_list.append(truth)

        split_batch = []
        # print("batch:",batch)
        for i in range(0, self.batchsize):
            pts = [0,0,0]
            done = [False,False,False]
            # print("curr_len from:",batch[0][1][0][i].shape, batch[0][1][1][i].shape, batch[0][1][2][i].shape)
            curr_len = max(batch[0][1][0][i].shape[0], batch[0][1][1][i].shape[0], batch[0][1][2][i].shape[0])
            # print("curr_len:",curr_len)
            for k in range(0,curr_len):
                # print("curr_len:",curr_len)
                if done[0] == True and done[1] == True and done[2] == True:
                    break
                for j in range(0,3):
                    # print("j,i,k:",j,i,k)
                    # print("done:",done)
                    # print("thingy:",k == batch[0][1][j][i].shape[0])
                    if k == batch[0][1][j][i].shape[0] and k != curr_len:
                        # print("Should be done thingy, k:",k)
                        pts[j] = k
                        done[j] = True
                    elif done[j] == False and torch.all(batch[0][0][j][i][k] == torch.tensor((0,0),dtype=torch.float64)) and k != 0: # (batch[0][0][j][i][k][1] != 0 and batch[0][0][j][i][k][0] != 0) and k != 0
                        # print("Should be done oldboy, k:",k)
                        pts[j] = k
                        done[j] = True
                        
            print("pts:",pts)
            if self.device == torch.device("cpu"):
                coord_t = [torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()]
                input_t = [torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()]
            else:
                coord_t = [torch.cuda.FloatTensor(), torch.cuda.FloatTensor(), torch.cuda.FloatTensor()]
                input_t = [torch.cuda.FloatTensor(), torch.cuda.FloatTensor(), torch.cuda.FloatTensor()]
            for j in range(3):
                # coord_t[j] = torch.tensor(batch[0][0][j][i], device=device)
                # input_t[j] = torch.tensor(batch[0][1][j][i], device=device)
                input_t[j] = torch.tensor(torch.split(batch[0][1][j][i],(0,pts[j],curr_len-pts[j]))[1], device=self.device)
                coord_t[j] = torch.tensor(torch.split(batch[0][0][j][i],(0,pts[j],curr_len-pts[j]))[1], device=self.device)
            split_batch.append([coord_t,input_t])
        return split_batch, truth_list, entries, rse
