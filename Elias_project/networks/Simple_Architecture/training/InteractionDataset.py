import os,sys,time

import ROOT as rt
import numpy as np
from larcv import larcv
# sys.path.append("/mnt/disk1/nutufts/kmason/sparsenet/ubdl/larflow/larcvdataset")
# from larcvdataset.larcvserver import LArCVServer
import torch
from torch.utils import data as torchdata
import MinkowskiEngine as ME

class InteractionClassifierDataset(torchdata.Dataset):
    def __init__(self, infile, verbosity):
        self.infile = infile
        self.verbosity = verbosity
        
        print("Initializing IOManager for",self.infile)
        self.iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickForward)
        self.iocv.reverse_all_products() # Do I need this?
        self.iocv.add_in_file(self.infile)
        self.iocv.initialize()

        self.quantization_size = 1
        
        self.nentries = self.iocv.get_n_entries()

    def __del__(self):
        print("Deleting IOManager for",self.infile)
        self.iocv.finalize()

    def __len__(self):
        return self.nentries

    def __getitem__(self, idx):
        print("Loading Entry:", idx)
        self.iocv.read_entry(idx)

        # ev_wire
        ev_sparse    = self.iocv.get_data(larcv.kProductSparseImage,"nbkrnd_sparse")
        ev_sparse_v = ev_sparse.SparseImageArray()
        
        # Get Sparse Image to a Numpy Array
        ev_sparse_np = larcv.as_sparseimg_ndarray(ev_sparse_v[0])
        
        sparse_ev_shape = ev_sparse_np.shape
        # print("sparse_ev_shape:",sparse_ev_shape)
        # print("ev_sparse_np:",ev_sparse_np)

        # Extracting subrun truth informtion:
        subrun = ev_sparse.subrun()

        nc_cc = (subrun % 10000) // 1000
        flavors = (subrun % 1000) // 100
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

        interactionType = (subrun % 100) //10

        planes = subrun % 10

        subrun = subrun // 10000

        # Extracting event truth information
        event = ev_sparse.event()

        num_protons = (event % 10000) // 1000
        if num_protons > 2:
            num_protons = 3

        num_neutrons = (event % 1000) // 100
        if num_neutrons > 2:
            num_neutrons = 3

        num_pion_charged = (event % 100) // 10
        if num_pion_charged > 2:
            num_pion_charged = 3

        num_pion_neutral = event % 10
        if num_pion_neutral > 2:
            num_pion_neutral = 3

        event = event // 10000

        truth_list = torch.tensor([c_flavors, interactionType, num_protons, num_pion_charged, num_pion_neutral, num_neutrons, planes])
        # print("truth_list:",truth_list)


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

        
        # coords_u = np.empty((0,2))
        # coords_v = np.empty((0,2))
        # coords_y = np.empty((0,2))
        # inputs_u = np.empty((0,1))
        # inputs_v = np.empty((0,1))
        # inputs_y = np.empty((0,1))
        # coords = [coords_u.astype('long'), coords_v.astype('long'), coords_y.astype('long')]
        # inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]
        # inputs_l = [[], [], []]
        
        # # print(inputs_l)
        # pts = sparse_ev_shape[0]
        # for j in range (0,pts):
        #     for k in range(2,5):
        #         if ev_sparse_np[j,k] != 0:
        #             coord_pair = [[ev_sparse_np[j,1],ev_sparse_np[j,0]]]
        #             coords[k-2] = np.append(coords[k-2], coord_pair, axis=0).astype("long")
        #             inputs[k-2] = np.append(inputs[k-2], [[ev_sparse_np[j,k]]], axis=0)
        #             # inputs_l[k-2].append(ev_sparse_np[j,k])
        # shape = [coords[0].shape[0],coords[1].shape[0],coords[2].shape[0]]
        # # print("shape:",shape)
        # # print("coords[0].shape:",coords[0].shape)
        # # i = torch.tensor(coords[0])
        # # v = torch.tensor(inputs_l[0])
        
        # # print("v:",v.dtype)
        # # s = torch.sparse_coo_tensor(i.t(), v, (3456, 1008))

        # # data_l = []
        # # for i in range(3):
        # coord_u = torch.tensor(coords[0])
        # # val_u = torch.tensor(inputs_l[0])
        # val_u = torch.tensor(inputs[0])
        # # data_u = torch.sparse_coo_tensor(coord_u.t(), val_u, (3456, 1008))

        # coord_v = torch.tensor(coords[1])
        # # val_v = torch.tensor(inputs_l[1])
        # val_v = torch.tensor(inputs[1])
        # # data_v = torch.sparse_coo_tensor(coord_v.t(), val_v, (3456, 1008))

        # coord_y = torch.tensor(coords[2])
        # # val_y = torch.tensor(inputs_l[2])
        # val_y = torch.tensor(inputs[2])
        # # data_y = torch.sparse_coo_tensor(coord_y.t(), val_y, (3456, 1008))
        
        # coord_list = [coord_u, coord_v, coord_y]
        # val_list = [val_u, val_v, val_y]
        # data_l.append(s)
        # print("coords:",coords)
        # print("inputs:",inputs)
        
        # coords_np = np.asarray(coords, dtype=object)
        # inputs_np = np.asarray(inputs, dtype=object)
        # coords_t = torch.tensor(coords)
        # inputs_t = torch.tensor(inputs)


        
        # print("coords_np:",coords_np)
        # print("inputs_np:",inputs_np)
        # print("coords_np[0][0].shape:",coords_np[0][0].shape)
        # print("inputs_np[0][0].shape:",inputs_np[0][0].shape)
        # data = torch.tensor([coords,inputs])
        # data = np.asarray([coords_np,inputs_np])
        # rse = torch.tensor([ev_sparse.run(), subrun, event])
        # planes = torch.tensor([data_u, data_v, data_y])
        # data = torch.tensor([data_u, data_v, data_y, truth_list, rse])
        # TODO: pass each plane individually, make a function to do a batch
        #       collate thing on them. See debug_classify for more

        coords = ev_sparse_np[:,0:2]
        # print("coords.shape:",coords.shape)
        # print("coords:",coords)
        zero_indices = np.argwhere(ev_sparse_np[:,2] == 0).flatten()
        # print("0s indices:", zero_indices)
        feats = ev_sparse_np[:,2].reshape(ev_sparse_np.shape[0], 1)
        # print("feats.shape:",feats.shape)
        # print("feats:",feats)
        pruned_feats = np.delete(feats, zero_indices, axis=0)
        pruned_coords = np.delete(coords, zero_indices, axis=0)
        # print("pruned_feats:", pruned_feats)
        # print("pruned_coords:", pruned_coords)


        # temp2feats = temp_feats.reshape(feats.shape[0] - zero_indices.shape[0], 1)
        # print("temp:feats2:", temp2feats)
        # print("temp:feats.shape:", temp_feats.shape)

        # print("feats.shape:",feats.shape)
        # print("coords.shape:",coords.shape)

        # print("0s indices shape:", zero_indices.shape)
        # print("pruned_feats.shape:", pruned_feats.shape)
        # print("pruned_coords.shape:", pruned_coords.shape)




        return pruned_coords, pruned_feats, truth_list 

def planes_collate_fn(data_labels):
    coord_u, coord_v, coord_y, feat_u, feat_v, feat_y, labels = list(zip(*data_labels))

    # Create batched coordinates for the SparseTensor input
    bcoord_u = ME.utils.batched_coordinates(coord_u)
    bcoord_v = ME.utils.batched_coordinates(coord_v)
    bcoord_y = ME.utils.batched_coordinates(coord_y)
    # bcoords_l = []
    # print("coords:",coords[0])
    # for coord in coords[0]:
    #     print("coord",coord)
    #     bcoords = ME.utils.batched_coordinates(coord)
    #     bcoords_l.append(bcoords)

    

    # Concatenate all lists
    bfeat_u = torch.from_numpy(np.concatenate(feat_u, 0)).float()
    bfeat_v = torch.from_numpy(np.concatenate(feat_v, 0)).float()
    bfeat_y = torch.from_numpy(np.concatenate(feat_y, 0)).float()

    # bfeats_l = []
    # for feat in feats:
    #     feats_batch = torch.from_numpy(np.concatenate(feat, 0)).float()
    #     bfeats_l.append(feats_batch)



    labels_batch = torch.from_numpy(np.concatenate(labels, 0)).int()

    return bcoord_u, bcoord_v, bcoord_y, bfeat_u, bfeat_v, bfeat_y, labels_batch

    
    # def set_nentries(self,nentries):
    #     self.nentries = nentries
    
    # def get_len_eff(self):
    #     len_eff = self.feeder[8]
    #     return len_eff
    
    # def collate_fn(batch):
    #     return batch
