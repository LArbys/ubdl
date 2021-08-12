import random
import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp

import torch
# from MiscFunctions import cropped_np, unravel_array, reravel_array
# from LArMatchModel import get_larmatch_features


"""
load_rootfile_training
Purpose: loads in the rootfile for training/validation
Parameters: infile, batchsize, number of batches, bool verbosity for printing,
            start entry (default 0), end entry (default end of file), bool rand
            (default true)
Returns: a list of: sparse images, runs, subruns, events, truth, filepaths,
                    entries, max pixel length, effective length(length of the image list)
"""
def load_rootfile_training(infile, batchsize, nbatches, verbosity, start_entry = 0, end_entry = -1, rand=True):
    # Initialize the IO Manager:
    iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickForward)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile)
    iocv.initialize()

    nentries_cv = iocv.get_n_entries()

    # set up empty lists for saving data
    full_image_list = []
    runs = []
    subruns   = []
    events    = []
    truth = []
    filepaths = []
    entries   = []

    # entry number checks
    if end_entry > nentries_cv or end_entry == -1:
        end_entry = nentries_cv
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    max_len = 0

    # determine how to pull entries
    if rand:
        entry_list = random.sample(range(start_entry,end_entry),nbatches*batchsize)
    else:
        if (start_entry + nbatches*batchsize) < end_entry:
            entry_list = [*range(start_entry,start_entry + nbatches*batchsize)]
        else:
            entry_list = [*range(start_entry,end_entry)]
    # begin iterating over entries
    for i in entry_list:
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        iocv.read_entry(i)

        # ev_wire
        ev_sparse    = iocv.get_data(larcv.kProductSparseImage,"nbkrnd_sparse")
        ev_sparse_v = ev_sparse.SparseImageArray()
        
        # Get Sparse Image to a Numpy Array
        ev_sparse_np = larcv.as_sparseimg_ndarray(ev_sparse_v[0])

        sparse_ev_shape = ev_sparse_np.shape
        print(sparse_ev_shape)
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
        
        if verbosity:
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
        coords = [coords_u, coords_v, coords_y]
        inputs = [inputs_u, inputs_v, inputs_y]
        coords = [coords_u.astype('long'), coords_v.astype('long'), coords_y.astype('long')]
        # coords = [coords_u.astype('float32'), coords_v.astype('float32'), coords_y.astype('float32')]
        inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]
        pts = ev_sparse_np.shape[0]
        for j in range (0,pts):
            for k in range(2,5):
                if ev_sparse_np[j,k] != 0:
                    coords[k-2] = np.append(coords[k-2],[[ev_sparse_np[j,0],ev_sparse_np[j,1]]],axis=0)
                    inputs[k-2] = np.append(inputs[k-2],[[ev_sparse_np[j,k]]],axis=0)
        print(len(coords[0]))
        print(len(coords[1]))
        print(len(coords[2]))
        data = [coords,inputs]
        full_image_list.append(data)
        runs.append(ev_sparse.run())
        subruns.append(subrun)
        events.append(event)
        truth.append(truth_list)
        filepaths.append(infile)
        entries.append(i)
    print()
    print("MAX LEN:",max_len)
    len_eff = len(full_image_list)
    for i in range(len_eff):
        # print("padding entry ",i)
        for j in range(0,3):
            full_image_list[i][0][j] = np.append(full_image_list[i][0][j],np.zeros((max_len-full_image_list[i][0][j].shape[0],2),dtype=np.float32),0)
            full_image_list[i][1][j] = np.append(full_image_list[i][1][j],np.zeros((max_len-full_image_list[i][1][j].shape[0],1),dtype=np.float32),0)
    iocv.finalize()
    return full_image_list, runs, subruns, events, truth, filepaths, entries, max_len, len_eff

"""
split_batch
Purpose: Takes the batch and splits into truth and img list. Trims 0-padding off
         images, puts all the data in the correct device, puts images into tensors
Parameters: batch, batchsize, device
Returns: a list of coordinates/values as torch tensors and the truth list
"""
def split_batch(batch, batchsize, device):
    working_device = torch.device(device)
    truth_list_temp = batch[1]
    truth_list = []
    for truth in truth_list_temp:
        truth = truth.to(device)
        truth_list.append(truth)

    split_batch = []
    # print("batch:",batch)
    for i in range(0, batchsize):
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
        if device == torch.device("cpu"):
            coord_t = [torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()]
            input_t = [torch.FloatTensor(), torch.FloatTensor(), torch.FloatTensor()]
        else:
            coord_t = [torch.cuda.FloatTensor(), torch.cuda.FloatTensor(), torch.cuda.FloatTensor()]
            input_t = [torch.cuda.FloatTensor(), torch.cuda.FloatTensor(), torch.cuda.FloatTensor()]
        for j in range(3):
            
            # coord_t[j] = torch.tensor(batch[0][0][j][i], device=device)
            # input_t[j] = torch.tensor(batch[0][1][j][i], device=device)
            input_t[j] = torch.tensor(torch.split(batch[0][1][j][i],(0,pts[j],curr_len-pts[j]))[1], device=device)
            coord_t[j] = torch.tensor(torch.split(batch[0][0][j][i],(0,pts[j],curr_len-pts[j]))[1], device=device)
        split_batch.append([coord_t,input_t])

    return split_batch,truth_list