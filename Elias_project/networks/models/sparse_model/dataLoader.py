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
Parameters: infile, optional start entry/end entry. Default will run through the
            whole file
Returns: a list of: sparse images, runs, subruns, events, truth, filepaths, entries
"""
def load_rootfile_training(infile, start_entry = 0, end_entry = -1):
    truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()

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
    # beginning the loop
    for i in range(start_entry, end_entry):
        print()
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        print()
        iocv.read_entry(i)

        # ev_wire
        ev_sparse    = iocv.get_data(larcv.kProductSparseImage,"nbkrnd_sparse")
        ev_sparse_v = ev_sparse.SparseImageArray()
        
        # Get Sparse Image to a Numpy Array
        ev_sparse_np = larcv.as_sparseimg_ndarray(ev_sparse_v[0])

        print("shape test")
        print(ev_sparse_np.shape)
        
        full_image_list.append(ev_sparse_np)
        
        # Extracting subrun truth informtion:
        subrun = ev_sparse.subrun()
        print("raw subrun:",subrun)

        nc_cc = (subrun%10000)//1000
        print("nc_cc:",nc_cc)
        flavors = (subrun%1000)//100
        print("flavors:",flavors)
        if nc_cc == 1:
            if flavors == 0 or flavors == 1:
                c_flavors = 0
            elif flavors == 2 or flavors == 3:
                c_flavors = 1
        else: # nc_cc = 0
            if flavors == 0 or flavors == 1:
                c_flavors = 2
            elif flavors == 2 or flavors == 3:
                c_flavors = 3
        print("c_flavors:",c_flavors)

        interactionType = (subrun%100)//10
        print("interactionType:",interactionType)

        planes = subrun%10
        print("planes:",planes)

        subrun = subrun//10000
        print("subrun:",subrun)

        # Extracting event truth information
        event = ev_sparse.event()
        print("raw event:", event)

        num_protons = (event%10000)//1000
        if num_protons > 2:
            num_protons = 3
        print("num_protons:",num_protons)

        num_neutrons = (event%1000)//100
        if num_neutrons > 2:
            num_neutrons = 3
        print("num_neutrons:", num_neutrons)

        num_pion_charged = (event%100)//10
        if num_pion_charged > 2:
            num_pion_charged = 3
        print("num_pion_charged:", num_pion_charged)

        num_pion_neutral = event%10
        if num_pion_neutral > 2:
            num_pion_neutral = 3
        print("num_pion_neutral:",num_pion_neutral)

        event = event//10000
        print("event:",event)

        truth_list = [c_flavors, interactionType, num_protons, num_pion_charged, num_pion_neutral, num_neutrons, planes]

        runs.append(ev_sparse.run())
        subruns.append(subrun)
        events.append(event)
        truth.append(truth_list)
        filepaths.append(infile)
        entries.append(i)

    return full_image_list, runs, subruns, events, truth, filepaths, entries

"""
split_into_planes
Purpose: Takes a sparse image and splits it into each individual plane, with a
         set of coordinates for each point on the plane. Also discounts entries
         with dead planes and entries that are too small to make it through the
         network
Parameters: all_data: the full data/truth information, optional start entry/end
            entry. Default will run through the whole file
Returns: a list of the sparse coordinates/values as np arrays, a list of the truth values
"""
def split_into_planes(all_data, start_entry = 0, end_entry = -1):
    img_list = []
    truth_list = []
    coords_u = np.empty((0,2))
    coords_v = np.empty((0,2))
    coords_y = np.empty((0,2))
    inputs_u = np.empty((0,1))
    inputs_v = np.empty((0,1))
    inputs_y = np.empty((0,1))
    coords = [coords_u, coords_v, coords_y]
    inputs = [inputs_u, inputs_v, inputs_y]

    full_img_list = all_data[0]
    full_truth_list = all_data[1]

    if end_entry > len(full_img_list) or end_entry == -1:
        end_entry = len(full_img_list)
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
        coords = [coords_u.astype('float32'), coords_v.astype('float32'), coords_y.astype('float32')]
        inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]

        sparse_data = full_img_list[0]
        print("sparse_data.shape",sparse_data.shape)
        pts = sparse_data.shape[0]
        plane_count = full_truth_list[6]
        for j in range (0,pts):
            for k in range(2,5):
                if sparse_data[j,k] != 0:
                    coords[k-2] = np.append(coords[k-2],[[sparse_data[j,0],sparse_data[j,1]]],axis=0)
                    inputs[k-2] = np.append(inputs[k-2],[[sparse_data[j,k]]],axis=0)
        if pts <= 82 or plane_count[0] != 0:
            for i in range(0,3):
                print("coords:",coords[i].shape)
            print("INPUT TOO SMALL")
        else:
            truth = full_truth_list
            planes = [coords,inputs]
            img_list.append(planes)
            truth_list.append(truth)
            
    return img_list, truth_list

"""
get_coords_inputs_tensor
Purpose: Takes the image list and converts from np arrays to torch tensors
Parameters: img_list: list of coordinates/values, optional start entry/end
            entry. Default will run through the whole file
Returns: a list of coordinates/values as torch tensors
"""
def get_coords_inputs_tensor(img_list, start_entry = 0, end_entry = -1):
    if end_entry > len(img_list) or end_entry == -1:
        end_entry = len(img_list)
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    coords_inputs_t = []
    for i in range(start_entry, end_entry):
        coord_t = [torch.Tensor(), torch.Tensor(), torch.Tensor()]
        input_t = [torch.Tensor(), torch.Tensor(), torch.Tensor()]
        for j in range(3):
            coord_t[j] = torch.from_numpy(img_list[i][0][j])
            input_t[j] = torch.from_numpy(img_list[i][1][j])
        coords_inputs_t.append([coord_t,input_t])

    return coords_inputs_t




"""
get_truth_planes NOTE: not used, won't work for batchsize > 1
Purpose: pops the dead planes truth info off the truth list for a given entry
Parameters: truth list, entry number
Returns: the truth value for dead planes
"""
def get_truth_planes(truth, entry):
    planes = truth.pop()
    return planes
