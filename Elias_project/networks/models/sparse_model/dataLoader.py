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
Parameters: infile, optional start entry/end entry. Default will run through the
            whole file
Returns: a list of: sparse images, runs, subruns, events, truth, filepaths, entries
"""
def load_rootfile_training(infile, batchsize, nbatches, verbosity, start_entry = 0, end_entry = -1):
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
    max_len = 0
    bad_entries = 0
    good_entries = 0
    # beginning the loop
    while good_entries < nbatches*batchsize:
        print("entries done so far:",good_entries)
        # i = 73848
        i = random.randrange(end_entry)
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
            if flavors == 0 or flavors == 1:
                c_flavors = 2
            elif flavors == 2 or flavors == 3:
                c_flavors = 3

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

        
        if planes == 0:
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
            # print("coords.shape",len(coords[0]))
            if len(coords[0]) <= 20:
                print("INPUT TOO SMALL")
                bad_entries += 1
            elif len(coords[1]) <= 20:
                print("INPUT TOO SMALL")
                bad_entries += 1
            elif len(coords[2]) <= 20:
                print("INPUT TOO SMALL")
                bad_entries += 1
            else:
                good_entries += 1
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
        else:
            print("DEAD PLANES")
            bad_entries += 1
    print()
    print("MAX LEN:",max_len)
    len_eff = len(full_image_list)
    for i in range(start_entry, len_eff):
        # print("padding entry ",i)
        for j in range(0,3):
            full_image_list[i][0][j] = np.append(full_image_list[i][0][j],np.zeros((max_len-full_image_list[i][0][j].shape[0],2),dtype=np.float32),0)
            full_image_list[i][1][j] = np.append(full_image_list[i][1][j],np.zeros((max_len-full_image_list[i][1][j].shape[0],1),dtype=np.float32),0)
    iocv.finalize()
    return full_image_list, runs, subruns, events, truth, filepaths, entries, max_len, len_eff

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
    # truth_list = []
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
    print("Len of full_img_list:",len(full_img_list))
    print("Len of full_truth_list:",len(full_truth_list))
    
    good_entries = []
    
    if end_entry > len(full_img_list) or end_entry == -1:
        end_entry = len(full_img_list)
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
        coords = [coords_u.astype('float32'), coords_v.astype('float32'), coords_y.astype('float32')]
        inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]

        sparse_data = full_img_list[i]
        # print("sparse_data.shape",sparse_data.shape)
        pts = sparse_data.shape[0]
        for j in range(0,pts):
            if sparse_data[j][0] == 0:
                pts = j - 1
                break
        plane_count = full_truth_list[6]
        for j in range (0,pts):
            for k in range(2,5):
                if sparse_data[j,k] != 0:
                    coords[k-2] = np.append(coords[k-2],[[sparse_data[j,0],sparse_data[j,1]]],axis=0)
                    inputs[k-2] = np.append(inputs[k-2],[[sparse_data[j,k]]],axis=0)
        print("coords.shape",len(coords[0]))
        if pts <= 82 or plane_count[0] != 0:
            for j in range(0,3):
                print("coords:",coords[j].shape)
            print("INPUT TOO SMALL")
        else:
            good_entries.append(i)
            planes = [coords,inputs]
            img_list.append(planes)
    print("good_entries:",good_entries)
    for j in range(0,len(full_truth_list)):
        temp_truth_1 = torch.chunk(full_truth_list[j],end_entry)
        temp_truth_t = torch.empty(0,dtype=torch.long)
        for k in good_entries:
            temp_truth_t = torch.cat((temp_truth_t,temp_truth_1[k]),0)
        full_truth_list[j] = temp_truth_t
        
    return img_list, full_truth_list

"""
get_coords_inputs_tensor
Purpose: Takes the image list and converts from np arrays to torch tensors
Parameters: img_list: list of coordinates/values, optional start entry/end
            entry. Default will run through the whole file
Returns: a list of coordinates/values as torch tensors
"""
def split_batch(batch, batchsize, device):
    working_device = torch.device(device)
    truth_list_temp = batch[1]
    truth_list = []
    for truth in truth_list_temp:
        truth = truth.to(device)
        truth_list.append(truth)
        # print("truth:",truth)
        # print("truth.device:",truth.device)
    # print("truth_list:",truth_list)
    # for truth in truth_list:
    #     print("truth:",truth)
    #     print("truth.device:",truth.device)

    split_batch = []
    for i in range(0, batchsize):
        pts = [0,0,0]
        done = [False,False,False]
        curr_len = max(batch[0][1][0][i].shape[0], batch[0][1][1][i].shape[0], batch[0][1][2][i].shape[0])
        # print("curr_len:",curr_len)
        for k in range(0,curr_len):
            if done[0] == True and done[1] == True and done[2] == True:
                break
            for j in range(0,3):
                # print("batch[0][0][j][i][k]:",batch[0][0][j][i][k])
                # print("torch.tensor((0,0),dtype=torch.float64):",torch.tensor((0,0),dtype=torch.float64))
                # print("boolean:",torch.all(batch[0][0][j][i][k] != torch.tensor((0,0),dtype=torch.float64)) and done[j] == False)
                if torch.all(batch[0][0][j][i][k] == torch.tensor((0,0),dtype=torch.float64)) and k != 0 and done[j] == False: # (batch[0][0][j][i][k][1] != 0 and batch[0][0][j][i][k][0] != 0) and k != 0
                    # print("good value")
                    # print("done[j]:",done[j])
                    # note: [coords/inputs or truth][coords or inputs][plane (0-2)][batch][pts][first coord]
                    # print("k:",k)
                    pts[j] = k
                    done[j] = True
                    
        print("pts:",pts)
        coord_t = [torch.cuda.FloatTensor(), torch.cuda.FloatTensor(), torch.cuda.FloatTensor()]
        input_t = [torch.cuda.FloatTensor(), torch.cuda.FloatTensor(), torch.cuda.FloatTensor()]
        for j in range(3):
            # print("coord_t[j]:",coord_t[j].device)
            # coord_t[j] = torch.split(batch[0][0][j][i],(0,pts[j],curr_len-pts[j]))[1]
            # coord_t[j] = torch.tensor(coord_t[j].detach().clone(), device=device)
            coord_t[j] = torch.tensor(torch.split(batch[0][0][j][i],(0,pts[j],curr_len-pts[j]))[1], device=device)
            # print("temp:",temp.device)
            # coord_t[j] = torch.cuda.FloatTensor(torch.split(batch[0][0][j][i],(0,pts[j],curr_len-pts[j]))[1])

            # coord_t[j] = torch.split(batch[0][0][j][i],(0,pts[j],curr_len-pts[j]))[1]
            # print("coord_t[j]:",coord_t[j].device)

            # print("coord.dtype:",coord_t[j].dtype)
            # coord_t[j] = batch[0][0][j][i]
            
            
            input_t[j] = torch.tensor(torch.split(batch[0][1][j][i],(0,pts[j],curr_len-pts[j]))[1], device=device)
            # input_t[j] = torch.split(batch[0][1][j][i],(0,pts[j],curr_len-pts[j]))[1]
            # print("input.dtypeP",input_t[j].dtype)
            # coord_t[j] = torch.from_numpy(batch[0][0][j][i])
            # input_t[j] = torch.from_numpy(batch[0][1][j][i])
        split_batch.append([coord_t,input_t])

    return split_batch,truth_list




"""
get_truth_planes NOTE: not used, won't work for batchsize > 1
Purpose: pops the dead planes truth info off the truth list for a given entry
Parameters: truth list, entry number
Returns: the truth value for dead planes
"""
def get_truth_planes(truth, entry):
    planes = truth.pop()
    return planes
