import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp
# from MiscFunctions import cropped_np, unravel_array, reravel_array
# from LArMatchModel import get_larmatch_features

def load_rootfile_training(infile, start_entry = 0, end_entry = -1):
    truthtrack_SCE = ublarcvapp.mctools.TruthTrackSCE()
    # infile = PARAMS['INFILE']
    iocv = larcv.IOManager(larcv.IOManager.kREAD,"io",larcv.IOManager.kTickForward)
    iocv.reverse_all_products() # Do I need this?
    iocv.add_in_file(infile)
    iocv.initialize()

    nentries_cv = iocv.get_n_entries()

    # Get Rid of those pesky IOManager Warning Messages (orig cxx)
	# larcv::logger larcv_logger
	# larcv::msg::Level_t log_level = larcv::msg::kCRITICAL
	# larcv_logger.force_level(log_level)
    full_image_list = []
    runs = []
    subruns   = []
    events    = []
    truth = []
    filepaths = []
    entries   = []


    if end_entry > nentries_cv or end_entry == -1:
        end_entry = nentries_cv
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
    # for i in range(8,9):
        print()
        print("Loading Entry:", i, "of range", start_entry, end_entry)
        print()
        iocv.read_entry(i)
        
        # ev_wire
        ev_sparse    = iocv.get_data(larcv.kProductSparseImage,"nbkrnd_sparse")
        ev_sparse_v = ev_sparse.SparseImageArray()
        # Get Sparse Image to a Numpy Array
        
        # rows = y_wire_image2d.meta().rows()
        # cols = y_wire_image2d.meta().cols()
        # for c in range(cols):
        #     for r in range(rows):
        #         y_wire_np[c][r] = y_wire_image2d.pixel(r,c)
        
        # if PARAMS['USE_CONV_IM']:
        #     y_wire_np = get_larmatch_features(PARAMS, y_wire_image2d)
        # else:
        
        
        ev_sparse_np = larcv.as_sparseimg_ndarray(ev_sparse_v[0])
        # for i in range(0,3):
        #     ev_sparse_np[i] = larcv.as_sparseimg_ndarray(ev_sparse_v[i]) # I am Speed.
            
        print("shape test")
        print(ev_sparse_np.shape)
        # print("printing the data: ")
        # print(ev_sparse_np)
        full_image_list.append(ev_sparse_np)
        
        
        
        subrun = ev_sparse.subrun()
        print("raw subrun:",subrun)
        nc_cc = (subrun%10000)//1000
        print("nc_cc:",nc_cc)
        flavors = (subrun%1000)//100
        print("flavors:",flavors)
        interactionType = (subrun%100)//10
        print("interactionType:",interactionType)
        planes = subrun%10
        print("planes:",planes)
        subrun = subrun//10000
        print("subrun:",subrun)
        
        event = ev_sparse.event()
        print("raw event:", event)
        num_protons = (event%10000)//1000
        print("num_protons:",num_protons)
        num_neutrons = (event%1000)//100
        print("num_neutrons:", num_neutrons)
        num_pion_charged = (event%100)//10
        print("num_pion_charged:", num_pion_charged)
        num_pion_neutral = event%10
        print("num_pion_neutral:",num_pion_neutral)
        event = event//10000
        print("event:",event)
        truth_list = [nc_cc, flavors,interactionType, num_protons, num_pion_charged, num_pion_neutral, num_neutrons, planes]
        
        runs.append(ev_sparse.run())
        subruns.append(subrun)
        events.append(event)
        truth.append(truth_list)
        filepaths.append(infile)
        entries.append(i)
        
        # print("SHAPE TEST")
        # print(y_wire_np.shape)
    return full_image_list, runs, subruns, events, truth, filepaths, entries


def split_into_planes(full_image_list, start_entry = 0, end_entry = -1):
    
    coords_u = np.empty((0,2))
    coords_v = np.empty((0,2))
    coords_y = np.empty((0,2))
    inputs_u = np.empty((0,1))
    inputs_v = np.empty((0,1))
    inputs_y = np.empty((0,1))
    coords = [coords_u, coords_v, coords_y]
    inputs = [inputs_u, inputs_v, inputs_y]
    
    
    if end_entry > len(full_image_list) or end_entry == -1:
        end_entry = len(full_image_list)
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
        coords = [coords_u.astype('float32'), coords_v.astype('float32'), coords_y.astype('float32')]
        inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]
        print("index:",i)
        sparse_data = full_image_list[i]
        # print("full sparse_data:")
        # print(sparse_data)
        print("sparse_data.shape",sparse_data.shape)
        pts = sparse_data.shape[0]
        for j in range (0,pts):
            for k in range(2,5):
                if sparse_data[j,k] != 0:
                    coords[k-2] = np.append(coords[k-2],[[sparse_data[j,0],sparse_data[j,1]]],axis=0)
                    inputs[k-2] = np.append(inputs[k-2],[[sparse_data[j,k]]],axis=0)
        # print("coords[0]:",coords[0])
        # print("inputs[0]:",inputs[0])
        # print("coords[1]:",coords[1])
        # print("inputs[1]:",inputs[1])
        # print("coords[2]:",coords[2])
        # print("inputs[2]:",inputs[2])
    img_list = [coords, inputs]
    return img_list
    
def get_truth_planes(truth, start_entry = 0, end_entry = -1):
    
    if end_entry > len(truth) or end_entry == -1:
        end_entry = len(truth)
    if start_entry > end_entry or start_entry < 0:
        start_entry = 0
    for i in range(start_entry, end_entry):
        truth_list = truth[i]
        planes = truth_list[7]
    
    
    return planes
    
    
    
