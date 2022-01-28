import random
import ROOT
import numpy as np
from larcv import larcv
from larlite import larlite
from array import *
from larlite import larutil
from ublarcvapp import ublarcvapp

import torch

infile = "/media/data/larbys/ebengh01/SparseClassifierTrainingSet_1.root"
batchsize = 1
nbatches = 1
verbosity = True
entry = 339712
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
# if end_entry > nentries_cv or end_entry == -1:
#     end_entry = nentries_cv
# if start_entry > end_entry or start_entry < 0:
#     start_entry = 0
max_len = 0
bad_entries = 0
good_entries = 0
# beginning the loop

i = entry
print("Loading Entry:", i)
print()
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
print("raw subrun:",subrun)

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
print("raw event:", event)

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
    print("nc_cc:",nc_cc)
    print("flavors:",flavors)
    print("c_flavors:",c_flavors)
    print("interactionType:",interactionType)
    print("planes:",planes)
    print("subrun:",subrun)
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
print(len(coords[0]))
print(len(coords[1]))
print(len(coords[2]))
print()
print("MAX LEN:",max_len)
len_eff = len(full_image_list)
for i in range(0, len_eff):
    # print("padding entry ",i)
    for j in range(0,3):
        full_image_list[i][0][j] = np.append(full_image_list[i][0][j],np.zeros((max_len-full_image_list[i][0][j].shape[0],2),dtype=np.float32),0)
        full_image_list[i][1][j] = np.append(full_image_list[i][1][j],np.zeros((max_len-full_image_list[i][1][j].shape[0],1),dtype=np.float32),0)

# return full_image_list, runs, subruns, events, truth, filepaths, entries, max_len, len_eff