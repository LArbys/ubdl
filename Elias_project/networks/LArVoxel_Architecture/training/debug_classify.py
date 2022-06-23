import os,sys
import torch
import numpy as np
from InteractionDataset import InteractionClassifierDataset
from InteractionDataset import planes_collate_fn
import MinkowskiEngine as ME
# import custom_shuffle

# debug the setup of larmatch minkowski model
sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/LArVoxel_Architecture/models")
from InteractionClassifier import InteractionClassifier

# use CPU or GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device("cuda")
#DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# Just create model
model = InteractionClassifier()
#print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ",pytorch_total_params/1.0e6," M")
model = model.to(DEVICE)

# Load some test data and push it through

niter = 1
batch_size = 4
nbatches = 1
verbosity = False
train_file_name = "/media/data/larbys/ebengh01/SparseClassifierTrainingSet_5.root"


trainSet = InteractionClassifierDataset(train_file_name, verbosity)

print("NENTRIES: ",len(trainSet))
dataLoader = torch.utils.data.DataLoader(trainSet,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=planes_collate_fn)
batch = next(iter(dataLoader))


# coords_inputs_t,truth_list,entries,rse = RootFileDataLoader.split_batch(train_file, batch)


# test = InteractionClassifier( filelist=["prepdata/testdata/testdata.root"])
# print("NENTRIES: ",len(test))
# loader = torch.utils.data.DataLoader(test,batch_size=batch_size,collate_fn=InteractionClassifier.collate_fn)
# batch = next(iter(loader))
# print("batch.shape:",batch.shape)

# print("batch:",batch)
# print("len(batch):",len(batch))
# print("data?:",data)

# TODO: ME.utils.batch_sparse_collate each individual plane together, make 3
#       different inputs then pass all 3 as a list to the network
# TODO: make a training script

# print("data blob keys: ",data.keys())
# print("coord: ",data["coord"].shape,"  feat: ",data["feat"].shape)
# print("data:",data)
bcoord_u, bcoord_v, bcoord_y, bfeat_u, bfeat_v, bfeat_y, labels_batch = batch
in_u = ME.TensorField(features=bfeat_u,coordinates=bcoord_u, device=DEVICE)
in_v = ME.TensorField(features=bfeat_v,coordinates=bcoord_v, coordinate_manager=in_u.coordinate_manager, device=DEVICE)
in_y = ME.TensorField(features=bfeat_y,coordinates=bcoord_y, coordinate_manager=in_u.coordinate_manager, device=DEVICE)

input = [in_u, in_v, in_y]

# print(test_in.device)
# print("bfeat_u.ndim:",bfeat_u.ndim)
# print("bcoord_u.ndim:", bcoord_u.ndim)
# test_in = ME.SparseTensor(bfeat_u, bcoord_u)
print("input made")

with torch.no_grad():
    out = model(input)
print("output returned")
print(out)
print(out.F)



# coord_u = batch[0]
# feat_u = batch[3]
# test_in = ME.SparseTensor(feat_u, coord_u)
# # coords, feats = ME.utils.sparse_collate(coord_u,feat_u)
# # sp = ME.TensorField(features=feats,coordinates=coords)
# print("input made")
# #print(sp)

# with torch.no_grad():
#     out = model(sp)
# print("output returned")
# print(out)
# print(out.F)