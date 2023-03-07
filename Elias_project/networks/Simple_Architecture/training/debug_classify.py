import os,sys
import torch
import numpy as np
from InteractionDataset import InteractionClassifierDataset
from InteractionDataset import planes_collate_fn
import MinkowskiEngine as ME
# import custom_shuffle

# debug the setup of larmatch minkowski model
sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/Simple_Architecture/models")
from SimpleInteractionClassifier import InteractionClassifier

# use CPU or GPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# DEVICE = torch.device("cuda")
# DEVICE = torch.device("cpu")
print("Device:", DEVICE)

# Just create model
model = InteractionClassifier()
#print(model)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of parameters: ",pytorch_total_params/1.0e6," M")
model = model.to(DEVICE)

# Load some test data and push it through

niter = 1
batch_size = 2
nbatches = 1
verbosity = False
train_file_name = "/media/data/larbys/ebengh01/output_10001.root"


trainSet = InteractionClassifierDataset(train_file_name, verbosity)

print("NENTRIES: ",len(trainSet))
dataLoader = torch.utils.data.DataLoader(trainSet,
                                         batch_size=batch_size,
                                         shuffle=True,
                                         collate_fn=ME.utils.batch_sparse_collate)
coords, feats, labels = next(iter(dataLoader))

# input = ME.SparseTensor(features=feats, coordinates=coords, device=DEVICE)
input = ME.TensorField(features=feats, coordinates=coords, device=DEVICE)

# in_u = ME.TensorField(features=bfeat_u,coordinates=bcoord_u, device=DEVICE)
# in_v = ME.TensorField(features=bfeat_v,coordinates=bcoord_v, coordinate_manager=in_u.coordinate_manager, device=DEVICE)
# in_y = ME.TensorField(features=bfeat_y,coordinates=bcoord_y, coordinate_manager=in_u.coordinate_manager, device=DEVICE)

# input = [in_u, in_v, in_y]

# print(test_in.device)
# print("bfeat_u.ndim:",bfeat_u.ndim)
# print("bcoord_u.ndim:", bcoord_u.ndim)
# test_in = ME.SparseTensor(bfeat_u, bcoord_u)
print("input made")
print("input:",input)

indices = []
for i in range(batch_size-1):
    indices.append(np.argwhere(np.array(input.coordinates.cpu())[:,0] == i).flatten()[-1])
print(indices)

split_coords = np.split(np.array(input.coordinates.cpu()), indices)
split_feats = np.split(np.array(input.features.cpu()), indices)

# print("split_coords:",split_coords)
# print("split_feats:", split_feats)

# for coordinate in split_coords:
#     print("Coordinate shape:", coordinate.shape)

# for feature in split_feats:
#     print("Feature shape:", feature.shape)


model.train()
with torch.no_grad():
    out = model(input)
print("output returned")
# print(out)
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