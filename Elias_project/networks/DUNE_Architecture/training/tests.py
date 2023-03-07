
import os,sys
import shutil
import time
import traceback
import numpy as np
import random
import torch
import socket
from tensorboardX import SummaryWriter
import DataLoader
import pdb

# shape = [3, 5, 4]
setnum = 0

shapeList = [[774, 736, 828],
[1286, 911, 909],
[158, 187, 217],
[1953, 1435, 2477],
[1484, 2096, 2772],
[2512, 2567, 2699],
[918, 1195, 706],
[161, 98, 226],
[1678, 2185, 2926],
[412, 1542, 1356],
[1122, 1430, 736],
[76, 596, 538]]

# shapeList = [[3, 5, 4],
# [2,3,5]]

for count, shape in enumerate(shapeList):
    coords_u = np.empty((0,2))
    coords_v = np.empty((0,2))
    coords_y = np.empty((0,2))
    inputs_u = np.empty((0,1))
    inputs_v = np.empty((0,1))
    inputs_y = np.empty((0,1))
    coord = [coords_u.astype('long'), coords_v.astype('long'), coords_y.astype('long')]
    inputs = [inputs_u.astype('float32'), inputs_v.astype('float32'), inputs_y.astype('float32')]
    for i in range(3):
        # inputs[i] = np.random.uniform(low=10, high=100, size=shape[i]).astype('float32')
        for j in range(shape[i]):
            sampl = [[np.random.random_integers(low=0, high=1008),np.random.random_integers(low=0, high=3456)]]#, size=(1,2)
            coord[i] = np.append(coord[i], sampl ,axis=0).astype("long")
            
            num = [np.random.random(size=(1,))]
            # print("num:",num)
            while num[0] < 0.01:
                # print("NUM TOO SMALL AAAAAA")
                num = [np.random.random(size=(1,))]
            num[0] = num[0]*100
            # print("final num:",num)
            inputs[i] = np.append(inputs[i], num ,axis=0).astype("float32")

# [254, 899, 753]
# inputs[0] = np.random.uniform(low=10, high=100, size=inputs[0].shape).astype('float32')
# inputs[1] = np.random.uniform(low=10, high=100, size=inputs[1].shape).astype('float32')
# inputs[2] = np.random.uniform(low=10, high=100, size=inputs[2].shape).astype('float32')


        # inputs[i] = np.append(inputs[i], np.random.uniform(low=10, high=100, size=1) ,axis=0).astype("float32")
# print("inputs:",inputs)
# print(coord)
        print("coord[%d]:"%(i),coord[i])
        print("inputs[%d]:"%(i),inputs[i])
        coordfname = "rand_data/%d/coord[%d].csv"%(count,i)
        print(coordfname)
        np.savetxt(coordfname, coord[i],fmt="%10.5f", delimiter=",")
        
        inputfname = "rand_data/%d/input[%d].csv"%(count,i)
        print(inputfname)
        np.savetxt(inputfname, inputs[i], delimiter=",")


for i in range(2):
    for j in range(3):
        inCoord1 = np.loadtxt("rand_data/%d/coord[%d].csv"%(i,j),delimiter=",")
        inInput1 = np.loadtxt("rand_data/%d/input[%d].csv"%(i,j),delimiter=",")
        inCoord2 = np.empty((0,2))
        for c in inCoord1:
            c2 = [c]
            inCoord2 = np.append(inCoord2, c2 ,axis=0).astype("long")
            
        inInput2 = np.empty((0,1))
        for n in inInput1:
            n2 = [[n]]
            inInput2 = np.append(inInput2, n2, axis=0).astype("float32")
        
        
        print("inCoord2:",inCoord2)
        print("inInput2:",inInput2)
        
        
        
        
        
