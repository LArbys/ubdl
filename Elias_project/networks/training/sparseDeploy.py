#!/bin/env python

## IMPORT

# python,numpy
import os,sys
import shutil
import time
import traceback
import numpy as np

# ROOT, larcv
import ROOT
from larcv import larcv

# torch
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
# import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as models
# import torch.nn.functional as F

# tensorboardX
from tensorboardX import SummaryWriter
import socket

# for confusion matrix
import sklearn.metrics as skm
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/models/sparse_model")
from networkmodel_vanilla import SparseClassifier
from DataLoader import RootFileDataLoader
from SparseDataset import SparseClassifierDataset
from loss_sparse_classifier import SparseClassifierLoss
import sparseconvnet as scn



# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=True
RUNPROFILER=False

CHECKPOINT_FILE="/media/data/larbys/ebengh01/checkpoint_gpuOT.2th.tar"
# INPUTFILE_TRAIN="/media/data/larbys/ebengh01/SparseClassifierTrainingSet_2.root" # output_10001.root SparseClassifierTrainingSet_2.root
INPUTFILE_DEPLOY="/media/data/larbys/ebengh01/SparseClassifierValidationSet_2.root" # output_9656.root SparseClassifierValidationSet_2.root
TICKBACKWARD=False
start_iter  = 0
num_iters   = 10
IMAGE_WIDTH=3458 # real image 3456
IMAGE_HEIGHT=1026 # real image 1008, 1030 for not depth concat
BATCHSIZE_DEPLOY=1
NWORKERS_VALID=1
DEVICE_IDS=[0,1] # TODO: get this working for multiple gpu
GPUID=DEVICE_IDS[0]
# map multi-training weights
# CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
#                           "cuda:1":"cuda:1"}
# CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=1000
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights

# set log directory
DateTimeHostname = time.strftime("%b%d_%X") + "_" + socket.gethostname()
logdir = "/media/data/larbys/ebengh01/runs/" + DateTimeHostname
print("logdir:",logdir)
writer = SummaryWriter(logdir=logdir) #logdir="runs/Jul28_13-58-40_goeppert"

def main():

    global best_prec1
    global writer
    global num_iters

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(GPUID))
        torch.cuda.set_device(GPUID)
        CHECKPOINT_MAP_LOCATIONS = "cuda:%d"%(GPUID)
    else:
        DEVICE = torch.device("cpu")
        CHECKPOINT_MAP_LOCATIONS = "cpu"
    print("device:",DEVICE)
    # create model, mark it to run on the GPU
    imgdims = 2
    ninput_features  = 64
    noutput_features = 64
    # self, inputshape, nin_features, nout_features, show_sizes
    model = SparseClassifier( (IMAGE_HEIGHT,IMAGE_WIDTH), 
                           ninput_features, noutput_features,
                           show_sizes=False).to(DEVICE)
    # Resume training option
    print ("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
    checkpoint = torch.load( CHECKPOINT_FILE, map_location="cpu" ) # load weights to gpuid
    best_prec1 = checkpoint["best_prec1"]
    # if CHECKPOINT_FROM_DATA_PARALLEL:
    #     model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
    model.load_state_dict(checkpoint["state_dict"])
    model.to(DEVICE)

    # if not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
    #     model = nn.DataParallel( model, device_ids=DEVICE_IDS ).to(device=DEVICE) # distribute across device_ids

    # uncomment to dump model
    if False:
        print ("Loaded model: ",model)
        # print("model.children:",list(model.children()))
        return
  
    # define loss function (criterion)
    criterion = SparseClassifierLoss(DEVICE).to(device=DEVICE)

    # training parameters
    # lr = 1.0e-3
    # weight_decay = 1.0e-4

    # training length
    batchsize_deploy = BATCHSIZE_DEPLOY#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    iter_per_epoch = None # determined later
    iter_per_valid = 1

    nbatches_per_iterdeploy = 100
    itersize_valid         = batchsize_deploy*nbatches_per_iterdeploy
    validbatches_per_print = 1
    
    # confusion matrix normalization
    normalization = "true" # "none" "pred" "true" "all"
    
    
    # SETUP OPTIMIZER
    # RMSPROP
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                                 lr=lr,
    #                                 weight_decay=weight_decay)

    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True
    
    # TODO: fix NENTRIES, will be hard coded in
    # NENTRIES = iotrain.get_len_eff()
    NENTRIES = 75000 # rough estimate
    print ("Number of entries in validation set: ",NENTRIES)

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_valid)
        if num_iters is None:
            # we set it by the number of request epochs
            num_iters = (epochs-start_epoch)*NENTRIES
        else:
            epochs = num_iters/NENTRIES
    else:
        iter_per_epoch = 1

    print( "Number of epochs: ",epochs)
    print ("Iter per epoch: ",iter_per_epoch)

    with torch.autograd.profiler.profile(enabled=RUNPROFILER) as prof:
       print ("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
       checkpoint = torch.load( CHECKPOINT_FILE, map_location="cpu" )
       best_prec1 = checkpoint["best_prec1"]
       model.load_state_dict(checkpoint["state_dict"])
       # optimizer.load_state_dict(checkpoint['optimizer'])
       # if GPUMODE:
       #    optimizer.cuda(GPUID)
       verbosity = False
       # for param_group in optimizer.param_groups:
       #     print ("lr=%.3e"%(param_group['lr'])),
       #     print()
       # evaluate on validation set
       
       deploy_file = RootFileDataLoader(INPUTFILE_DEPLOY,
                                        DEVICE,
                                        BATCHSIZE_DEPLOY,
                                        nbatches_per_iterdeploy,
                                        verbosity,
                                        rand=False)
       
       try:
           

           # switch to evaluate mode
           model.eval()
           start_entry = 0
           
           for ii in range(start_iter, num_iters):
              print ("MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),)

              # accruacy and loss meters and confusion matrix
              accnames = ("flavors",
                          "Interaction Type",
                          "Num Protons",
                          "Num Charged Pions",
                          "Num Neutral Pions",
                          "Num Neutrons")
              acclabelnames = (["CC NuE","CC NuMu","NC NuE","NC NuMu"],
                           ["QE", "RES", "DIS", "Other"],
                           ["0", "1", "2", ">2"],
                           ["0", "1", "2", ">2"],
                           ["0", "1", "2", ">2"],
                           ["0", "1", "2", ">2"])

              acc_meters = {}
              acc_meters_binary = {}
              acc_label_list = {}
              idx = 0
              for n in accnames:
                  acc_meters[n] = AverageMeter()
                  acc_meters_binary[n] = AverageMeter()
                  acc_label_list[n] = acclabelnames[idx]
                  idx+=1

              lossnames = ("total" ,
                          "fl_loss",
                          "iT_loss",
                          "nP_loss",
                          "nCP_loss",
                          "nNP_loss",
                          "nN_loss")

              loss_meters = {}
              for l in lossnames:
                  loss_meters[l] = AverageMeter()
              
              # confusion_true = [[],[],[],[],[],[]]
              # confusion_pred = [[],[],[],[],[],[]]

              # timers for profiling
              time_meters = {}
              for l in ["batch","data","forward","backward","accuracy"]:
                  time_meters[l] = AverageMeter()
              
              iterstart = time.time()
              nnone = 0
              # print("start entry before loading data:",start_entry)
              # iodeploy = load_classifier_larcvdata( "validation", INPUTFILE_VALID,
              #                                   BATCHSIZE_DEPLOY, NWORKERS_VALID,
              #                                   nbatches_per_iterdeploy, verbosity,
              #                                   input_producer_name="nbkrnd_sparse",
              #                                   true_producer_name="nbkrnd_sparse",
              #                                   tickbackward=TICKBACKWARD,
              #                                   readonly_products=None,
              #                                   start_entry=start_entry, end_entry=-1, rand=False)
              print("start entry:",start_entry)
              iodeploy = SparseClassifierDataset(deploy_file, start_entry=start_entry)
              iodeploy.set_nentries(iodeploy.get_len_eff())
              start_entry+=(BATCHSIZE_DEPLOY*nbatches_per_iterdeploy)
              # print("start entry after loading data:",start_entry)
              batched_data = torch.utils.data.DataLoader(iodeploy, batch_size=BATCHSIZE_DEPLOY, shuffle=True)
              
              confusion_true = [[],[],[],[],[],[]]
              confusion_pred = [[],[],[],[],[],[]]
              acc_hist = {}
              for n in accnames:
                 acc_hist[n] = [[],[],[],[]]

              
              for i in range(0,nbatches_per_iterdeploy):
                  if i == 0:
                      print("confusion_true:",confusion_true)
                      print("confusion_pred:",confusion_pred)
                      print("acc_hist:",acc_hist)
                  print("iiter ",ii," batch ",i," of ",nbatches_per_iterdeploy)
                  batchstart = time.time()
                  tdata_start = time.time()
                   
                  num_good_entries = 0
                  full_predict = [torch.empty((0,3),device=DEVICE), torch.empty((0,4),device=DEVICE), torch.empty((0,4),device=DEVICE), torch.empty((0,4),device=DEVICE), torch.empty((0,4),device=DEVICE), torch.empty((0,4),device=DEVICE)]
                  print("GETTING DATA:")
                  batch = next(iter(batched_data))
                  # print("batch:",batch)
                  coords_inputs_t,truth_list = RootFileDataLoader.split_batch(deploy_file, batch)
                    
                  time_meters["data"].update( time.time()-tdata_start )

                  # compute output
                  tforward = time.time()
                    
                  # for j in range(0,BATCHSIZE_DEPLOY):
                  j = 0
                  coord_t = coords_inputs_t[j][0]
                  input_t = coords_inputs_t[j][1]
                  print("Entry",j,"in batch")
                  with torch.no_grad():
                      predict_t = model.deploy(coord_t, input_t, BATCHSIZE_DEPLOY)
                  # print("predict_t:",predict_t)
                  for k in range(0,len(predict_t)):
                      full_predict[k] = torch.cat((full_predict[k],predict_t[k]),0)
                  confusion_true, confusion_pred = update_confusion(full_predict,truth_list,confusion_true,confusion_pred)
                  # loss calc:
                  with torch.no_grad():
                      fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss = criterion(full_predict, truth_list)
                
                  time_meters["forward"].update(time.time()-tforward)

                  # measure accuracy and update meters
                  # update loss meters
                  loss_meters["total"].update( totloss.item() )
                  loss_meters["fl_loss"].update( fl_loss.item() )
                  loss_meters["iT_loss"].update( iT_loss.item() )
                  loss_meters["nP_loss"].update( nP_loss.item() )
                  loss_meters["nCP_loss"].update( nCP_loss.item() )
                  loss_meters["nNP_loss"].update( nNP_loss.item() )
                  loss_meters["nN_loss"].update( nN_loss.item() )

                  end = time.time()
                  # measure accuracy and update meters
                  acc_values = accuracy(full_predict,
                                        truth_list,
                                        acc_meters,
                                        acc_hist,
                                        acc_meters_binary)

                  # update time meter
                  time_meters["accuracy"].update(time.time()-end)
                
                  # write to tensorboard
                  loss_scalars = { x:y.avg for x,y in loss_meters.items() }
                  writer.add_scalars('data/valid_loss', loss_scalars, i )

                  acc_scalars = { x:y.avg for x,y in acc_meters.items() }
                  writer.add_scalars('data/valid_accuracy', acc_scalars, i )
                  # print("acc_scalars_binary:",acc_scalars_binary)
                  acc_scalars_binary = { x:y.avg for x,y in acc_meters_binary.items() }
                  writer.add_scalars('data/valid_accuracy_binary', acc_scalars_binary, i )

                  # measure elapsed time for batch
                  time_meters["batch"].update(time.time()-batchstart)
                
                  time_scalars = { x:y.avg for x,y in time_meters.items() }
                  writer.add_scalars('data/valid times', time_scalars, i )

                  if validbatches_per_print>0 and i % validbatches_per_print == 0:
                      prep_status_message( "valid-batch", i, acc_meters, loss_meters, time_meters, False )
              save_lists(confusion_true,confusion_pred, acc_hist, ii)
           print("about to make confusion")
           make_confusion_matrix(normalization)
           print("about to make histogram")
           
           make_histogram(accnames, acc_label_list)           

           make_particle_count()
           # make pngs of the first layer of kernels
           # make_kernel_img(model)
           print("done with all that")
           prep_status_message( "Valid-Iter", ii, acc_meters, loss_meters, time_meters, False )

       except ValueError:
           print ("Error in validation routine!")
           # print (ValueError.message)
           print (ValueError.__class__.__name__)
           traceback.print_exc(ValueError)

    print ("FIN")
    print ("PROFILER")
    if RUNPROFILER:
        print (prof)
    writer.close()

def update_confusion(full_predict,truth_list,confusion_true,confusion_pred):
    for i in range(0,len(full_predict)):
        # print("truth_list[i]",truth_list[i])
        # print("truth_list[i][0].item():",truth_list[i][0].item())
        confusion_true[i].append(truth_list[i][0].item())
        confusion_pred[i].append(torch.argmax(full_predict[i][0]).item())
    return confusion_true, confusion_pred

def save_lists(confusion_true,confusion_pred, acc_hist, ii):
    # clear txt files on first iteration
    if ii == 0:
        # print("first iteration")
        open('temp_txts/confusion_true_lists.txt', 'w').close()
        open('temp_txts/confusion_pred_lists.txt', 'w').close()
        open('temp_txts/histogram_accuracy_lists.txt', 'w').close()
    
    with open(r"temp_txts/confusion_true_lists.txt","a") as con_t_file:
        # print("saving confusion true")
        # print(confusion_true)
        for outputs in confusion_true:
            for i in outputs:
                con_t_file.write(str(i))
            con_t_file.write("\n")
    with open(r"temp_txts/confusion_pred_lists.txt","a") as con_p_file:
        # print("saving confusion pred")
        # print(confusion_pred)
        for outputs in confusion_pred:
            for i in outputs:
                con_p_file.write(str(i))
            con_p_file.write("\n")
    with open(r"temp_txts/histogram_accuracy_lists.txt","a") as hist_file:
        # print("saving histogram accuracy")
        # print(acc_hist)
        for names in acc_hist:
            # print("outputs:",names)
            outputs = acc_hist[names]
            for output in outputs:
                # print("output",output)
                for val in output:
                    # print("val:",val)
                    hist_file.write(str(val))
                    hist_file.write(" ")
                hist_file.write("\n")

def make_confusion_matrix(normalization):
    plot_list = []
    label_list = [["CC NuE","CC NuMu","NC NuE","NC NuMu"],
                  ["QE","RES","DIS","Other"],
                  ["0","1","2",">2"],
                  ["0","1","2",">2"],
                  ["0","1","2",">2"],
                  ["0","1","2",">2"]]
    title_list = ["Flavor",
                  "Interaction Type",
                  "Number of Protons",
                  "Number of Charged Pions",
                  "Number of Uncharged Pions",
                  "Number of Neutrons"]
    con_t = [[],[],[],[],[],[]]
    con_p = [[],[],[],[],[],[]]
    with open(r"temp_txts/confusion_true_lists.txt","r") as con_t_file:
        lines = con_t_file.readlines()
        count = 0
        for line in lines:
            count = count % 6
            line = line[:-1] # removes \n from end
            for i in line:
                con_t[count].append(int(i))
            count+=1
    with open(r"temp_txts/confusion_pred_lists.txt","r") as con_p_file:
        lines = con_p_file.readlines()
        count = 0
        for line in lines:
            count = count % 6
            # print("count:",count)
            line = line[:-1] # removes \n from end
            for i in line:
                con_p[count].append(int(i))
            count+=1
    # print("con_t:",con_t)
    # print("con_p:",con_p)
    for i in range(0,len(con_p)):
        # print("making array")
        if normalization == "none":
            array = skm.confusion_matrix(con_t[i],con_p[i], labels=[0,1,2,3])
        else:
            array = skm.confusion_matrix(con_t[i],con_p[i], labels=[0,1,2,3], normalize=normalization)
        # print("making df_cm")
        df_cm = pd.DataFrame(array, range(4), range(4))
        # print("1")
        # plt.figure(figsize=(10,7))
        sn.set(font_scale=1.4)
        # print("2")
        fig,ax = plt.subplots(figsize=(10,7))
        # print("3")
        map = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
        # print("4")
        plt.xlabel("Prediction")
        # print("5")
        plt.ylabel("Truth")
        # print("6")
        plt.title(title_list[i])
        # print("7")
        plt.xticks(ticks=[0.5,1.5,2.5,3.5],labels=label_list[i])
        # print("8")
        plt.yticks(ticks=[0.5,1.5,2.5,3.5],labels=label_list[i])
        # print("9")
        plot_list.append(fig)
        # print("10")
        plt.savefig("confusion_matrices/Confusion_%d.png"%(i))
        # print("11")
        plt.close()
    n = normalization
    # print("about to make pdf:")
    with PdfPages('confusion_matrices/%s_normalized_gpuOT.pdf'%(n)) as pdf:
        for i in range(len(plot_list)):
            pdf.savefig(plot_list[i])

def make_particle_count():
    count_t = [[],[],[],[],[],[]]
    count_p = [[],[],[],[],[],[]]
    with open(r"temp_txts/confusion_true_lists.txt","r") as count_t_file:
        lines = count_t_file.readlines()
        count = 0
        for line in lines:
            count = count % 6
            line = line[:-1] # removes \n from end
            for i in line:
                count_t[count].append(int(i))
            count+=1
    with open(r"temp_txts/confusion_pred_lists.txt","r") as count_p_file:
        lines = count_p_file.readlines()
        count = 0
        for line in lines:
            count = count % 6
            line = line[:-1] # removes \n from end
            for i in line:
                count_p[count].append(int(i))
            count+=1
    # print("truth counts:",count_t)
    # print("prediction counts:",count_p)
    differences = []
    for i in range(len(count_t[0])):
        sum_t = 0
        sum_p = 0
        for j in range(2,6):
            sum_t += count_t[j][i]
            sum_p += count_p[j][i]
        differences.append(sum_p - sum_t)
    # print("differences:",differences)
    
    plt.clf()
    plt.hist(differences, bins=25, range=(-12,12), histtype="step")
    plt.xlabel("Predicted Particles - Truth Particles")
    plt.ylabel("Number per bin")
    plt.title("Diff. between Predicted/Truth Particle Counts")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("histograms/particle_counts_histogram_gpuOT.png")
    plt.close()
    
def make_histogram(accnames, acc_label_list):
    acc_hist_2 = {}
    for n in accnames:
        acc_hist_2[n] = [[],[],[],[]]
    with open(r"temp_txts/histogram_accuracy_lists.txt","r") as hist_file:
        lines = hist_file.readlines()
        count = 0
        for line in lines:
            count = count % 24
            line = line[:-1] # removes \n from end
            num = ""
            for i in line:
                if i != " " and i != "\n":
                    num = num + i
                else:
                    acc_hist_2[accnames[count//4]][count%4].append(float(num))
                    num = ""
            count+=1
    for n in accnames:
        plt.clf()
        plt.hist(acc_hist_2[n], bins=20, range=(0,1), histtype="step", label=acc_label_list[n])
        plt.xlabel("Prediction Value")
        plt.ylabel("Number per bin")
        plt.title(n)
        plt.legend()
        plt.tight_layout()
        plt.savefig("histograms/%s_histogram_gpuOT.png"%(n))
        plt.close()

def make_kernel_img(model):
    no_of_layers=0
    conv_layers=[]
     
    model_children=list(model.children())
    # print("model_children:",model_children)

    for child in model_children:
        if type(child)==scn.SubmanifoldConvolution:
            no_of_layers+=1
            conv_layers.append(child)
        else:
            sub_children = list(child.children())
            for s_child in sub_children:
                if type(s_child) == scn.SubmanifoldConvolution:
                    no_of_layers+=1
                    conv_layers.append(s_child)
                elif type(s_child)==scn.Sequential:
                    for layer in s_child.children():
                        if type(layer)==scn.SubmanifoldConvolution:
                            no_of_layers+=1
                            conv_layers.append(layer)
    kernel_list = []
    conv = conv_layers[0]
    kernels = conv.weight.detach().cpu()
    # print("kernels:",kernels)
    kernels = kernels - kernels.min()
    kernels = kernels / kernels.max()

    # for conv in conv_layers:
    for i in range(kernels.shape[3]):
        kernel = []
        for j in range(kernels.shape[0]):
            kernel.append(kernels[j][0][0][i])
        kernel_t = torch.tensor(kernel).view((7,7))
        kernel_a = kernel_t
        sn.set(font_scale=1.4)
        img = plt.imshow(kernel_t)
        plt.savefig("kernels/1st_kernel_%d.png"%(i))
        plt.close()


# def save_checkpoint(state, is_best, p, filename='/media/data/larbys/ebengh01/checkpoint.pth.tar'):
#     if p>0:
#         filename = "/media/data/larbys/ebengh01/checkpoint.%dth.tar"%(p)
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.tar')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(predict,true,acc_meters,acc_hist,acc_meters_binary):
    """Computes the accuracy metrics."""
    # inputs:
    #  assuming all pytorch tensors
    # metrics:
    
    profile = False

    # needs to be as gpu as possible!
    if profile:
        start = time.time()
    sums = [0,0,0,0,0,0]

    for i in range(0,true[0].shape[0]):
        acc_meters["flavors"].update( predict[0][i][true[0][i]] )
        acc_meters["Interaction Type"].update( predict[1][i][true[1][i]] )
        acc_meters["Num Protons"].update( predict[2][i][true[2][i]] )
        acc_meters["Num Charged Pions"].update( predict[3][i][true[3][i]] )
        acc_meters["Num Neutral Pions"].update( predict[4][i][true[4][i]] )
        acc_meters["Num Neutrons"].update( predict[5][i][true[5][i]] )
        
        acc_hist["flavors"][true[0][i]].append(predict[0][i][true[0][i]].item())
        acc_hist["Interaction Type"][true[1][i]].append(predict[1][i][true[1][i]].item())
        acc_hist["Num Protons"][true[2][i]].append(predict[2][i][true[2][i]].item())
        acc_hist["Num Charged Pions"][true[3][i]].append(predict[3][i][true[3][i]].item())
        acc_hist["Num Neutral Pions"][true[4][i]].append(predict[4][i][true[4][i]].item())
        acc_hist["Num Neutrons"][true[5][i]].append(predict[5][i][true[5][i]].item())
        
        if torch.argmax(predict[0][i]).item() == true[0][i]:
            sums[0]+=1
        if torch.argmax(predict[1][i]).item() == true[1][i]:
            sums[1]+=1
        if torch.argmax(predict[2][i]).item() == true[2][i]:
            sums[2]+=1
        if torch.argmax(predict[3][i]).item() == true[3][i]:
            sums[3]+=1
        if torch.argmax(predict[4][i]).item() == true[4][i]:
            sums[4]+=1
        if torch.argmax(predict[5][i]).item() == true[5][i]:
            sums[5]+=1
    acc_meters_binary["flavors"].update(sums[0])
    acc_meters_binary["Interaction Type"].update(sums[1])
    acc_meters_binary["Num Protons"].update(sums[2])
    acc_meters_binary["Num Charged Pions"].update(sums[3])
    acc_meters_binary["Num Neutral Pions"].update(sums[4])
    acc_meters_binary["Num Neutrons"].update(sums[5])
    
    if profile:
        torch.cuda.synchronize()
        start = time.time()

    return acc_meters["flavors"],acc_meters["Interaction Type"],acc_meters["Num Protons"],acc_meters["Num Charged Pions"],acc_meters["Num Neutral Pions"],acc_meters["Num Neutrons"]

# def dump_lr_schedule( startlr, numepochs ):
#     for epoch in range(0,numepochs):
#         lr = startlr*(0.5**(epoch//300))
#         if epoch%10==0:
#             print ("Epoch [%d] lr=%.3e"%(epoch,lr))
#     print ("Epoch [%d] lr=%.3e"%(epoch,lr))
#     return

def prep_status_message( descripter, iternum, acc_meters, loss_meters, timers, istrain ):
    print ("------------------------------------------------------------------------")
    print (" Iter[",iternum,"] ",descripter)
    print ("  Time (secs): iter[%.2f] batch[%.3f] Forward[%.3f/batch] Backward[%.3f/batch] Acc[%.3f/batch] Data[%.3f/batch]"%(timers["batch"].sum,
                                                                                                                             timers["batch"].avg,
                                                                                                                             timers["forward"].avg,
                                                                                                                             timers["backward"].avg,
                                                                                                                             timers["accuracy"].avg,
                                                                                                                             timers["data"].avg))
    print ("  Loss: Total[%.2f]"%(loss_meters["total"].avg))
    print ("  Accuracy: flavors[%.1f] Interaction Type[%.1f] Num Protons[%.1f] Num Charged Pions[%.1f] Num Neutral Pions[%.1f] Num Neutrons[%.1f]"%(acc_meters["flavors"].avg*100,acc_meters["Interaction Type"].avg*100,acc_meters["Num Protons"].avg*100,acc_meters["Num Charged Pions"].avg*100,acc_meters["Num Neutral Pions"].avg*100,acc_meters["Num Neutrons"].avg*100)
)
    print ("------------------------------------------------------------------------")


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()