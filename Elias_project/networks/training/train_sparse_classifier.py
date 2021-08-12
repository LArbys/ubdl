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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn.functional as F

# tensorboardX
from tensorboardX import SummaryWriter
import socket


sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/models/sparse_model")
from networkmodel_big_kernels import SparseClassifier
from SparseClassDataset import load_classifier_larcvdata
from loss_sparse_classifier import SparseClassifierLoss
import dataLoader as dl
import custom_shuffle


# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False

CHECKPOINT_FILE="/media/data/larbys/ebengh01/checkpoint.150th.tar"
INPUTFILE_TRAIN="/media/data/larbys/ebengh01/SparseClassifierTrainingSet_3.root" # output_10001.root SparseClassifierTrainingSet.root
INPUTFILE_VALID="/media/data/larbys/ebengh01/SparseClassifierValidationSet_2.root" # output_9656.root SparseClassifierValidationSet.root
TICKBACKWARD=False
PLANE = 0
start_iter  = 0
num_iters   = 20
IMAGE_WIDTH=3458 # real image 3456
IMAGE_HEIGHT=1026 # real image 1008, 1030 for not depth concat
BATCHSIZE_TRAIN=8
BATCHSIZE_VALID=8
NWORKERS_TRAIN=1
NWORKERS_VALID=1
ADC_THRESH=0.0
DEVICE_IDS=[0,1] # TODO: get this working for multiple gpu
GPUID=DEVICE_IDS[0]
# map multi-training weights
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=150 # roughly every day
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights

# set log directory
DateTimeHostname = time.strftime("%b%d_%X") + "_" + socket.gethostname()
logdir = "/media/data/larbys/ebengh01/runs/" + DateTimeHostname
print("logdir:",logdir)
writer = SummaryWriter(logdir=logdir)

def main():

    global best_prec1
    global writer
    global num_iters

    if GPUMODE:
        DEVICE = torch.device("cuda:%d"%(GPUID))
        torch.cuda.set_device(GPUID)
    else:
        DEVICE = torch.device("cpu")
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
    if RESUME_FROM_CHECKPOINT:
        print ("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
        checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        # if CHECKPOINT_FROM_DATA_PARALLEL:
        #     model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])

    # if not CHECKPOINT_FROM_DATA_PARALLEL and len(DEVICE_IDS)>1:
    #     model = nn.DataParallel( model, device_ids=DEVICE_IDS ).to(device=DEVICE) # distribute across device_ids

    # uncomment to dump model
    if False:
        print ("Loaded model: ",model)
        return

    # define loss function (criterion) and optimizer
    criterion = SparseClassifierLoss().to(device=DEVICE)
    
    # training parameters
    lr = 1.0e-1
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchsize_train = BATCHSIZE_TRAIN
    batchsize_valid = BATCHSIZE_VALID#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    iter_per_epoch = None # determined later
    iter_per_valid = 10


    nbatches_per_itertrain = 5
    itersize_train         = batchsize_train*nbatches_per_itertrain
    trainbatches_per_print = -1

    nbatches_per_itervalid = 5
    itersize_valid         = batchsize_valid*nbatches_per_itervalid
    validbatches_per_print = -1

    # SETUP OPTIMIZER

    # SGD w/ momentum
    # optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)

    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=lr,
    #                             weight_decay=weight_decay)
    # RMSPROP
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True

        
    NENTRIES = 337567 # Taken from rootfile
    print ("Number of entries in training set: ",NENTRIES)

    if NENTRIES>0:
        iter_per_epoch = NENTRIES/(itersize_train)
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

        # Resume training option
        if RESUME_FROM_CHECKPOINT:
           print ("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
           checkpoint = torch.load( CHECKPOINT_FILE, map_location=CHECKPOINT_MAP_LOCATIONS )
           best_prec1 = checkpoint["best_prec1"]
           model.load_state_dict(checkpoint["state_dict"])
           optimizer.load_state_dict(checkpoint['optimizer'])
        # if GPUMODE:
        #    optimizer.cuda(GPUID)
        verbosity = False
        
        for ii in range(start_iter, num_iters):
            lr = adjust_learning_rate(optimizer, ii, lr)
            print ("MainLoop Iter:%d Epoch:%d.%d "%(ii,ii/iter_per_epoch,ii%iter_per_epoch),)
            for param_group in optimizer.param_groups:
                print ("lr=%.3e"%(param_group['lr'])),
                print()
            
            # train for one iteration
            try:
                _ = train(INPUTFILE_TRAIN, DEVICE, BATCHSIZE_TRAIN, model,
                          criterion, optimizer,
                          nbatches_per_itertrain, ii, trainbatches_per_print, NWORKERS_TRAIN, TICKBACKWARD, verbosity)
          
            except ValueError:
                print ("Error in training routine!")
                # print (ValueError.message)
                print (ValueError.__class__.__name__)
                traceback.print_exc(ValueError)
                break

            # evaluate on validation set
            if ii%iter_per_valid==0 and ii>0:
                try:
                    totloss = validate(INPUTFILE_VALID, DEVICE, BATCHSIZE_VALID, model,
                              criterion, optimizer,
                              nbatches_per_itervalid, ii, validbatches_per_print, NWORKERS_VALID, TICKBACKWARD, verbosity)
                except ValueError:
                    print ("Error in validation routine!")
                    # print (ValueError.message)
                    print (ValueError.__class__.__name__)
                    traceback.print_exc(ValueError)
                    break
            
                # remember best prec@1 and save checkpoint
                prec1   = totloss
                is_best =  prec1 < best_prec1
                best_prec1 = max(prec1, best_prec1)
            
                # check point for best model
                if is_best:
                    print ("Saving best model")
                    save_checkpoint({
                        'iter':ii,
                        'epoch': ii/iter_per_epoch,
                        'state_dict': model.state_dict(),
                        'best_prec1': best_prec1,
                        'optimizer' : optimizer.state_dict(),
                    }, is_best, -1)

            # periodic checkpoint
            if ii>0 and ii%ITER_PER_CHECKPOINT==0:
                print ("saving periodic checkpoint")
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, ii)
            if ii==2505:
                print ("saving periodic checkpoint")
                save_checkpoint({
                    'iter':ii,
                    'epoch': ii/iter_per_epoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, False, 0, filename="/media/data/larbys/ebengh01/checkpoint2505.pth.tar")
            # flush the print buffer after iteration
            sys.stdout.flush()

        # end of profiler context
        print ("saving last state")
        save_checkpoint({
            'iter':num_iters,
            'epoch': num_iters/iter_per_epoch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, False, num_iters)


    print ("FIN")
    print ("PROFILER")
    if RUNPROFILER:
        print (prof)
    writer.close()


def train(INPUTFILE_TRAIN, device, batchsize, model, criterion, optimizer, nbatches, iiter, print_freq, NWORKERS_TRAIN, TICKBACKWARD, verbosity):
    print("IN TRAIN")
    global writer
    # timers for profiling
    batch_time = AverageMeter() # total for batch
    data_time = AverageMeter()
    forward_time = AverageMeter()
    backward_time = AverageMeter()
    acc_time = AverageMeter()

    # accruacy and loss meters

    accnames = ("flavors",
                "Interaction Type",
                "Num Protons",
                "Num Charged Pions",
                "Num Neutral Pions",
                "Num Neutrons")
    
    acc_meters  = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()

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

    grad_names = ("conv1",
                  "SEResNetB2_1.convA",
                  "final_conv")
    grad_meters = {}
    for g in grad_names:
        grad_meters[g] = AverageMeter()
    
    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()
    iotrain = load_classifier_larcvdata( "training", INPUTFILE_TRAIN,
                                      batchsize, NWORKERS_TRAIN,
                                      nbatches, verbosity,
                                      input_producer_name="nbkrnd_sparse",
                                      true_producer_name="nbkrnd_sparse",
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )
    iotrain.set_nentries(iotrain.get_len_eff())
    
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty(()).uniform_().item()))
    
    rand_sample = custom_shuffle.RandomSampler(iotrain, generator=generator)
    batch_sample = custom_shuffle.BatchSampler(rand_sample, batchsize, True)

    nnone = 0
    batched_data = torch.utils.data.DataLoader(iotrain, batch_sampler=batch_sample)
    
    time_meters["data"].update(time.time()-end)
    
    for i in range(0,nbatches):
        print("iiter ",iiter," batch ",i," of ",nbatches)
        batchstart = time.time()

        # GET THE DATA
        full_predict = [torch.empty((0,3),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device)]
        print("GETTING DATA:")
        batch = next(iter(batched_data))
        coords_inputs_t,truth_list = dl.split_batch(batch, batchsize, device)
        
        for j in range(0,batchsize):
            end = time.time()
            coord_t = coords_inputs_t[j][0]
            input_t = coords_inputs_t[j][1]
            # compute output
            if RUNPROFILER:
                torch.cuda.synchronize()
            print("Entry",j,"in batch")
            predict_t = model(coord_t, input_t, batchsize, device)
            for k in range(0,len(predict_t)):
                full_predict[k] = torch.cat((full_predict[k],predict_t[k]),0)
            time_meters["forward"].update(time.time()-end)
        # loss calc:
        fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss = criterion(full_predict, truth_list, device)
        
        if RUNPROFILER:
            torch.cuda.synchronize()
        
        # compute gradient and do SGD step
        if RUNPROFILER:
            torch.cuda.synchronize()
        end = time.time()
        optimizer.zero_grad()
        totloss.backward()
        optimizer.step()
        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["backward"].update(time.time()-end)
    
        # measure accuracy and record loss
        end = time.time()
    
        # update loss meters
        loss_meters["total"].update( totloss.item() )
        loss_meters["fl_loss"].update( fl_loss.item() )
        loss_meters["iT_loss"].update( iT_loss.item() )
        loss_meters["nP_loss"].update( nP_loss.item() )
        loss_meters["nCP_loss"].update( nCP_loss.item() )
        loss_meters["nNP_loss"].update( nNP_loss.item() )
        loss_meters["nN_loss"].update( nN_loss.item() )
        
        # print("model.conv1.weight:",model.conv1.weight)
        # print("model.conv1.weight.grad:",model.conv1.weight.grad)
        # print("model.SEResNetB2_1.convA.weight.grad:",model.SEResNetB2_1.convA.weight.grad)
        # 
        # for module in model.SEResNetBN.purpleBlockP1_1:
        #     print("model.SEResNetBN.purpleBlockP1_1.module.weight.grad:",module.weight.grad)
        # for module in model.SEResNetBN.greenBlockP1_1:
        #     print("model.SEResNetBN.greenBlockP1_1.module.weight.grad:",module.weight.grad)
        # for module in model.SEResNetBN.orangeBlockP1_1:
        #     print("model.SEResNetBN.orangeBlockP1_1.module.weight.grad:",module.weight.grad)
        # for module in model.SEResNetBN.blueBlockP1_1:
        #     print("model.SEResNetBN.blueBlockP1_1.module.weight.grad:",module.weight.grad)
        # print("model.final_conv.weight.grad:",model.final_conv.weight.grad)
        
        writer.add_histogram("model.conv1.weight.grad", model.conv1.weight.grad, iiter)
        writer.add_histogram("model.SEResNetB2_1.convA.weight.grad", model.SEResNetB2_1.convA.weight.grad, iiter)
        for module in model.SEResNetBN.purpleBlockP1_1:
            writer.add_histogram("model.SEResNetBN.purpleBlockP1_1.%s"%(module), module.weight.grad, iiter)
        for module in model.SEResNetBN.greenBlockP1_1:
            writer.add_histogram("model.SEResNetBN.greenBlockP1_1.%s"%(module), module.weight.grad, iiter)
        for module in model.SEResNetBN.orangeBlockP1_1:
            writer.add_histogram("model.SEResNetBN.orangeBlockP1_1.%s"%(module), module.weight.grad, iiter)
        for module in model.SEResNetBN.blueBlockP1_1:
            writer.add_histogram("model.SEResNetBN.blueBlockP1_1.%s"%(module), module.weight.grad, iiter)
        writer.add_histogram("model.final_conv.weight.grad", model.final_conv.weight.grad, iiter)
        # writer.add_histogram("", , iiter)
        # writer.add_histogram("", , iiter)
        # writer.add_histogram("", , iiter)
        # writer.add_histogram("", , iiter)
        
        # grad_meters["conv1"].update(model.conv1.weight.grad)
        # grad_meters["SEResNetB2_1.convA"].update(model.SEResNetB2_1.convA.weight.grad)
        # grad_meters["final_conv"].update(model.final_conv.weight.grad)
        # for tag, parm in model.named_parameters():
        #     writer.add_histogram(tag, parm.grad, iiter)
        
        # measure accuracy and update meters
        # acc_values = accuracy(full_predict,
        #                  truth_list,
        #                  acc_meters)
    
        # update time meter
        time_meters["accuracy"].update(time.time()-end)
        print("writing to tensorboard")
        # write to tensorboard
        loss_scalars = { x:y.avg for x,y in loss_meters.items() }
        writer.add_scalars('data/train_loss', loss_scalars, iiter )
        
        # grad_scalars = { x:y.avg for x,y in grad_meters.items() }
        # writer.add_scalars('data/train_grad', grad_scalars, iiter )
        
        # acc_scalars = { x:y.avg for x,y in acc_meters.items() }
        # writer.add_scalars('data/train_accuracy', acc_scalars, iiter )

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)
        
        time_scalars = { x:y.avg for x,y in time_meters.items() }
        writer.add_scalars('data/train time', time_scalars, iiter )

        # print status
        if print_freq>0 and i%print_freq == 0:
            prep_status_message( "train-batch", i, acc_meters, loss_meters, time_meters,True )
    
    prep_status_message( "Train-Iteration", iiter, acc_meters, loss_meters, time_meters, True )
        
    return loss_meters['total'].avg

def validate(INPUTFILE_VALID, device, batchsize, model, criterion, optimizer, nbatches, iiter, print_freq, NWORKERS_VALID, TICKBACKWARD, verbosity):
    print("IN VALIDATE")
    """
    inputs
    ------
    all_data: instance of LArCVDataSet for loading data
    batchsize (int): image (sets) per batch
    model (pytorch model): network
    criterion (pytorch module): loss function
    nbatches (int): number of batches to process
    print_freq (int): number of batches before printing output
    iiter (int): current iteration number of main loop
    outputs
    -------
    average percent of predictions within 5 pixels of truth
    """
    global writer

    # accruacy and loss meters
    accnames = ("flavors",
                "Interaction Type",
                "Num Protons",
                "Num Charged Pions",
                "Num Neutral Pions",
                "Num Neutrons")

    acc_meters  = {}
    acc_meters_binary = {}
    for n in accnames:
        acc_meters[n] = AverageMeter()
        acc_meters_binary[n] = AverageMeter()

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

    # timers for profiling
    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    # switch to evaluate mode
    # model.eval()

    iterstart = time.time()
    nnone = 0
    
    iovalid = load_classifier_larcvdata( "validation", INPUTFILE_VALID,
                                      batchsize, NWORKERS_VALID,
                                      nbatches, verbosity,
                                      input_producer_name="nbkrnd_sparse",
                                      true_producer_name="nbkrnd_sparse",
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )
    iovalid.set_nentries(iovalid.get_len_eff())
    
    generator = torch.Generator()
    generator.manual_seed(int(torch.empty(()).uniform_().item()))
    
    rand_sample = custom_shuffle.RandomSampler(iovalid, generator=generator)
    batch_sample = custom_shuffle.BatchSampler(rand_sample, batchsize, True)
    
    batched_data = torch.utils.data.DataLoader(iovalid, batch_sampler=batch_sample)
    
    
    for i in range(0,nbatches):
        print("iiter ",iiter," batch ",i," of ",nbatches)
        batchstart = time.time()

        tdata_start = time.time()
        
        num_good_entries = 0
        full_predict = [torch.empty((0,3),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device)]
        print("GETTING DATA:")
        batch = next(iter(batched_data))
        coords_inputs_t,truth_list = dl.split_batch(batch, batchsize, device)
        
        time_meters["data"].update( time.time()-tdata_start )

        # compute output
        for j in range(0,batchsize):
            tforward = time.time()
            coord_t = coords_inputs_t[j][0]
            input_t = coords_inputs_t[j][1]
            print("Entry",j,"in batch")
            predict_t = model.deploy(coord_t, input_t, batchsize, device)
            for k in range(0,len(predict_t)):
                full_predict[k] = torch.cat((full_predict[k],predict_t[k]),0)
            time_meters["forward"].update(time.time()-tforward)

        # loss calc:
        fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss = criterion(full_predict, truth_list, device)
    
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
                         acc_meters_binary,
                         batchsize)

        # update time meter
        time_meters["accuracy"].update(time.time()-end)
        
        # write to tensorboard
        loss_scalars = { x:y.avg for x,y in loss_meters.items() }
        writer.add_scalars('data/valid_loss', loss_scalars, iiter )

        acc_scalars = { x:y.avg for x,y in acc_meters.items() }
        writer.add_scalars('data/valid_accuracy', acc_scalars, iiter )
        # print("acc_scalars_binary:",acc_scalars_binary)
        acc_scalars_binary = { x:y.avg for x,y in acc_meters_binary.items() }
        writer.add_scalars('data/valid_accuracy_binary', acc_scalars_binary, iiter )

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)
        
        time_scalars = { x:y.avg for x,y in time_meters.items() }
        writer.add_scalars('data/valid times', time_scalars, iiter )

        if print_freq>0 and i % print_freq == 0:
            prep_status_message( "valid-batch", i, acc_meters, loss_meters, time_meters, False )


    prep_status_message( "Valid-Iter", iiter, acc_meters, loss_meters, time_meters, False )

    return loss_meters['total'].avg#,acc_meters['nP_loss'].avg

def save_checkpoint(state, is_best, p, filename='/media/data/larbys/ebengh01/checkpoint.pth.tar'):
    if p>0:
        filename = "/media/data/larbys/ebengh01/checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '/media/data/larbys/ebengh01/model_best.tar')


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


def adjust_learning_rate(optimizer, iter, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (iter // 300))
    # lr = lr
    if iter % (8440/2) == 0:
        lr = lr/10
        print("Adjust learning rate to ",lr)
    #lr = lr*0.992
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(predict,true,acc_meters,acc_meters_binary,batchsize):
    """Computes the accuracy metrics."""
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
    acc_meters_binary["flavors"].update(sums[0]/batchsize)
    acc_meters_binary["Interaction Type"].update(sums[1]/batchsize)
    acc_meters_binary["Num Protons"].update(sums[2]/batchsize)
    acc_meters_binary["Num Charged Pions"].update(sums[3]/batchsize)
    acc_meters_binary["Num Neutral Pions"].update(sums[4]/batchsize)
    acc_meters_binary["Num Neutrons"].update(sums[5]/batchsize)


    if profile:
        torch.cuda.synchronize()
        start = time.time()

    return acc_meters["flavors"],acc_meters["Interaction Type"],acc_meters["Num Protons"],acc_meters["Num Charged Pions"],acc_meters["Num Neutral Pions"],acc_meters["Num Neutrons"]

def dump_lr_schedule( startlr, numepochs ):
    for epoch in range(0,numepochs):
        lr = startlr*(0.5**(epoch//300))
        if epoch%10==0:
            print ("Epoch [%d] lr=%.3e"%(epoch,lr))
    print ("Epoch [%d] lr=%.3e"%(epoch,lr))
    return

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