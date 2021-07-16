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

sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/models/sparse_model")
from networkmodel import SparseClassifier
from SparseClassDataset import load_classifier_larcvdata
from loss_sparse_classifier import SparseClassifierLoss
import dataLoader as dl
# import custom_shuffle


# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False

CHECKPOINT_FILE="model_best.tar"
INPUTFILE_TRAIN="/media/data/larbys/ebengh01/SparseClassifierTrainingSet.root" # output_10001.root SparseClassifierTrainingSet.root
INPUTFILE_VALID="/media/data/larbys/ebengh01/SparseClassifierValidationSet.root" # output_9656.root SparseClassifierValidationSet.root
TICKBACKWARD=False
PLANE = 0
start_iter  = 0
num_iters   = 10000
IMAGE_WIDTH=3458 # real image 3456
IMAGE_HEIGHT=1026 # real image 1008, 1030 for not depth concat
BATCHSIZE_TRAIN=8
BATCHSIZE_VALID=8
NWORKERS_TRAIN=1
NWORKERS_VALID=1
ADC_THRESH=0.0
DEVICE_IDS=[0,1] # TODO: get this working for multiple gpu
GPUID=DEVICE_IDS[1]
# map multi-training weights
CHECKPOINT_MAP_LOCATIONS={"cuda:0":"cuda:0",
                          "cuda:1":"cuda:1"}
CHECKPOINT_MAP_LOCATIONS=None
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=1000
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights
writer = SummaryWriter()

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
    lr = 1.0e-3
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
    optimizer = torch.optim.SGD(model.parameters(), lr,
                               momentum=momentum,
                               weight_decay=weight_decay)

    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=lr,
    #                             weight_decay=weight_decay)
    # RMSPROP
    # optimizer = torch.optim.RMSprop(model.parameters(),
    #                                 lr=lr,
    #                                 weight_decay=weight_decay)

    # optimize algorithms based on input size (good if input size is constant)
    cudnn.benchmark = True

    # LOAD THE DATASET
    
    # iotrain = load_classifier_larcvdata( "training", INPUTFILE_TRAIN,
    #                                   BATCHSIZE_TRAIN, NWORKERS_TRAIN,
    #                                   input_producer_name="nbkrnd_sparse",
    #                                   true_producer_name="nbkrnd_sparse",
    #                                   tickbackward=TICKBACKWARD,
    #                                   readonly_products=None )
    # iotrain.set_nentries(iotrain.get_len_eff())
    # iovalid = load_classifier_larcvdata( "validation", INPUTFILE_VALID,
    #                                   BATCHSIZE_VALID, NWORKERS_VALID,
    #                                   input_producer_name="nbkrnd_sparse",
    #                                   true_producer_name="nbkrnd_sparse",
    #                                   tickbackward=TICKBACKWARD,
    #                                   readonly_products=None )
    # iovalid.set_nentries(iovalid.get_len_eff())
    # print ("pause to give time to feeders")
    
    # TODO: fix NENTRIES, will be hard coded in
    # NENTRIES = iotrain.get_len_eff()
    NENTRIES = 257000 # rough estimate
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
            # TODO: load in data
            adjust_learning_rate(optimizer, ii, lr)
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

    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    # switch to train mode
    model.train()
    
    iotrain = load_classifier_larcvdata( "training", INPUTFILE_TRAIN,
                                      batchsize, NWORKERS_TRAIN,
                                      nbatches, verbosity,
                                      input_producer_name="nbkrnd_sparse",
                                      true_producer_name="nbkrnd_sparse",
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )
    iotrain.set_nentries(iotrain.get_len_eff())
    

    nnone = 0
    batched_data = torch.utils.data.DataLoader(iotrain, batch_size=batchsize, shuffle=True)#, shuffle=True

    for i in range(0,nbatches):
        print("iiter ",iiter," batch ",i," of ",nbatches)
        batchstart = time.time()

        # GET THE DATA
        end = time.time()
        time_meters["data"].update(time.time()-end)
        
        full_predict = [torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device)]
        print("GETTING DATA:")
        batch = next(iter(batched_data))
        # print("batch:",batch)
        # print("batch[0]:",batch[0])
        # print("batch[0][0]:",batch[0][0])
        # note: [coords/inputs or truth][coords or inputs][plane (0-2)][batch][pts][first coord]
        # batch[0][0][j][i][k][1]
        # print("device:",device)
        coords_inputs_t,truth_list = dl.split_batch(batch, batchsize, device)
        # print("truth_list[0].device:",truth_list[0].device)
        for j in range(0,batchsize):
            coord_t = coords_inputs_t[j][0]
            input_t = coords_inputs_t[j][1]
            # print("coord_t device:",coord_t[0].device)
            # print("input_t device:",input_t[0].device)
            # print("coord_t device:",coord_t[1].device)
            # print("input_t device:",input_t[1].device)
            # print("coord_t device:",coord_t[2].device)
            # print("input_t device:",input_t[2].device)
            # compute output
            if RUNPROFILER:
                torch.cuda.synchronize()
            end = time.time()
            print("Entry",j,"in batch")
            predict_t = model(coord_t, input_t, batchsize, device)
            
            for k in range(0,len(predict_t)):
                full_predict[k] = torch.cat((full_predict[k],predict_t[k]),0)
                # full_predict[k] = torch.tensor(torch.cat((full_predict[k],predict_t[k]),0), device=device)
        
        # loss calc:
        fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss = criterion(full_predict, truth_list, device)
    
        if RUNPROFILER:
            torch.cuda.synchronize()
        time_meters["forward"].update(time.time()-end)
    
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
    
        # measure accuracy and update meters
        # acc_values = accuracy(full_predict,
        #                  truth_list,
        #                  acc_meters)
    
        # update time meter
        time_meters["accuracy"].update(time.time()-end)
        
        # write to tensorboard
        loss_scalars = { x:y.avg for x,y in loss_meters.items() }
        writer.add_scalars('data/train_loss', loss_scalars, iiter )
        
        # acc_scalars = { x:y.avg for x,y in acc_meters.items() }
        # writer.add_scalars('data/train_accuracy', acc_scalars, iiter )

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)

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

    # timers for profiling
    time_meters = {}
    for l in ["batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    # switch to evaluate mode
    # model.eval()

    iterstart = time.time()
    nnone = 0
    
    iovalid = load_classifier_larcvdata( "validation", INPUTFILE_VALID,
                                      BATCHSIZE_VALID, NWORKERS_VALID,
                                      nbatches, verbosity,
                                      input_producer_name="nbkrnd_sparse",
                                      true_producer_name="nbkrnd_sparse",
                                      tickbackward=TICKBACKWARD,
                                      readonly_products=None )
    iovalid.set_nentries(iovalid.get_len_eff())
    
    batched_data = torch.utils.data.DataLoader(iovalid, batch_size=batchsize, shuffle=True)
    
    
    for i in range(0,nbatches):
        print("iiter ",iiter," batch ",i," of ",nbatches)
        batchstart = time.time()

        tdata_start = time.time()
        
        num_good_entries = 0
        full_predict = [torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device), torch.empty((0,4),device=device)]
        print("GETTING DATA:")
        batch = next(iter(batched_data))
        coords_inputs_t,truth_list = dl.split_batch(batch, batchsize, device)
        
        time_meters["data"].update( time.time()-tdata_start )

        # compute output
        tforward = time.time()
        
        for j in range(0,batchsize):
            coord_t = coords_inputs_t[j][0]
            input_t = coords_inputs_t[j][1]
            print("Entry",j,"in batch")
            predict_t = model.deploy(coord_t, input_t, batchsize, device)
            for k in range(0,len(predict_t)):
                full_predict[k] = torch.cat((full_predict[k],predict_t[k]),0)

        # loss calc:
        fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss = criterion(full_predict, truth_list, device)
    
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


        # measure accuracy and update meters
        acc_values = accuracy(full_predict,
                         truth_list,
                         acc_meters)

        # update time meter
        end = time.time()
        time_meters["accuracy"].update(time.time()-end)
        
        # write to tensorboard
        loss_scalars = { x:y.avg for x,y in loss_meters.items() }
        writer.add_scalars('data/valid_loss', loss_scalars, iiter )

        acc_scalars = { x:y.avg for x,y in acc_meters.items() }
        writer.add_scalars('data/valid_accuracy', acc_scalars, iiter )

        # measure elapsed time for batch
        time_meters["batch"].update(time.time()-batchstart)

        # measure elapsed time for batch
        time_meters["batch"].update( time.time()-batchstart )
        if print_freq>0 and i % print_freq == 0:
            prep_status_message( "valid-batch", i, acc_meters, loss_meters, time_meters, False )


    prep_status_message( "Valid-Iter", iiter, acc_meters, loss_meters, time_meters, False )

    return loss_meters['total'].avg#,acc_meters['nP_loss'].avg

def save_checkpoint(state, is_best, p, filename='/media/data/larbys/ebengh01/checkpoint.pth.tar'):
    if p>0:
        filename = "/media/data/larbys/ebengh01/checkpoint.%dth.tar"%(p)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.tar')


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


def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (epoch // 300))
    lr = lr
    #lr = lr*0.992
    #print "adjust learning rate to ",lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(predict,true,acc_meters):
    """Computes the accuracy metrics."""
    # inputs:
    #  assuming all pytorch tensors
    # metrics:

    
    profile = False

    # needs to be as gpu as possible!
    if profile:
        start = time.time()
    
    for i in range(0,true[0].shape[0]):
        acc_meters["flavors"].update( predict[0][i][true[0][i]] )
        acc_meters["Interaction Type"].update( predict[1][i][true[1][i]] )
        acc_meters["Num Protons"].update( predict[2][i][true[2][i]] )
        acc_meters["Num Charged Pions"].update( predict[3][i][true[3][i]] )
        acc_meters["Num Neutral Pions"].update( predict[4][i][true[4][i]] )
        acc_meters["Num Neutrons"].update( predict[5][i][true[5][i]] )


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