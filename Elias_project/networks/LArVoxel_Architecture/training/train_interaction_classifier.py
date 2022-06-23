import os,sys
import shutil
import time
import torch
import numpy as np
from InteractionDataset import InteractionClassifierDataset
from InteractionDataset import planes_collate_fn
import MinkowskiEngine as ME
# import custom_shuffle

# debug the setup of larmatch minkowski model
sys.path.append("/home/ebengh01/ubdl/Elias_project/networks/LArVoxel_Architecture/models")
from InteractionClassifier import InteractionClassifier

from tensorboardX import SummaryWriter
import socket

from loss_LArVoxel import SparseClassifierLoss



# ===================================================
# TOP-LEVEL PARAMETERS
GPUMODE=True
RESUME_FROM_CHECKPOINT=False
RUNPROFILER=False

CHECKPOINT_FILE = "/media/data/larbys/ebengh01/checkpoint_gpuOT.800th.tar"
TRAINFILENAME   = "/media/data/larbys/ebengh01/output_10001.root" # output_10001.root SparseClassifierTrainingSet_5.root
VALIDFILENAME   = "/media/data/larbys/ebengh01/SparseClassifierValidationSet_4.root"
start_iter  = 0
niters   = 3000
IMAGE_WIDTH = 3458 # real image 3456
IMAGE_HEIGHT = 1026 # real image 1008, 1030 for not depth concat
BATCHSIZE_TRAIN = 16
BATCHSIZE_VALID = 16
DEVICE_IDS = [0,1] # TODO: get this working for multiple gpu
GPUID=DEVICE_IDS[0]
CHECKPOINT_FROM_DATA_PARALLEL=False
ITER_PER_CHECKPOINT=2000 # roughly every day
# ===================================================

# global variables
best_prec1 = 0.0  # best accuracy, use to decide when to save network weights

# set log directory
DateTimeHostname = time.strftime("%b%d_%X") + "_" + socket.gethostname()
logdir = "/media/data/larbys/ebengh01/runs/" + DateTimeHostname + "_OT"
print("logdir:",logdir)
tbWriter = SummaryWriter(logdir=logdir)


def main():
    global best_prec1
    # use CPU or GPU
    # DEVICE = torch.device("cuda:%d"%(GPUID) if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda:%d"%(GPUID))
        torch.cuda.set_device(GPUID)
        CHECKPOINT_MAP_LOCATIONS = "cuda:%d"%(GPUID)
    else:
        DEVICE = torch.device("cpu")
        CHECKPOINT_MAP_LOCATIONS = "cpu"
    # DEVICE = torch.device("cuda")
    #DEVICE = torch.device("cpu")
    print("Device:", DEVICE)

    # Just create model
    model = InteractionClassifier().to(DEVICE)
    #print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters: ",pytorch_total_params/1.0e6," M")
    # model = model.to(DEVICE)
    if RESUME_FROM_CHECKPOINT:
        print ("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
        checkpoint = torch.load( CHECKPOINT_FILE, map_location="cpu" ) # load weights to gpuid
        best_prec1 = checkpoint["best_prec1"]
        # if CHECKPOINT_FROM_DATA_PARALLEL:
        #     model = nn.DataParallel( model, device_ids=DEVICE_IDS ) # distribute across device_ids
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device=DEVICE)
    
    # define loss function (criterion) and optimizer
    criterion = SparseClassifierLoss(DEVICE, size_average=True).to(device=DEVICE)

    # training parameters
    lr = 1.0e-1
    momentum = 0.9
    weight_decay = 1.0e-4

    # training length
    batchSizeTrain = BATCHSIZE_TRAIN
    batchSizeValid = BATCHSIZE_VALID#*len(DEVICE_IDS)
    start_epoch = 0
    epochs      = 10
    iter_per_epoch = None # determined later, itersPerEpoch
    iter_per_valid = 10


    nBatchesIterTrain = 5
    iterSizeTrain         = batchSizeTrain*nBatchesIterTrain
    trainbatches_per_print = -1

    nBatchesIterValid = 5
    iterSizeValid         = batchSizeValid*nBatchesIterValid
    validbatches_per_print = -1


    # SGD w/ momentum
    # optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)

    # ADAM
    # betas default: (0.9, 0.999) for (grad, grad^2). smoothing coefficient for grad. magnitude calc.
    # NOTE: ADAMW is better
    # optimizer = torch.optim.Adam(model.parameters(),
    #                             lr=lr,
    #                             weight_decay=weight_decay)
    # RMSPROP
    optimizer = torch.optim.RMSprop(model.parameters(),
                                    lr=lr,
                                    weight_decay=weight_decay)

    # Load some test data and push it through

    # batchSize = 4
    # nbatches = 1
    verbosity = False
    # train_file_name = "/media/data/larbys/ebengh01/SparseClassifierTrainingSet_5.root"


    trainSet = InteractionClassifierDataset(TRAINFILENAME, verbosity)
    NENTRIES = len(trainSet)
    print(f"Number of Entries in Training Set = {NENTRIES}")


    gen = torch.Generator().manual_seed(653307226)

    trainLoader = torch.utils.data.DataLoader(trainSet,
                                            batch_size=batchSizeTrain,
                                            shuffle=True,
                                            collate_fn=planes_collate_fn,
                                            generator=gen)
    
    
    validSet = InteractionClassifierDataset(VALIDFILENAME, verbosity)
    print(f"Number of Entries in Validation Set = {len(validSet)}")

    validLoader = torch.utils.data.DataLoader(validSet,
                                            batch_size=batchSizeValid,
                                            shuffle=True,
                                            collate_fn=planes_collate_fn)


    if NENTRIES > 0:
        itersPerEpoch = NENTRIES / iterSizeTrain
        # if niters is None:
        #     # we set it by the number of request epochs
        #     niters = (epochs - start_epoch) * NENTRIES
        # else:
        #     epochs = niters / NENTRIES
    else:
        itersPerEpoch = 1
    print( "Number of Epochs: ",epochs) # TODO: delete epochs variable (unused?)
    print ("Iterations per Epoch: ", itersPerEpoch)

    if RESUME_FROM_CHECKPOINT:
        print ("RESUMING FROM CHECKPOINT FILE ",CHECKPOINT_FILE)
        checkpoint = torch.load( CHECKPOINT_FILE, map_location="cpu" )
        best_prec1 = checkpoint["best_prec1"]
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint['optimizer'])


    loss_meters, acc_meters, acc_meters_binary, time_meters = make_meters()
    
    for iiter in range(start_iter, niters):
        if iiter != 0:
            lr = adjust_learning_rate(optimizer, iiter, lr)

        print(f"Iteration {iiter} (Training)")
        _ = iteration(trainLoader, model, DEVICE, criterion, optimizer, nBatchesIterTrain, tbWriter, iiter, loss_meters, acc_meters, acc_meters_binary, time_meters, isTrain=True)

        if iiter != 0 and iiter % iter_per_valid == 0:
            print(f"Iteration {iiter} (Validation)")
            loss_meters_valid, acc_meters_valid, acc_meters_binary_valid, time_meters_valid = make_meters()
            totloss = iteration(trainLoader, model, DEVICE, criterion, optimizer, nBatchesIterValid, tbWriter, iiter, loss_meters_valid, acc_meters_valid, acc_meters_binary_valid, time_meters_valid, isTrain=False)
            # TODO: decide to save checkpoint
            prec1   = totloss
            is_best =  prec1 < best_prec1
            best_prec1 = max(prec1, best_prec1)
        
            # check point for best model
            if is_best:
                print ("Saving best model")
                save_checkpoint({
                    'iter':iiter,
                    'epoch': iiter/itersPerEpoch,
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, -1)
    
    print ("saving last state")
    save_checkpoint({
        'iter':niters,
        'epoch': niters/itersPerEpoch,
        'state_dict': model.state_dict(),
        'best_prec1': best_prec1,
        'optimizer' : optimizer.state_dict(),
    }, False, niters)

def iteration(dataLoader, model, device, criterion, optimizer, nBatches, tbWriter, iiter, loss_meters, acc_meters, acc_meters_binary, time_meters, isTrain=True):
    if isTrain:
        model.train()
        # optimizer.zero_grad()
    else:
        model.eval()

    for i in range(nBatches):
        dtBatch = time.time()
        print(f"Batch {i + 1} of {nBatches}")

        dtData = time.time()
        batch = next(iter(dataLoader))
        bcoord_u, bcoord_v, bcoord_y, bfeat_u, bfeat_v, bfeat_y, batchTruth = batch
        batchSize = len(batchTruth) // 7

        in_u = ME.TensorField(features=bfeat_u,coordinates=bcoord_u, device=device)
        in_v = ME.TensorField(features=bfeat_v,coordinates=bcoord_v, coordinate_manager=in_u.coordinate_manager, device=device)
        in_y = ME.TensorField(features=bfeat_y,coordinates=bcoord_y, coordinate_manager=in_u.coordinate_manager, device=device)
        input = [in_u, in_v, in_y]
        time_meters["data"].update(time.time() - dtData)

        # print(test_in.device)
        # print("bfeat_u.ndim:",bfeat_u.ndim)
        # print("bcoord_u.ndim:", bcoord_u.ndim)
        # test_in = ME.SparseTensor(bfeat_u, bcoord_u)
        # print("input made")
        dtForward = time.time()
        if isTrain:
            out = model(input)
        else:
            out = model.deploy(input)
        time_meters["forward"].update(time.time() - dtForward)
        print("Output Returned")
        # print(out)
        # print(out.F)

        # loss calc:
        # print(batchTruth)
        # print(batchTruth.view(7,4))
        # print(batchTruth.view(4,7))
        # temp = torch.transpose(batchTruth.view(batchSize,7), 0, 1).to(device)
        # print(temp)
        # temp2 = torch.cuda.LongTensor(temp)
        truth = torch.tensor(torch.transpose(batchTruth.view(batchSize,7), 0, 1), dtype=torch.int64, device=device)
        # fl_loss, iT_loss, nP_loss, nCP_loss, nNP_loss, nN_loss, totloss = criterion(out, truth)
        totloss = criterion(out, truth)

        dtBackward = time.time()
        if isTrain:
            optimizer.zero_grad()
            print("Backward Pass")
            totloss.backward()
            print("Step")
            optimizer.step()
        time_meters["backward"].update(time.time() - dtBackward)

        loss_meters["total"].update( totloss.item() )
        # loss_meters["fl_loss"].update( fl_loss.item() )
        # loss_meters["iT_loss"].update( iT_loss.item() )
        # loss_meters["nP_loss"].update( nP_loss.item() )
        # loss_meters["nCP_loss"].update( nCP_loss.item() )
        # loss_meters["nNP_loss"].update( nNP_loss.item() )
        # loss_meters["nN_loss"].update( nN_loss.item() )

        dtAccuracy = time.time()
        if not isTrain:
            update_acc_meters(out, truth, acc_meters, acc_meters_binary, batchSize)
        time_meters["accuracy"].update(time.time() - dtAccuracy)
        time_meters["batch"].update(time.time() - dtBatch)


        # write to tensorboard
        if isTrain:
            loss_scalars = { x:y.val for x,y in loss_meters.items() }
            tbWriter.add_scalars('data/train_loss', loss_scalars, iiter )

            time_scalars = { x:y.val for x,y in time_meters.items() }
            tbWriter.add_scalars('data/train time', time_scalars, iiter )
        else:
            loss_scalars = { x:y.val for x,y in loss_meters.items() }
            tbWriter.add_scalars('data/valid_loss', loss_scalars, iiter )

            time_scalars = { x:y.val for x,y in time_meters.items() }
            tbWriter.add_scalars('data/valid_time', time_scalars, iiter )

            acc_scalars = { x:y.val for x,y in acc_meters.items() }
            tbWriter.add_scalars('data/valid_accuracy', acc_scalars, iiter )

            acc_scalars_binary = { x:y.val for x,y in acc_meters_binary.items() }
            tbWriter.add_scalars('data/valid_accuracy_binary', acc_scalars_binary, iiter )
    return totloss


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

def make_meters():
    loss_meters = {}
    for l in ["total", "fl_loss", "iT_loss", "nP_loss", "nCP_loss", "nNP_loss", "nN_loss"]:
        loss_meters[l] = AverageMeter()

    acc_meters  = {}
    acc_meters_binary = {}
    for n in ["flavors", "Interaction Type", "Num Protons", "Num Charged Pions", "Num Neutral Pions", "Num Neutrons"]:
        acc_meters[n] = AverageMeter()
        acc_meters_binary[n] = AverageMeter()

    # timers for profiling
    time_meters = {}
    for l in ["iteration","batch","data","forward","backward","accuracy"]:
        time_meters[l] = AverageMeter()

    return loss_meters, acc_meters, acc_meters_binary, time_meters

def update_acc_meters(predict, true, acc_meters, acc_meters_binary, batchsize):
    """Computes the accuracy metrics."""
    profile = False

    # needs to be as gpu as possible!
    if profile:
        start = time.time()
    
    sums = [0,0,0,0,0,0]
    for i in range(0, true[0].shape[0]):
        acc_meters["flavors"].update(predict[0][i][true[0][i]])
        # acc_meters["Interaction Type"].update(predict[1][i][true[1][i]])
        # acc_meters["Num Protons"].update(predict[2][i][true[2][i]])
        # acc_meters["Num Charged Pions"].update(predict[3][i][true[3][i]])
        # acc_meters["Num Neutral Pions"].update(predict[4][i][true[4][i]])
        # acc_meters["Num Neutrons"].update(predict[5][i][true[5][i]])
        
        if torch.argmax(predict[0][i]).item() == true[0][i]: sums[0] += 1
        # if torch.argmax(predict[1][i]).item() == true[1][i]: sums[1] += 1
        # if torch.argmax(predict[2][i]).item() == true[2][i]: sums[2] += 1
        # if torch.argmax(predict[3][i]).item() == true[3][i]: sums[3] += 1
        # if torch.argmax(predict[4][i]).item() == true[4][i]: sums[4] += 1
        # if torch.argmax(predict[5][i]).item() == true[5][i]: sums[5] += 1

    acc_meters_binary["flavors"].update(sums[0] / batchsize)
    # acc_meters_binary["Interaction Type"].update(sums[1] / batchsize)
    # acc_meters_binary["Num Protons"].update(sums[2] / batchsize)
    # acc_meters_binary["Num Charged Pions"].update(sums[3] / batchsize)
    # acc_meters_binary["Num Neutral Pions"].update(sums[4] / batchsize)
    # acc_meters_binary["Num Neutrons"].update(sums[5] / batchsize)


    if profile:
        torch.cuda.synchronize()
        start = time.time()

def adjust_learning_rate(optimizer, iter, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #lr = lr * (0.5 ** (iter // 300))
    # lr = lr
    if iter % 300 == 0:
        lr = lr/10
        # lr = 1.0e-3
        print("Adjust learning rate to ",lr)
    #lr = lr*0.992
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    #dump_lr_schedule(1.0e-2, 4000)
    main()