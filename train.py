import argparse
import pdb
import numpy.random
import functions
import LSANet as model
import torch
import torch.nn as nn
import time
import os
import copy
import random
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import math
from time import *
import pylab as pl
import skimage.io
import numpy
from sklearn.preprocessing import MinMaxScaler
import scipy.io as sio
from tqdm import tqdm
import dataloader
from torch.utils.data import DataLoader
from torch.autograd import Variable
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', help='input image dir',default='dataset/train/')
    parser.add_argument('--test_dir', help='test image dir', default='dataset/test/')
    parser.add_argument('--val_dir', help='val image dir', default='dataset/val/')
    parser.add_argument('--outputs_dir',help='output model dir',default='output/model')
    parser.add_argument('--batchSize', default=4)
    parser.add_argument('--testBatchSize', default=1)
    parser.add_argument('--epoch', default=2000)
    parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
    parser.add_argument('--not_cuda', action='store_true', help='disables cuda', default=0)
    parser.add_argument('--device',default=torch.device('cuda:0'))
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--lr',type=float,default=0.0001,help='G‘s learning rate')
    parser.add_argument('--gamma',type=float,default=0.01,help='scheduler gamma')
    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = False

    train_set = dataloader.get_training_set(opt.input_dir)
    val_set = dataloader.get_val_set(opt.val_dir)

    train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        raise ('CUDA is not available')
    net = model.LSAnet().to(opt.device)
    for module in net.modules():
     if isinstance(module, nn.BatchNorm2d):
      module.eval()
    optimizer = torch.optim.AdamW(net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[1600],gamma=opt.gamma)
    loss = torch.nn.MSELoss()
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        net = net.cuda()
        loss = loss.cuda()
    best_epoch = 0
    best_SAM=1.0
    for i in range(opt.epoch):
        net.train()
        for batch_idx, (HRMSBatch, HRHSBatch) in enumerate(train_loader):
            if torch.cuda.is_available():
                HRMSBatch, HRHSBatch = HRMSBatch.cuda(), HRHSBatch.cuda()
                HRMSBatch = Variable(HRMSBatch.to(torch.float32))
                HRHSBatch = Variable(HRHSBatch.to(torch.float32))
            N = len(train_loader)
            net.zero_grad()
            out = net(HRMSBatch)
            outLoss = loss(out, HRHSBatch)
            outLoss.backward(retain_graph=True)
            optimizer.step()
            training_state = '  '.join(['Epoch: {}', '[{} / {}]', 'outLoss: {:.6f}'])
            training_state = training_state.format(i, batch_idx, N, outLoss)
            print(training_state)
        print('successful train.')
        torch.save(net.state_dict(), os.path.join(opt.outputs_dir, 'epoch_{}.pth'.format(i)))
        net.eval()
        epoch_SAM=functions.AverageMeter()
        with torch.no_grad():
            for j, (msTest, gtTest) in enumerate(val_loader):
                if torch.cuda.is_available():
                    msTest,gtTest = msTest.cuda(),gtTest.cuda()
                    msTest = Variable(msTest.to(torch.float32))
                    gtTest = Variable(gtTest.to(torch.float32))
                    net = net.cuda()
                mp = net(msTest)
                test_SAM=functions.SAM(mp, gtTest)

                epoch_SAM.update(test_SAM,msTest.shape[0])
            print('eval SAM: {:.6f}'.format(epoch_SAM.avg))

        if epoch_SAM.avg < best_SAM:
            best_epoch = i
            best_SAM = epoch_SAM.avg
            best_weights = copy.deepcopy(net.state_dict())
        print('best epoch:{:.0f}'.format(best_epoch))
        scheduler.step()

    print('best epoch: {}, epoch_SAM: {:.6f}'.format(best_epoch, best_SAM))
    torch.save(best_weights, os.path.join(opt.outputs_dir, 'best.pth'))
