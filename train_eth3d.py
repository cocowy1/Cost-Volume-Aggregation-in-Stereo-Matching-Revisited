from __future__ import print_function
import sys
sys.path.append("dataloader")
from torch.autograd import Variable
from models.gwcnet_dca_g import *
import argparse
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import *
import time
import dataloader.datasets as DA

import os
from models.loss import model_loss
import torch.backends.cudnn as cudnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='GwcNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath_kitti2015', default='/home/wy/Data/kitti_stereo_2015/training/',
                    help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_kitti2015_test', default='/home/wy/Data/kitti_stereo_2015/testing/',
                     help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_kitti', default='/home/wy/Data/kitti_stereo_2012/training/',
                     help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_kitti_test', default='/home/wy/Data/kitti_stereo_2012/testing/',
                    help='datapath for sceneflow monkaa dataset')

parser.add_argument('--datapath', default='/home/wy/Data',
                    help='datapath for sceneflow monkaa dataset')

parser.add_argument('--epochs', type=int, default=800,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='./fined/h3/kitti15/checkpoint_699.tar',
                    help='load model')
parser.add_argument('--gpus', type=int, nargs='+', default=[0])
parser.add_argument('--savemodel', default='/home/wy/DCANet/fined/KITTI15',
                    help='save model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--print_freq', type=int, default=1, help='the frequency of printing losses (iterations)')
parser.add_argument('--lrepochs', type=str, default="200:2", help='the epochs to decay lr: the downscale rate')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--focal_coefficient', type=float, default=5.0,  help='initial learning rate')
parser.add_argument('--sparse', type=bool, default=False, help='initial learning rate')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp = DA.dataloader('%s/eth3d'%args.datapath)
eth3dloadertrain = DA.myImageFloder_eth3d(all_left_img, all_right_img, all_left_disp, True)
test_left_img, test_right_img, test_left_disp = DA.dataloader('%s/eth3d'%args.datapath)
eth3dloadertest = DA.myImageFloder_eth3d(all_left_img, all_right_img, all_left_disp, False)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_eth3d(all_left_img, all_right_img, all_left_disp, True),
    batch_size=1, shuffle=True, num_workers=4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_eth3d(all_left_img, all_right_img, all_left_disp, False),
    batch_size=1, shuffle=False, num_workers=4, drop_last=False)

model = GwcNet(args.maxdisp)
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'])

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

def train(imgL, imgR, disp_true):
    model.train()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    mask = ((disp_true < 192) & (disp_true > 0)).byte().bool()
    mask.detach_()
    # ----
    optimizer.zero_grad()
    disp_ests = model(imgL, imgR)

    loss = model_loss(disp_ests, disp_true, mask)
    epe = torch.mean(torch.abs(disp_ests[-1][mask] - disp_true[mask]))

    loss.backward()
    optimizer.step()

    return loss.item(), epe.item()

def mytest(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))
    if args.cuda:
        imgL, imgR = imgL.cuda(), imgR.cuda()

    mask = (disp_true > 0) & (disp_true < args.maxdisp)

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        top_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        top_pad = 0

    if imgL.shape[3] % 16 != 0:
        times = imgL.shape[3] // 16
        right_pad = (times + 1) * 16 - imgL.shape[3]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0))
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0))

    with torch.no_grad():
        output3, _ = model(imgL, imgR)
    if top_pad != 0:
        pred_disp = output3[:, top_pad:, :]
    else:
        pred_disp = output3
    pred_disp = pred_disp.data.cpu()

    if len(disp_true[mask]) == 0:
        loss = 0
        epe = 0
    else:
        loss = F.smooth_l1_loss(pred_disp[mask], disp_true[mask], size_average=True)
        # epe = torch.mean(torch.abs(pred_disp[mask]-disp_true[mask]))  # end-point-error
        epe = F.l1_loss(pred_disp[mask], disp_true[mask], size_average=True)
    return loss, epe

print("Traindataset is %d"%len(TrainImgLoader))
print("Testdataset is %d"%len(TestImgLoader))

def main():
    for epoch in range(1, args.epochs):
        print('This is %d-th epoch' % (epoch))
        total_train_loss = 0.0
        total_train_epe = 0.0
        learning_rate_adjust(optimizer, epoch)

        # ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        
            loss, epe = train(imgL_crop, imgR_crop, disp_crop_L)
            total_train_loss += loss
            total_train_epe += epe
        
            if batch_idx % args.print_freq == 0:
               print('### batch_idx %5d of total %5d, loss---%.3f, EPE---%.3f ###' %
                     (batch_idx + 1,
                      len(TrainImgLoader),
                      float(total_train_loss / (batch_idx + 1)),
                      float(total_train_epe / (batch_idx + 1))))
        
        if (epoch + 1) % 1 == 0:
            avg_epe = total_train_epe / len(TrainImgLoader)
            avg_loss = total_train_loss / len(TrainImgLoader)
            print('Train Epoch----%5d of %d, train_loss---%.3f, train_EPE---%.3f' %
                  (epoch + 1, len(TrainImgLoader), avg_loss, avg_epe))
        
        ## SAVE
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        if epoch > 449:
            torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

        # ------------- TEST ------------------------------------------------------------
        if epoch > -1:
            total_test_loss = 0.0
            total_test_epe = 0.0
            start_time = time.time()
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, test_epe = mytest(imgL, imgR, disp_L)
                total_test_loss += test_loss
                total_test_epe += test_epe

            # print("this epoch total time is %.3f"%(time.time()-start_time))
            if (epoch + 1) % 1 == 0:
                avg_epe = total_test_epe / len(TestImgLoader)
                avg_loss = total_test_loss / len(TestImgLoader)
                print('Test epoch--%5d pf %d, loss-%.3f, EPE-%.3f' %
                    (epoch + 1, len(TestImgLoader), avg_loss, avg_epe))


if __name__ == '__main__':
    main()

