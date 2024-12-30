from __future__ import print_function
import sys
sys.path.append("dataloader")
from torch.autograd import Variable
from models.gwcnet_dca_g import *
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from util import *
import time
import dataloader.datasets as DA
import os
from models.loss import StereoFocalLoss, model_loss
import torch.backends.cudnn as cudnn

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description='GwcNet')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--datapath_kitti2015', default='/data/ywang/dataset/kitti_2015/training/',
                    help='datapath for sceneflow monkaa dataset')
parser.add_argument('--datapath_kitti', default='/data/ywang/dataset/kitti_2012/training/',
                     help='datapath for sceneflow monkaa dataset')

parser.add_argument('--epochs', type=int, default=1000,
                    help='number of epochs to train')
parser.add_argument('--loadmodel', default='/home/ywang/my_projects/my/GwcNet-augment/fined/final_kitti.tar',
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

all_left_img_0, all_right_img_0, all_left_disp_0,  \
test_left_img_0, test_right_img_0, test_left_disp_0 = DA.dataloader_KITTI2015(args.datapath_kitti2015)
#
all_left_img_1, all_right_img_1, all_left_disp_1,  \
test_left_img_1, test_right_img_1, test_left_disp_1 = DA.dataloader_KITTI(args.datapath_kitti)

all_left_img = all_left_img_0 + all_left_img_1 
all_right_img = all_right_img_0 + all_right_img_1
all_left_disp = all_left_disp_0 + all_left_disp_1 

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_KITTI(all_left_img, all_right_img, all_left_disp, True),
    batch_size=12, shuffle=True, num_workers=4, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_KITTI(test_left_img_1, test_right_img_1, test_left_disp_1, False),
    batch_size=1, shuffle=False, num_workers=4, drop_last=False)

model = GwcNet(args.maxdisp)
if args.cuda:
    model = nn.DataParallel(model)
    model.cuda()

if args.loadmodel is not None:
    print('Load pretrained model')
    state_dict = torch.load(args.loadmodel)
    from collections import OrderedDict
    model_state_dict = OrderedDict()

    for k, v in state_dict['state_dict'].items():
        k = k.replace('module.', '')
        model_state_dict[k] = v
    model.load_state_dict(state_dict['state_dict'])

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

    focal_loss_evaluator = \
        StereoFocalLoss(max_disp=args.maxdisp, focal_coefficient=args.focal_coefficient, sparse=args.sparse)

    cost_au_0, cost_au_1, disp_ests = model(imgL, imgR)
    #
    # output1 = torch.squeeze(output1, 1)
    # output2 = torch.squeeze(output2, 1)
    # output3 = torch.squeeze(output3, 1)

    CE_loss = 5 * focal_loss_evaluator(cost_au_0, disp_true, variance=1) + 10 * focal_loss_evaluator(cost_au_1, disp_true, variance=1)

    loss_disp = model_loss(disp_ests, disp_true, mask)
    loss = loss_disp + CE_loss
    epe = torch.mean(torch.abs(disp_ests[-1][mask] - disp_true[mask]))
    # print('ce_loss:%.3f, output3:%.3f'%(tensor2float(5*CE_loss), tensor2float(loss_disp)))

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
        output3 = model(imgL, imgR)
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

        ### SAVE
        savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
        if epoch > 449:
            torch.save({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'train_loss': total_train_loss / len(TrainImgLoader),
        }, savefilename)

        # ------------- TEST ------------------------------------------------------------
        if epoch > 0:
            total_test_loss = 0.0
            total_test_epe = 0.0
            start_time = time.time()
            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, test_epe = mytest(imgL, imgR, disp_L)
                total_test_loss += test_loss
                total_test_epe += test_epe
                if batch_idx % args.print_freq == 0:
                    print('### batch_idx %5d of total %5d, loss---%.3f, EPE---%.3f ###' %
                          (batch_idx + 1,
                           len(TestImgLoader),
                           float(total_test_loss / (batch_idx + 1)),
                           float(total_test_epe / (batch_idx + 1))))

            # print("this epoch total time is %.3f"%(time.time()-start_time))
            if (epoch + 1) % 1 == 0:
                avg_epe = total_test_epe / len(TestImgLoader)
                avg_loss = total_test_loss / len(TestImgLoader)
                print('Test epoch----%5d pf %d, loss---%.3f, EPE---%.3f' %
                    (epoch + 1, len(TestImgLoader), avg_loss, avg_epe))


if __name__ == '__main__':
    main()

