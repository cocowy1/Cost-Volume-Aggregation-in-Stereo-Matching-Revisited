from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from models.gwcnet_dca_g import *
from utils import *

import dataloader.datasets as DA
from models.loss import focal_loss, model_loss
cudnn.benchmark = True

import numpy as np
np.seterr(divide='ignore', invalid='ignore')

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')
parser.add_argument('--datapath', default='/home/wy/Data/SceneFlow/', help='datapath')

parser.add_argument('--print_freq', type=int, default=200, help='the freuency of printing losses (iterations)')
parser.add_argument('--lrepochs', type=str, default="12,20,24,28:2", help='the epochs to decay lr: the downscale rate')
parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
parser.add_argument('--cuda', action='store_true', default=True, help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--epochs', type=int, default=40, help='number of epochs to train')
parser.add_argument('--savemodel', default='./trained/', help='save model')
parser.add_argument('--loadmodel', default='./trained/checkpoint_final.tar', help='load model')
parser.add_argument('--focal_coefficient', type=float, default=5.0, help='initial learning rate')
parser.add_argument('--sparse', type=bool, default=False, help='initial learning rate')

# parse arguments, set seeds
args = parser.parse_args()
# dataset, dataloader
if args.cuda:
    torch.cuda.manual_seed(args.seed)

all_left_img, all_right_img, all_left_disp, test_left_img, test_right_img, test_left_disp = DA.dataloader_SceneFlow(args.datapath)

TrainImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_SceneFlow(all_left_img, all_right_img, all_left_disp, True),
    batch_size=1, shuffle=True, num_workers=8, drop_last=False)

TestImgLoader = torch.utils.data.DataLoader(
    DA.myImageFloder_SceneFlow(test_left_img, test_right_img, test_left_disp, False),
    batch_size=1, shuffle=False, num_workers=8, drop_last=False)

# model, optimizer
model = GwcNet(args.maxdisp)
model = nn.DataParallel(model, device_ids=[0])
model.cuda()

# load parameters
if args.loadmodel is not None:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    model.load_state_dict(pretrain_dict['state_dict'], strict=True)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

class SegmentationMetric(object):
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        #  PA = acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return classAcc  # 返回的是一个列表值，如：[0.90, 0.80, 0.96]，表示类别1 2 3各类别的预测准确率

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)  # np.nanmean 求平均值，nan表示遇到Nan类型，其值取为0
        return meanAcc  # 返回单个值，如：np.nanmean([0.90, 0.80, 0.96, nan, nan]) = (0.90 + 0.80 + 0.96） / 3 =  0.89

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)  # 取对角元素的值，返回列表
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)  # axis = 1表示混淆矩阵行的值，返回列表； axis = 0表示取混淆矩阵列的值，返回列表
        IoU = intersection / union  # 返回列表，其值为各个类别的IoU
        mIoU = np.nanmean(IoU)  # 求各类别IoU的平均
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):  # 同FCN中score.py的fast_hist()函数
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

def train(imgL, imgR, disp_true):
    model.train()
    imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()
    # ---------
    disp_true = disp_true
    mask = ((disp_true < 192) & (disp_true > 0)).byte().bool()
    mask.detach_()
    # ----
    optimizer.zero_grad()
    cls_outputs, disp_outputs = model(imgL, imgR)
    loss = focal_loss(cls_outputs, disp_true, args.maxdisp, args.focal_coefficient, args.sparse) + \
            model_loss(disp_outputs, disp_true, mask)

    # disp_outputs = model(imgL, imgR)
    # loss = model_loss(disp_outputs, disp_true, mask)
    epe = torch.mean(torch.abs(disp_outputs[-1][mask] - disp_true[mask]))

    loss.backward()
    optimizer.step()
    return loss.item(), epe.item()

def mytest(imgL, imgR, disp_true):
    model.eval()
    imgL = Variable(torch.FloatTensor(imgL))
    imgR = Variable(torch.FloatTensor(imgR))

    if args.cuda:
        imgL, imgR, disp_true = imgL.cuda(), imgR.cuda(), disp_true.cuda()

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
        output3, pred_au = model(imgL, imgR)

    if top_pad != 0:
        pred_disp = output3.squeeze(1)[:, top_pad:, :]
    else:
        pred_disp = output3
    # pred_disp = pred_disp.data.cpu()

    if len(disp_true[mask]) == 0:
        loss = 0
        metrics = {
            'epe': 0,
            '1px': 0,
            '3px': 0,
        }
        mpa = {
            'mpa0': 0,
            'mpa1': 0,
            'mpa2': 0,
        }

        mIoU = {
            'mIoU0': 0,
            'mIoU1': 0,
            'mIoU2': 0,
        }
        print('it meet a 0 number')
    else:
        loss = F.smooth_l1_loss(pred_disp[mask], disp_true[mask], size_average=True)
        # epe = torch.mean(torch.abs(pred_disp[mask]-disp_true[mask]))  # end-point-error
        # epe = F.l1_loss(pred_disp[mask], disp_true[mask], size_average=True)
        epe = (pred_disp - disp_true).abs()
        epe = epe.view(-1)[mask.view(-1)]

        metrics = {
            'epe': epe.mean().item(),
            '1px': (epe > 1).float().mean().item(),
            '3px': (epe > 3).float().mean().item(),
        }

        disp_gt = disp_true.clone()
        disp_gt = F.adaptive_avg_pool2d(disp_gt / 8, (540//8, 960//8)).floor().cpu().numpy().astype('int64')
        pred_au_0 = pred_au[0].argmax(1)[:, 1:, ...].cpu().numpy()
        pred_au_1 = pred_au[1].argmax(1)[:, 1:, ...].cpu().numpy()
        pred_au_2 = pred_au[2].argmax(1)[:, 1:, ...].cpu().numpy()

        metric = SegmentationMetric(24)  # 3表示有3个分类，有几个分类就填几
        metric.addBatch(pred_au_0, disp_gt)
        pa0 = metric.pixelAccuracy()
        cpa0 = metric.classPixelAccuracy()
        mpa0 = metric.meanPixelAccuracy()
        mIoU0 = metric.meanIntersectionOverUnion()

        metric.addBatch(pred_au_1, disp_gt)
        pa1 = metric.pixelAccuracy()
        cpa1 = metric.classPixelAccuracy()
        mpa1 = metric.meanPixelAccuracy()
        mIoU1 = metric.meanIntersectionOverUnion()

        metric.addBatch(pred_au_2, disp_gt)
        pa2 = metric.pixelAccuracy()
        cpa2 = metric.classPixelAccuracy()
        mpa2 = metric.meanPixelAccuracy()
        mIoU2 = metric.meanIntersectionOverUnion()

        mpa = {
            'mpa0': mpa0,
            'mpa1': mpa1,
            'mpa2': mpa2,
        }

        mIoU = {
            'mIoU0': mIoU0,
            'mIoU1': mIoU1,
            'mIoU2': mIoU2,
        }

    return loss, metrics, mpa, mIoU

def main():
    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        print('This is %d-th epoch, focal_loss=5' % (epoch + 1))
        total_train_loss = 0.0
        total_train_epe = 0.0
        adjust_learning_rate(optimizer, epoch, args.lr, args.lrepochs)

        # # ## training ##
        for batch_idx, (imgL_crop, imgR_crop, disp_crop_L) in enumerate(TrainImgLoader):
        
            loss, epe = train(imgL_crop, imgR_crop, disp_crop_L)
            total_train_loss += loss
            total_train_epe += epe
        
            if batch_idx % args.print_freq == 0:
                print('### batch_idx %4d of total %4d, loss---%.3f, EPE---%.3f ###' %
                      (batch_idx + 1,
                       len(TrainImgLoader),
                       float(total_train_loss / (batch_idx + 1)),
                       float(total_train_epe / (batch_idx + 1))))
        
        print('epoch %d total train loss = %.3f, total train epe = %.3f' % (epoch + 1,
                                                                            total_train_loss / len(TrainImgLoader),
                                                                            total_train_epe / len(TrainImgLoader)))
        
        # SAVE
        if (epoch + 1) > 18:
            savefilename = args.savemodel + '/checkpoint_' + str(epoch) + '.tar'
            torch.save({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'train_loss': total_train_loss / len(TrainImgLoader),
            }, savefilename)

        # ------------- TEST ------------------------------------------------------------
        if (epoch + 1) > -1:
            total_test_loss = 0.0
            total_test_epe = 0.0
            total_test_1px = 0.0
            total_test_3px = 0.0

            total_test_mpa0 = 0.0
            total_text_miou0 = 0.0
            total_test_mpa1 = 0.0
            total_text_miou1 = 0.0
            total_test_mpa2 = 0.0
            total_text_miou2 = 0.0


            for batch_idx, (imgL, imgR, disp_L) in enumerate(TestImgLoader):
                test_loss, metrics, mpa, miou = mytest(imgL, imgR, disp_L)
                total_test_loss += test_loss
                total_test_epe += metrics['epe']
                total_test_1px += metrics['1px']
                total_test_3px += metrics['3px']
                total_test_mpa0 += mpa['mpa0']
                total_text_miou0 += miou['mIoU0']
                total_test_mpa1 += mpa['mpa1']
                total_text_miou1 += miou['mIoU1']
                total_test_mpa2 += mpa['mpa2']
                total_text_miou2 += miou['mIoU2']

                if batch_idx % args.print_freq == 0:
                    print(
                        '### batch_idx %5d of total %5d, test loss=%.3f, test_epe=%.3f, test_1px=%.3f, test_3px=%.3f,'
                        'test_mpa0=%.3f, test_miou0=%.3f, test_mpa1=%.3f, test_miou1=%.3f, '
                        'test_mpa2=%.3f, test_miou2=%.3f' % (
                            batch_idx + 1,
                             len(TestImgLoader),
                            total_test_loss / (batch_idx + 1), total_test_epe / (batch_idx + 1),
                            (total_test_1px * 100) / (batch_idx + 1), (total_test_3px * 100) / (batch_idx + 1),
                            (total_test_mpa0 * 100) / (batch_idx + 1), (total_text_miou0 * 100) / (batch_idx + 1),
                            (total_test_mpa1 * 100) / (batch_idx + 1), (total_text_miou1 * 100) / (batch_idx + 1),
                            (total_test_mpa2 * 100) / (batch_idx + 1), (total_text_miou2 * 100) / (batch_idx + 1),
                        ))

            print(
                'epoch %d, total test loss=%.3f, total_test_epe=%.3f, total_test_1px=%.3f, total_test_3px=%.3f,'
                ' total_test_mpa0=%.3f, total_test_miou0=%.3f, total_test_mpa1=%.3f, total_test_miou1=%.3f, '
                'total_test_mpa2=%.3f, total_test_miou2=%.3f' % (
                    epoch + 1,
                    total_test_loss / len(TestImgLoader), total_test_epe / len(TestImgLoader),
                    (total_test_1px * 100) / len(TestImgLoader), (total_test_3px * 100) / len(TestImgLoader),
                    (total_test_mpa0 * 100) / len(TestImgLoader), (total_text_miou0 * 100) / len(TestImgLoader),
                    (total_test_mpa1 * 100) / len(TestImgLoader), (total_text_miou1 * 100) / len(TestImgLoader),
                    (total_test_mpa2 * 100) / len(TestImgLoader), (total_text_miou2 * 100) / len(TestImgLoader)
                ))

if __name__ == '__main__':
    main()
