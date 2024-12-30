from __future__ import print_function

import time

import torch
import torch.nn as nn
import torch.utils.data
import math
import matplotlib.pyplot as plt
from models.augment.cva import cva
from models.submodule import *

class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.ReLU(inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 16, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1, 1, 2)

        if self.concat_feature:
            self.lastconv = nn.Sequential(convbn(320, 128, 3, 1, 1, 1),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, concat_feature_channel, kernel_size=1, padding=0, stride=1,
                                                    bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion), )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.layer1(x)
        l2 = self.layer2(x)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        gwc_feature = torch.cat((l2, l3, l4), dim=1)

        if not self.concat_feature:
            return {"gwc_feature": gwc_feature}
        else:
            concat_feature = self.lastconv(gwc_feature)
            return {"gwc_feature": gwc_feature, "concat_feature": concat_feature}


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv3 = nn.Sequential(convbn_3d(in_channels * 2, in_channels * 4, 3, 2, 1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(in_channels * 4, in_channels * 4, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 4, in_channels * 2, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels * 2))

        self.conv6 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir1 = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)
        self.redir2 = convbn_3d(in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6

class PropgationNet_4x(nn.Module):
    def __init__(self, base_channels):
        super(PropgationNet_4x, self).__init__()
        self.base_channels = base_channels
        self.conv = nn.Sequential(convbn(base_channels, base_channels * 2, 3, 1, 1, 1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(base_channels * 2, 9 * 16, kernel_size=(3, 3), stride=(1, 1), padding=1,
                                            dilation=(1, 1), bias=False))

    def forward(self, guidance, disp):
        b, c, h, w = disp.shape
        disp = F.unfold(4 * disp, [3, 3], padding=1).view(b, 1, 9, 1, 1, h, w)
        mask = self.conv(guidance).view(b, 1, 9, 4, 4, h, w)
        mask = F.softmax(mask, dim=2)
        up_disp = torch.sum(mask * disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(b, 1, 4 * h, 4 * w)

class GwcNet(nn.Module):
    def __init__(self, maxdisp, use_concat_volume=True):
        super(GwcNet, self).__init__()
        self.maxdisp = maxdisp
        self.use_concat_volume = use_concat_volume

        self.num_groups = 40
        if self.use_concat_volume:
            self.concat_channels = 12
            self.feature_extraction = feature_extraction(concat_feature=True,
                                                         concat_feature_channel=self.concat_channels)
        else:
            self.concat_channels = 0
            self.feature_extraction = feature_extraction(concat_feature=False)

        self.dres0 = nn.Sequential(convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1))

        self.cva1 = cva(self.maxdisp, 32, downsample=True)
        self.cva2 = cva(self.maxdisp, 32, downsample=True)
        self.cva3 = cva(self.maxdisp, 32, downsample=True)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
        #
        self.guidance = Guidance(64)
        self.prop = PropgationNet_4x(64)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            # elif isinstance(m, nn.Linear):
            #     m.bias.data.zero_()

    def vis_weight(self, prob_volume):
        prob_volume = F.interpolate(prob_volume.unsqueeze(1), scale_factor=(8, 8, 8), mode='trilinear').squeeze(1)
        prob_volume = F.softmax(prob_volume, dim=1)
        prob_argmax = disparity_regression(prob_volume, 192).squeeze(1)
        mask = ((prob_argmax > 39) & (prob_argmax < 50))
        mask[:140, 470:] = 0
        mask[380:, :] = 0
        mask[100:400, 600:800] = 0
        prob_volume_t = prob_volume[:, 39:50].sum(1)
        weight = prob_volume_t.squeeze() * mask
        weight = weight[:, 12:,...]
        weight[:, 100:400, 600:800] = 0
        _, h, w = weight.shape

        weight = F.softmax(weight.reshape(-1), dim=0)
        weight = weight.reshape(h, w)
        plt.figure(); plt.imshow(weight.squeeze().cpu().numpy())
        import matplotlib.image as image
        image.imsave('./rebuttal/weight.svg', weight.squeeze().cpu().numpy())

    def forward(self, left, right, disp_true):
        # left, right = inputs[0,0,...].unsqueeze(0), inputs[0,1,...].unsqueeze(0)
        start_time = time.time()

        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)
        guidance = self.guidance(left)
        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4, self.num_groups)
        if self.use_concat_volume:
            concat_volume = build_concat_volume(features_left["concat_feature"], features_right["concat_feature"],
                                                self.maxdisp // 4)
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        #### augment cv ####
        prob_volume1, augmented_cost = self.cva1(cost0)
        out1 = cost0 + augmented_cost

        prob_volume2, out2 = self.cva2(out1)
        prob_volume3, out3 = self.cva3(out2)

        # convex upsample
        out3 = self.classif3(out3)
        # cost2 = F.upsample(out2, scale_factor=(4, 4, 4), mode='trilinear')
        cost3 = torch.squeeze(out3, 1)
        cost3 = F.softmax(cost3, dim=1)
        pred4 = disparity_regression(cost3, self.maxdisp//4)
        pred4 = self.prop(guidance['g'], pred4)

        # self.vis_weight(prob_volume3.squeeze(1))

        if self.training:
            out0 = self.classif0(cost0)
            # out0 = F.upsample(out0, scale_factor=(4, 4, 4), mode='trilinear')
            out0 = torch.squeeze(out0, 1)
            pred0 = F.softmax(out0, dim=1)
            # pred0 = disparity_regression(pred0, self.maxdisp)

            out_dca1 = F.upsample(prob_volume1, scale_factor=(2, 2, 2), mode='trilinear')
            out_dca1 = torch.squeeze(out_dca1, 1)
            pred_dca1 = F.softmax(out_dca1, dim=1)
            # pred_dca0 = disparity_regression(pred_dca0, self.maxdisp)

            out_dca2 = F.upsample(prob_volume2, scale_factor=(2, 2, 2), mode='trilinear')
            out_dca2 = torch.squeeze(out_dca2, 1)
            pred_dca2 = F.softmax(out_dca2, dim=1)
            # pred_dca1 = disparity_regression(pred_dca1, self.maxdisp)

            out_dca3 = F.upsample(prob_volume3, scale_factor=(8, 8, 8), mode='trilinear')
            out_dca3 = torch.squeeze(out_dca3, 1)
            pred_dca3 = F.softmax(out_dca3, dim=1)
            pred_dca3 = disparity_regression(pred_dca3, self.maxdisp)

            out1 = self.classif1(out1)
            # out1 = F.upsample(out1, scale_factor=(4, 4, 4), mode='trilinear')
            cost1 = torch.squeeze(out1, 1)
            pred1 = F.softmax(cost1, dim=1)
            # pred1 = disparity_regression(pred1, self.maxdisp)

            out2 = self.classif2(out2)
            # out2 = F.upsample(out2, scale_factor=(4, 4, 4), mode='trilinear')
            cost2 = torch.squeeze(out2, 1)
            pred2 = F.softmax(cost2, dim=1)

            return [pred0, pred_dca1, pred_dca2,
                    pred1, pred2], [pred_dca3, pred4]
            # return [pred0, pred_dca0, pred_dca1, pred_dca2, pred1, pred2, pred3]

        else:
            return pred4, prob_volume2.squeeze(1)

