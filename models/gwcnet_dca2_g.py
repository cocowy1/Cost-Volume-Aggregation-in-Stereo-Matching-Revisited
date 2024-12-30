from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
from models.submodule import *
import math
from models.augment.image_level import ImageLevelContext, DisparityLevelContext
from models.augment.semantic_level import SemanticLevelContext
from models.submodule import *
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

        self.cva1 = cva(self.maxdisp,32, downsample=True)
        self.cva2 = cva(self.maxdisp,32, downsample=True)

        self.classif0 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
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

    def forward(self, left, right):
        guidance = self.guidance(left)
        features_left = self.feature_extraction(left)
        features_right = self.feature_extraction(right)

        gwc_volume = build_gwc_volume(features_left["gwc_feature"], features_right["gwc_feature"], self.maxdisp // 4,
                                      self.num_groups)
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
        out1 = augmented_cost + cost0

        prob_volume2, out2 = self.cva2(out1)

        out2 = self.classif2(out2)
        # out3 = F.upsample(out3, scale_factor=(4, 4, 4), mode='trilinear')
        cost2 = torch.squeeze(out2, 1)
        prob2 = F.softmax(cost2, dim=1)
        pred2 = disparity_regression(prob2, self.maxdisp // 4)
        pred2 = self.prop(guidance['g'], pred2)

        if self.training:
            out0 = self.classif0(cost0)
            # out0 = F.upsample(out0, scale_factor=(4, 4, 4), mode='trilinear')
            out0 = torch.squeeze(out0, 1)
            pred0 = F.softmax(out0, dim=1)
            # pred0 = disparity_regression(pred0, self.maxdisp)

            out_dca0 = F.upsample(prob_volume1, scale_factor=(2, 2, 2), mode='trilinear')
            out_dca0 = torch.squeeze(out_dca0, 1)
            pred_dca0 = F.softmax(out_dca0, dim=1)
            # pred_dca0 = disparity_regression(pred_dca0, self.maxdisp)

            out_dca1 = F.upsample(prob_volume2, scale_factor=(2, 2, 2), mode='trilinear')
            out_dca1 = torch.squeeze(out_dca1, 1)
            pred_dca1 = F.softmax(out_dca1, dim=1)
            # pred_dca1 = disparity_regression(pred_dca1, self.maxdisp)


            out1 = self.classif1(out1)
            # out1 = F.upsample(out1, scale_factor=(4, 4, 4), mode='trilinear')
            cost1 = torch.squeeze(out1, 1)
            pred1 = F.softmax(cost1, dim=1)
            # pred1 = disparity_regression(pred1, self.maxdisp)


            # cost2 = F.upsample(out2, scale_factor=(4, 4, 4), mode='trilinear')
            # cost2 = torch.squeeze(cost2, 1)
            # pred2 = F.softmax(cost2, dim=1)
            # pred2 = disparity_regression(pred2, self.maxdisp)

            # cost3 = F.upsample(out3, scale_factor=(4, 4, 4), mode='trilinear')
            # cost3 = torch.squeeze(cost3, 1)
            # pred3 = F.softmax(cost3, dim=1)
            # pred3 = disparity_regression(pred3, self.maxdisp)

            return [pred0, pred_dca0, pred_dca1, pred1, pred_dca2, pred2], pred3.squeeze(1)

        else:
            return pred2.squeeze(1), prob_volume2.squeeze(1)


def GwcNet_G(d):
    return GwcNet(d, use_concat_volume=False)


def GwcNet_GC(d):
    return GwcNet(d, use_concat_volume=True)

