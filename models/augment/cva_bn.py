'''
Function:
    Implementation of cost volume aggregation
Author:
        Yun Wang
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.augment.semantic_level import SemanticLevelContext
from models.submodule_bn import *

class Multi_Aggregation(nn.Module):
    def __init__(self, in_channels):
        super(Multi_Aggregation, self).__init__()
        self.conv1 = nn.Sequential(convbn_3d(in_channels, in_channels * 2, 3, 2, 1),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(convbn_3d(in_channels*2, in_channels * 2, 3, 1, 1),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.ConvTranspose3d(in_channels * 2, in_channels, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(in_channels))

        self.redir = convbn_3d(in_channels, in_channels, kernel_size=1, stride=1, pad=0)

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = F.relu(self.conv3(conv2) + self.redir(x), inplace=True)

        return conv3

class cva(nn.Module):
    def __init__(self, max_disp, in_channel, downsample=True):
        super(cva, self).__init__()
        self.max_disp = max_disp
        self.channel = in_channel
        if downsample:
            self.downsample = nn.Sequential(nn.AvgPool3d((3, 3, 3), stride=2, padding=1),
                                            convbn_3d(32, 32, 3, 1, 1),
                                            nn.ReLU(inplace=True))

        # build semantic-level context module
        slc_cfg = {
            'feats_channels': self.channel,
            'transform_channels': self.channel,
            'concat_input': True,
        }
        self.slc_net = SemanticLevelContext(**slc_cfg)
        # build image-level context module
        self.classify = nn.Sequential(convbn_3d(self.channel, self.channel, 3, 1, 1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv3d(self.channel, 1, kernel_size=3, padding=1, stride=1, bias=False))

        self.fuse = nn.Sequential(convbn_3d(64, 32, 1, 1, 0),)
        self.cost_agg = Multi_Aggregation(self.channel)


    def forward(self, cost_volume, downsample=True):
        if downsample:
            cost_down = self.downsample(cost_volume)
            prob_volume = self.classify(cost_down).squeeze(1)
            augmented_cost_down = self.slc_net(cost_down, prob_volume)
            augmented_cost = F.interpolate(augmented_cost_down, scale_factor=(2,2,2), mode='trilinear')
        else:
            prob_volume = self.classify(cost_volume).squeeze(1)
            augmented_cost = self.slc_net(cost_volume, prob_volume)

        augmented_cost = self.fuse(torch.cat([augmented_cost, cost_volume], dim=1))
        augmented_cost = self.cost_agg(augmented_cost)

        return prob_volume.unsqueeze(1), augmented_cost