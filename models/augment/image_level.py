'''
Function:
    Implementation of ImageLevelContext
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.augment.SelfAttention import SelfAttentionBlock
from models.submodule import *

'''image-level context module'''
class ImageLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=True, **kwargs):
        super(ImageLevelContext, self).__init__()
        # norm_cfg, act_cfg, self.align_corners = kwargs['norm_cfg'], kwargs['act_cfg'], kwargs['align_corners']
        self.global_avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
        )
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv3d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(feats_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )

    '''forward'''
    def forward(self, x):
        x_global = self.global_avgpool(x)
        x_global = F.interpolate(x_global, size=x.size()[2:], mode='trilinear', align_corners=False)
        feats_il = self.correlate_net(x, torch.cat([x_global, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il


class DisparityLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, concat_input=True, **kwargs):
        super(DisparityLevelContext, self).__init__()
        # norm_cfg, act_cfg, self.align_corners = kwargs['norm_cfg'], kwargs['act_cfg'], kwargs['align_corners']
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feats_channels = feats_channels
        self.correlate_net = SelfAttentionBlock(
            key_in_channels=feats_channels * 2,
            query_in_channels=feats_channels,
            transform_channels=transform_channels,
            out_channels=feats_channels,
            share_key_query=False,
            query_downsample=None,
            key_downsample=None,
            key_query_num_convs=2,
            value_out_num_convs=1,
            key_query_norm=True,
            value_out_norm=True,
            matmul_norm=True,
            with_out_project=True,
        )
        if concat_input:
            self.bottleneck = nn.Sequential(
                nn.Conv3d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm3d(feats_channels),
                nn.LeakyReLU(0.1, inplace=True),
            )

    '''forward'''
    def forward(self, x):
        # new add
        # x = x.repeat(1, self.feats_channels, 1, 1, 1)
        ####
        b, c, d, h, w = x.shape
        x_new = x.view(b, -1, h, w)
        x_global = self.global_avgpool(x_new)
        x_global = F.interpolate(x_global, size=x.size()[3:], mode='bilinear', align_corners=False)
        x_global_new = x_global.reshape(b, c, d, h, w)
        feats_il = self.correlate_net(x, torch.cat([x_global_new, x], dim=1))
        if hasattr(self, 'bottleneck'):
            feats_il = self.bottleneck(torch.cat([x, feats_il], dim=1))
        return feats_il