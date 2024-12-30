'''
Function:
    Implementation of SemanticLevelContext
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.augment.SelfAttention import SelfAttentionBlock
from models.submodule import *


class SElayer(nn.Module):

    def __init__(self, num_channels, reduction_ratio=2):
        super(SElayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        num_channels_reduced = num_channels // reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=False)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.shape
        # avgpool 3d
        x = x.permute(0,2,1,3,4)
        squeeze_x = self.avg_pool(x)
        # import pdb
        # pdb.set_trace()
        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_x.view(b, d)))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        out_x = torch.mul(x, fc_out_2.view(b, d, 1, 1, 1)) # b,d,c,h,w
        out_x = out_x.permute(0, 2, 1, 3, 4)

        return out_x

class SEBlock(nn.Module):

    def __init__(self, inplanes, planes, kernel_size=3, stride=1, pad=1, reduciton=2):
        super(SEBlock, self).__init__()
        self.conv1 = convbn_3d(inplanes, planes, kernel_size, stride, pad)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = convbn_3d(planes, planes, kernel_size, stride, pad)
        self.relu2 = nn.ReLU(inplace=True)
        self.se = SElayer(24, reduciton)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        out = self.se(out)

        out = out + residual
        out = self.relu(out)
        return out

'''semantic-level context module'''
class SemanticLevelContext(nn.Module):
    def __init__(self, feats_channels, transform_channels, reduction=8, concat_input=True, **kwargs):
        super(SemanticLevelContext, self).__init__()
        # self.se = SEBlock(inplanes=feats_channels, planes=feats_channels, reduciton=reduction)
        # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        self.cross_attention = SelfAttentionBlock(
            key_in_channels=feats_channels,
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

        self.kernel = 3
        self.agg = nn.Sequential(
                    nn.Conv3d(feats_channels, feats_channels, kernel_size=self.kernel, stride=1, padding=(self.kernel-1)//2, bias=False),
                    nn.BatchNorm3d(feats_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv3d(feats_channels, feats_channels, kernel_size=self.kernel, stride=1, padding=(self.kernel-1)//2,
                    bias=False),
                    nn.BatchNorm3d(feats_channels),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv3d(feats_channels, feats_channels, kernel_size=self.kernel, stride=1, padding=(self.kernel-1)//2,
                    bias=False)
                )
        # if concat_input:
        #     self.bottleneck = nn.Sequential(
        #         nn.Conv3d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
        #         BatchNorm3d(feats_channels, momentum=BN_MOMENTUM),
        #         nn.LeakyReLU(0.1, inplace=True),
        #     )

    '''forward'''
    def forward(self, x, preds):
        inputs = x
        preds = F.softmax(preds, dim=1)

        masks = preds.max(1,True)[0]
        masks = (preds==masks).unsqueeze(1)

        feats_sl = masks*x
        feats_sl = self.agg(feats_sl)
        feats_sl = masks * feats_sl

        feats_sl = self.cross_attention(inputs, feats_sl)
        return feats_sl


    #     self.kernel = 3
    #     # self.agg = nn.Sequential(
    #     #             nn.Conv3d(feats_channels, feats_channels, kernel_size=5, stride=1, padding=2, bias=False),
    #     #             nn.BatchNorm3d(feats_channels),
    #     #             nn.LeakyReLU(0.1, inplace=True),
    #     #         )
    #     # if concat_input:
    #     #     self.bottleneck = nn.Sequential(
    #     #         nn.Conv3d(feats_channels * 2, feats_channels, kernel_size=3, stride=1, padding=1, bias=False),
    #     #         BatchNorm3d(feats_channels, momentum=BN_MOMENTUM),
    #     #         nn.LeakyReLU(0.1, inplace=True),
    #     #     )
    #
    # '''forward'''
    # def forward(self, x, preds):
    #     inputs = x
    #     preds = F.softmax(preds, dim=1)
    #     batch_size, num_channels, d, h, w = x.size()
    #     disparity_planes = preds.size(1)
    #
    #     feats_sl = torch.zeros(batch_size, h * w, num_channels * d).type_as(x)
    #     for batch_idx in range(batch_size):
    #         #   feats_iter: (C, D, H, W)  --> (D*H*W, C)
    #         #   preds_iter: (D, H, W) --> (H*W, D)
    #         feats_iter, preds_iter = x[batch_idx], preds[batch_idx]
    #         # feats_iter, preds_iter = feats_iter.reshape(num_channels*d, -1), preds_iter.reshape(disparity_planes, -1)
    #         # feats_iter, preds_iter = feats_iter.permute(1, 0), preds_iter.permute(1, 0)
    #         feats_iter, preds_iter = feats_iter.reshape(num_channels, d, h, w), preds_iter.reshape(disparity_planes, h, w)
    #
    #         # (H*W, )
    #         # hard fusion
    #         argmax = preds_iter.argmax(0)
    #         for disp_id in range(disparity_planes):
    #             mask_pred = (argmax == disp_id).float()
    #             if mask_pred.sum() == 0: continue
    #
    #             mask = mask_pred.masked_fill((~(mask_pred.byte().bool())), float('-inf'))
    #             mask = mask.unsqueeze(0).float()
    #
    #             feats_iter_cls = (feats_iter[...]) * mask_pred.unsqueeze(0)  # [c,d,h,w]
    #             feats_iter_cls = feats_iter_cls.reshape(-1, h, w)
    #             feats_iter_cls = feats_iter_cls.unsqueeze(0) # [b,c*d,h,w]
    #             feats_iter_cls_slice = F.unfold(feats_iter_cls, kernel_size=self.kernel, stride=1,
    #                                             padding=(self.kernel-1)//2).squeeze(0).reshape(num_channels*d, -1, h*w) # [c*d,9,h*w]
    #
    #             feats_iter_cls_slice = feats_iter_cls_slice.permute(0, 2, 1).contiguous() # [c*d,h*w,9]
    #
    #             preds_iter_cls = (preds_iter[disp_id, :, :]).unsqueeze(0)*mask  # [h*w]
    #             preds_iter_cls_slice = F.unfold(preds_iter_cls.unsqueeze(0), kernel_size=self.kernel, stride=1,
    #                                             padding=(self.kernel-1)//2).squeeze(0).reshape(-1, h*w) # [9, h*w]
    #             weight = F.softmax(preds_iter_cls_slice, dim=0)
    #             weight = weight.permute(1, 0).contiguous() # [h*w, 9]
    #             weight = torch.where(torch.isnan(weight), torch.full_like(weight, 0), weight)
    #             feats_iter_cls_slice = (feats_iter_cls_slice) * weight.unsqueeze(0) # [c*d, h*w, 9]
    #
    #             feats_iter_cls_new = feats_iter_cls_slice.sum(2)
    #             feats_iter_cls_new = feats_iter_cls_new.permute(1, 0).contiguous()[mask_pred.reshape(-1).byte().bool()] #[h*w, c*d]
    #             feats_sl[batch_idx][mask_pred.reshape(-1).byte().bool()] = feats_iter_cls_new
    #
    #
    #     feats_sl = feats_sl.reshape(batch_size, h, w, num_channels, d)
    #     feats_sl = feats_sl.permute(0, 3, 4, 1, 2).contiguous()
    #
    #     # feats_sl = self.se(feats_sl)
    #     feats_sl = self.cross_attention(inputs, feats_sl)
    #
    #     return feats_sl