'''
Function:
    Implementation of SelfAttentionBlock
Author:
    Zhenchao Jin
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

'''self attention block'''
class SelfAttentionBlock(nn.Module):
    def __init__(self, key_in_channels, query_in_channels, transform_channels, out_channels, share_key_query,
                 query_downsample, key_downsample, key_query_num_convs, value_out_num_convs, key_query_norm,
                 value_out_norm, matmul_norm, with_out_project, **kwargs):
        super(SelfAttentionBlock, self).__init__()
        # norm_cfg, act_cfg = kwargs['norm_cfg'], kwargs['act_cfg']
        # key project
        self.key_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels,
            num_convs=key_query_num_convs,
            use_norm=key_query_norm,
        )
        # query project
        if share_key_query:
            assert key_in_channels == query_in_channels
            self.query_project = self.key_project
        else:
            self.query_project = self.buildproject(
                in_channels=query_in_channels,
                out_channels=transform_channels,
                num_convs=key_query_num_convs,
                use_norm=key_query_norm,
            )
        # value project
        self.value_project = self.buildproject(
            in_channels=key_in_channels,
            out_channels=transform_channels if with_out_project else out_channels,
            num_convs=value_out_num_convs,
            use_norm=value_out_norm,
        )
        # out project
        self.out_project = None
        if with_out_project:
            self.out_project = self.buildproject(
                in_channels=transform_channels,
                out_channels=out_channels,
                num_convs=value_out_num_convs,
                use_norm=value_out_norm,
            )

        # downsample
        self.query_downsample = query_downsample
        self.key_downsample = key_downsample
        self.matmul_norm = matmul_norm
        self.transform_channels = transform_channels

    '''forward'''
    def forward(self, query_feats, key_feats):
        # query_feats: [batch, channels, disparity, height ,width]
        head_dim = 8
        batch_size, channels, disparity, height, width = query_feats.shape
        dhw = disparity*height*width
        query = self.query_project(query_feats)
        if self.query_downsample is not None: query = self.query_downsample(query)
        #query: b, h, hc, d*h*w
        query = query.reshape(batch_size, channels//head_dim, head_dim, dhw)
        # query = query.permute(0, 3, 2, 1).contiguous()  # batch, h*w, disparity, channels
        query = query.permute(0, 1, 3, 2).contiguous()  # batch, h*w, head, disparity, head_c --> b, head, d*h*w, head_c

        key = self.key_project(key_feats)
        value = self.value_project(key_feats)
        if self.key_downsample is not None:
            key = self.key_downsample(key)
            value = self.key_downsample(value)

        key = key.reshape(batch_size, channels//head_dim, head_dim, dhw)  # batch, h*w, head, head_c disparity --> b, head, head_c, d*h*w
        value = value.reshape(batch_size, channels//head_dim, head_dim, dhw)
        value = value.permute(0, 1, 3, 2)  # batch, head, d*h*w, head_c

        sim_map = torch.matmul(query, key)  # batch, head, dhw, dhw

        if self.matmul_norm:
            sim_map = (head_dim ** -0.5) * sim_map

        sim_map = F.softmax(sim_map, dim=-1)    # batch, head, dhw, dhw
        context = torch.matmul(sim_map, value)  # batch, head, dhw, head_c

        context = context.permute(0, 1, 3, 2).flatten(1, 2).contiguous()         # batch, channels, d*h*w
        context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # batch, channels, d, h, w
        if self.out_project is not None:
            context = self.out_project(context)
        return context


    # def forward(self, query_feats, key_feats):
    #     # query_feats: [batch, channels, disparity, height ,width]
    #     batch_size = query_feats.size(0)
    #     query = self.query_project(query_feats)
    #     if self.query_downsample is not None: query = self.query_downsample(query)
    #     #query: b, c, d, h*w
    #     query = query.reshape(*query.shape[:3], -1)
    #     query = query.permute(0, 3, 2, 1).contiguous()  # batch, h*w, disparity, channels
    #
    #     key = self.key_project(key_feats)
    #     value = self.value_project(key_feats)
    #     if self.key_downsample is not None:
    #         key = self.key_downsample(key)
    #         value = self.key_downsample(value)
    #
    #     key = key.reshape(*key.shape[:3], -1)   # batch, channels, d, h*w
    #     key = key.permute(0, 3, 1, 2).contiguous()  # batch, h*w, channels, d
    #     value = value.reshape(*value.shape[:3], -1)  # batch, channels, d, h*w
    #     value = value.permute(0, 3, 2, 1).contiguous()  # batch, h*w, d, channels
    #     # import pdb
    #     # pdb.set_trace()
    #     sim_map = torch.matmul(query, key)  # batch, h*w, d, d
    #
    #     if self.matmul_norm:
    #         sim_map = (self.transform_channels ** -0.5) * sim_map
    #
    #     sim_map = F.softmax(sim_map, dim=-1)    # batch, h*w, d, d
    #     context = torch.matmul(sim_map, value)  # batch, h*w, disparity, channels
    #     context = context.permute(0, 3, 2, 1).contiguous()  # batch, channels, d, h*w
    #     context = context.reshape(batch_size, -1, *query_feats.shape[2:])  # batch, channels, d, h, w
    #     if self.out_project is not None:
    #         context = self.out_project(context)
    #     return context

    '''build project'''
    def buildproject(self, in_channels, out_channels, num_convs, use_norm):
        if use_norm:
            convs = [
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm3d(out_channels),
                    nn.LeakyReLU(0.1, inplace=True)
                )
            ]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Sequential(
                        nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                        nn.BatchNorm3d(out_channels),
                        nn.LeakyReLU(0.1, inplace=True)
                    )
                )
        else:
            convs = [nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)]
            for _ in range(num_convs - 1):
                convs.append(
                    nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
                )
        if len(convs) > 1: return nn.Sequential(*convs)
        return convs[0]
