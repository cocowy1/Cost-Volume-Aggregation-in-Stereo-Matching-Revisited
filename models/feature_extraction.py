from __future__ import print_function
import torch.utils.data
from models.submodule import *
import math

class feature_extraction(nn.Module):
    def __init__(self, concat_feature=False, concat_feature_channel=12):
        super(feature_extraction, self).__init__()
        self.concat_feature = concat_feature

        self.inplanes = 32
        self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True),
                                       convbn(32, 32, 3, 1, 1, 1),
                                       nn.LeakyReLU(0.1, inplace=True))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1, 1, 1)
        self.layer2 = self._make_layer(BasicBlock, 64, 1, 2, 1, 1)
        self.layer3 = self._make_layer(BasicBlock, 128, 1, 2, 1, 1)
        self.layer4 = self._make_layer(BasicBlock, 256, 1, 2, 1, 1)

        self.pyramid_pooling = pyramidPooling(256, None, fusion_mode='sum', model_name='icnet')

        self.upconv4 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(256, 192, 3, 1, 1, 1),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.iconv4 = nn.Sequential(convbn(320, 192, 3, 1, 1, 1),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.upconv3 = nn.Sequential(nn.Upsample(scale_factor=2),
                                     convbn(192, 128, 3, 1, 1, 1),
                                     nn.LeakyReLU(0.1, inplace=True))
        self.iconv3 = nn.Sequential(convbn(192, 128, 3, 1, 1, 1),
                                     nn.LeakyReLU(0.1, inplace=True))

        self.gw = nn.Sequential(convbn(128, 160, 3, 1, 1, 1),
                                 nn.LeakyReLU(0.1, inplace=True),
                                 nn.Conv2d(160, 160, kernel_size=1, padding=0, stride=1,
                                           bias=False))

        if self.concat_feature:
            self.concat3 = nn.Sequential(convbn(128, 128, 3, 1, 1, 1),
                                          nn.LeakyReLU(0.1, inplace=True),
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
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.firstconv(x)
        x = self.layer1(x)                                                                      # scale 1/2
        l2 = self.layer2(x)                                                                     # scale 1/4
        l3 = self.layer3(l2)                                                                    # scale 1/8
        l4 = self.layer4(l3)                                                                    # scale 1/16                                                                  # scale 1/32
        l4 = self.pyramid_pooling(l4)                                                           # scale 1/16

        concat_8res = torch.cat((l3, self.upconv4(l4)), dim=1)              # 256, 1/8
        decov_8res = self.iconv4(concat_8res)                               # 192, 1/8
        concat3 = torch.cat((l2, self.upconv3(decov_8res)), dim=1)          # 192, 1/4
        decov_3 = self.iconv3(concat3)                                      # 128, 1/4

        gw3 = self.gw(decov_3)

        if not self.concat_feature:
            if is_list:
                gw3_left, gw3_right = torch.split(gw3, [batch_dim, batch_dim], dim=0)
                return {"gw_left": gw3_left, "gw_right": gw3_right}
            else:
                return {"gw": gw3}

        else:
            concat_feature3 = self.concat3(decov_3)
            if is_list:
                gw3_left, gw3_right = torch.split(gw3, [batch_dim, batch_dim], dim=0)
                concat_feature3_left, concat_feature3_right = torch.split(concat_feature3, [batch_dim, batch_dim], dim=0)
                return {"gwc_left_feature": gw3_left, "concat_left_feature": concat_feature3_left}, \
                        {"gwc_right_feature": gw3_right, "concat_right_feature": concat_feature3_right}

            else:
                return {"gwc_feature": gw3, "concat_feature": concat_feature3}
