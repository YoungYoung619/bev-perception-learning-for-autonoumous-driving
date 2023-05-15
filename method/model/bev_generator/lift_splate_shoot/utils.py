import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.resnet import resnet18
import math

from ...module.conv import ConvModule
from ...module.init_weights import normal_init
from method.util.mps_tools import (
    adaptive_mps_inverse,
    adaptive_mps_matmul,
    adaptive_mps_argsort,
    adaptive_mps_cumsum
)

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = adaptive_mps_cumsum(x, dim=0)
    # x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        # x = x.cumsum(0)
        x = adaptive_mps_cumsum(x, dim=0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        # back = torch.cumsum(kept, 0)
        back = adaptive_mps_cumsum(kept, dim=0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class SimpleConvEncoder(nn.Module):
    def __init__(
            self,
            input_channel,
            feat_channel,
            last_channel,
            stacked_convs=2,
            conv_cfg=None,
            norm_cfg=dict(type="BN"),
            activation="ReLU",
            kernel_size=3,
            **kwargs
    ):
        super(SimpleConvEncoder, self).__init__()
        self.input_channel = input_channel
        self.feat_channel = feat_channel
        self.last_channel = last_channel
        self.in_channel = input_channel
        self.stacked_convs = stacked_convs
        self.kernel_size = kernel_size
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.activation = activation

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        self.convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.input_channel if i == 0 else self.feat_channel
            self.convs.append(
                ConvModule(
                    chn,
                    self.feat_channel,
                    self.kernel_size,
                    stride=1,
                    padding=self.kernel_size//2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                )
            )
        self.final_conv = nn.Conv2d(
            self.feat_channel if self.stacked_convs > 0 else self.input_channel,
            self.last_channel, self.kernel_size, padding=self.kernel_size//2
        )
        pass

    def init_weights(self):
        for m in self.convs:
            normal_init(m.conv, std=0.01)
        bias_cls = -4.595
        normal_init(self.final_conv, std=0.01, bias=bias_cls)

    def forward(self, feats):
        x = feats
        for conv in self.convs:
            x = conv(x)
        outs = self.final_conv(x)
        return outs


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )
        self.init_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()