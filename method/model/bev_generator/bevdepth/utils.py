import torch
from torch import nn
import torch.nn.functional as F

from mmdet.models.backbones.resnet import BasicBlock
from mmcv.cnn import build_conv_layer
from mmdet.models import build_backbone
from mmdet3d.models import build_neck
from torch.cuda.amp import autocast
from method.data.transform.warp import BevTransform
from ..lift_splate_shoot.utils import (
    SimpleConvEncoder
)


class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, rots, trans, intrins, warps, bev_aug_args):
        device = x.device
        if rots.ndim == 4:
            bs, n_cams, _, _ = rots.size()
            sensor2ego = torch.eye(4).to(device)[None, None, :, :].repeat(bs, n_cams, 1, 1)
            sensor2ego[..., :3, :3] = rots
            sensor2ego[..., :3, -1] = trans

            sensor2ego = sensor2ego[:, None, ...]
            ida = warps[:, None, ...]
            intrins = intrins[:, None, ...]
            rot_mats = torch.eye(4).to(device)[None, :, :].repeat(bs, 1, 1)
            rot_mats_3x3 = torch.stack([BevTransform.transform_matrix(*args) for args in bev_aug_args]).to(device)
            rot_mats[:, :3, :3] = rot_mats_3x3
            bda = rot_mats
        elif rots.ndim == 5:
            bs, n_sweeps, n_cams, _, _ = rots.size()
        else:
            raise ValueError

        intrins = intrins[:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = ida[:, 0:1, ...]
        sensor2ego = sensor2ego[:, 0:1, ..., :3, :]
        bda = bda.view(batch_size, 1, 1, 4, 4).repeat(1, 1, num_cams, 1, 1)
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 2],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 2],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.adpative_mps_depth_conv(depth)
        return torch.cat([depth, context], dim=1)

    def adpative_mps_depth_conv(self, depth):
        raw_device = depth.device
        if hasattr(depth, 'is_mps') and depth.is_mps:
            if torch.cuda.is_available():
                depth = depth.to('cuda')
                self.depth_conv.to('cuda')
            else:
                depth = depth.to('cpu')
                self.depth_conv.to('cpu')
        depth = self.depth_conv(depth)
        if raw_device != depth.device:
            depth = depth.to(raw_device)
        return depth


bev_backbone_conf = dict(
    type='ResNet',
    in_channels=80,
    depth=18,
    num_stages=3,
    strides=(1, 2, 2),
    dilations=(1, 1, 1),
    out_indices=[0, 1, 2],
    norm_eval=False,
    base_channels=160,
)

bev_neck_conf = dict(type='SECONDFPN',
                     in_channels=[80, 160, 320, 640],
                     upsample_strides=[1, 2, 4, 8],
                     out_channels=[64, 64, 64, 64])

pos_embedding_conf = dict(
    enabel=True)


class BevEncode(nn.Module):
    def __init__(self,
                 image_channel,
                 dx,
                 bx,
                 nx,
                 bev_xy_transpose,
                 multi_frame,
                 bev_backbone_conf=bev_backbone_conf,
                 bev_neck_conf=bev_neck_conf,
                 pos_embedding_conf=pos_embedding_conf,
                 enable_pos_embedding=None):
        super().__init__()
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.bev_xy_transpose = bev_xy_transpose
        bev_backbone_conf['in_channels'] = image_channel
        bev_neck_conf['in_channels'][0] = image_channel
        if multi_frame:
            bev_backbone_conf['in_channels'] *= 2
            bev_neck_conf['in_channels'][0] *= 2
        if enable_pos_embedding is not None:
            self.enable_pos_embedding = enable_pos_embedding
        else:
            self.enable_pos_embedding = pos_embedding_conf['enabel']
        if self.enable_pos_embedding:
            print("enable pos embedding...")
            self.pos_embeddings = self.create_pos_embedding(bev_neck_conf['upsample_strides'])
            bev_backbone_conf['in_channels'] += 4
            bev_neck_conf['in_channels'] = [in_chnl + 4 for in_chnl in bev_neck_conf['in_channels']]

        self.bev_neck_conf = bev_neck_conf
        self.trunk = build_backbone(bev_backbone_conf)
        self.trunk.init_weights()
        self.neck = build_neck(bev_neck_conf)
        self.neck.init_weights()

    def create_pos_embedding(self, out_strides):
        xs = torch.linspace(0, self.nx[0] - 1, self.nx[0], dtype=torch.float).view(1, self.nx[0]).repeat(self.nx[1],
                                                                                                         1) + 0.5
        ys = torch.linspace(0, self.nx[1] - 1, self.nx[1], dtype=torch.float).view(self.nx[1], 1).repeat(1, self.nx[
            0]) + 0.5
        xys = torch.stack([xs, ys], -1)
        grids = (xys * self.dx[:2] + (self.bx[:2] - self.dx[:2] / 2.)) / -self.bx[:2]  # x y
        p = torch.sqrt(grids[..., 0] ** 2 + grids[..., 1] ** 2)
        if not self.bev_xy_transpose:
            grids = grids.flip(dims=[-1])  # yx
        alpha = grids / p[:, :, None]  # sin cos or cos sin
        pos_embedding = torch.cat([grids, alpha], dim=-1)

        pos_embedding_layers = []
        for stride in out_strides:
            pos_embedding_each_layer = pos_embedding[::stride, ::stride]
            pos_embedding_each_layer = pos_embedding_each_layer.permute(2, 0, 1)[None, ...]
            pos_embedding_layers.append(pos_embedding_each_layer)
        return pos_embedding_layers

    @autocast(False)
    def forward(self, x):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results for tasks.
        """
        # FPN
        device = x.device
        bs, _, _, _ = x.size()
        if self.enable_pos_embedding:
            pos_embedding = self.pos_embeddings[0].repeat(bs, 1, 1, 1).to(device)
            x = torch.cat([x, pos_embedding], dim=1)

        trunk_outs = [x]
        if self.trunk.deep_stem:
            x = self.trunk.stem(x)
        else:
            x = self.trunk.conv1(x)
            x = self.trunk.norm1(x)
            x = self.trunk.relu(x)
        for i, layer_name in enumerate(self.trunk.res_layers):
            res_layer = getattr(self.trunk, layer_name)
            x = res_layer(x)
            if i in self.trunk.out_indices:
                if self.enable_pos_embedding:
                    pos_embedding = self.pos_embeddings[i + 1].repeat(bs, 1, 1, 1).to(device)
                    x_out = torch.cat([x, pos_embedding], dim=1)
                    trunk_outs.append(x_out)
                else:
                    trunk_outs.append(x)
        fpn_output = self.neck(trunk_outs)
        return fpn_output[0]


class DepthAggregation(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x