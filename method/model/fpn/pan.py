# Modification 2020 RangiLyu
# Copyright 2018-2019 Open-MMLab.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn.functional as F
import torch.nn as nn
from .fpn import FPN
from ..module.conv import ConvModule, DepthwiseConvModule


class PAN(FPN):
    """Path Aggregation Network for Instance Segmentation.

    This is an implementation of the `PAN in Path Aggregation Network
    <https://arxiv.org/abs/1803.01534>`_.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        activation (str): Config dict for activation layer in ConvModule.
            Default: None.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            num_outs,
            start_level=0,
            end_level=-1,
            num_extra_level=0,
            use_depthwise=False,
            conv_cfg=None,
            norm_cfg=None,
            activation=None,
            **kwargs,
    ):
        super(PAN, self).__init__(
            in_channels,
            out_channels,
            num_outs,
            start_level,
            end_level,
            conv_cfg,
            norm_cfg,
            activation,
        )

        conv = DepthwiseConvModule if use_depthwise else ConvModule

        # extra layers
        self.extra_lvl_in_conv = nn.ModuleList()
        self.extra_lvl_out_conv = nn.ModuleList()
        for i in range(num_extra_level):
            self.extra_lvl_in_conv.append(
                conv(
                    in_channels[-1] if i == 0 else out_channels,
                    out_channels,
                    5,
                    stride=2,
                    padding=5 // 2,
                    norm_cfg=dict(type="BN"),
                    activation="LeakyReLU",
                )
            )
            self.extra_lvl_out_conv.append(
                conv(
                    out_channels,
                    out_channels,
                    5,
                    stride=2,
                    padding=5 // 2,
                    norm_cfg=dict(type="BN"),
                    activation="LeakyReLU",
                )
            )

        self.init_weights()

    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] += F.interpolate(
                laterals[i], size=prev_shape, mode="bilinear"
            )

        # build outputs
        # part 1: from original levels
        inter_outs = [laterals[i] for i in range(used_backbone_levels)]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            prev_shape = inter_outs[i + 1].shape[2:]
            inter_outs[i + 1] += F.interpolate(
                inter_outs[i], size=prev_shape, mode="bilinear"
            )

        outs = []
        outs.append(inter_outs[0])
        outs.extend([inter_outs[i] for i in range(1, used_backbone_levels)])

        # extra layers
        for extra_in_layer, extra_out_layer in zip(
                self.extra_lvl_in_conv, self.extra_lvl_out_conv
        ):
            outs.append(extra_in_layer(inputs[-1]) + extra_out_layer(outs[-1]))
        return tuple(outs)
