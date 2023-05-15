# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy

from .custom_csp import CustomCspNet
from .efficientnet_lite import EfficientNetLite
from .ghostnet import GhostNet
from .mobilenetv2 import MobileNetV2
from .repvgg import RepVGG
from .resnet import ResNet
from .shufflenetv2 import ShuffleNetV2
from .timm_wrapper import TIMMWrapper
from .mobile_former.mobile_former import build_mobile_former
from .mobilevit_v2.mobievit_v2 import MobileViTv2


def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "ResNet":
        return ResNet(**backbone_cfg)
    elif name == "ShuffleNetV2":
        return ShuffleNetV2(**backbone_cfg)
    elif name == "GhostNet":
        return GhostNet(**backbone_cfg)
    elif name == "MobileNetV2":
        return MobileNetV2(**backbone_cfg)
    elif name == "EfficientNetLite":
        return EfficientNetLite(**backbone_cfg)
    elif name == "CustomCspNet":
        return CustomCspNet(**backbone_cfg)
    elif name == "RepVGG":
        return RepVGG(**backbone_cfg)
    elif name == "TIMMWrapper":
        return TIMMWrapper(**backbone_cfg)
    elif name == "MobileFormer":
        return build_mobile_former(**backbone_cfg)
    elif name == "MobileViTv2":
        return MobileViTv2(**backbone_cfg)
    else:
        raise NotImplementedError
