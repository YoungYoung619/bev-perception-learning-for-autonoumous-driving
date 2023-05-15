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

from .lift_splate_shoot_generator import LiftSplatShootBEVGenerator
from .bevdepth_generator import BevDepthBEVGenerator

def build_bev_generator(cfg, bev_generator_cfg):
    bev_generator_cfg = copy.deepcopy(bev_generator_cfg)
    name = bev_generator_cfg.pop("name")
    if name == "LiftSplatShootBEVGenerator":
        return LiftSplatShootBEVGenerator(
            target_cameras=cfg.data.train.target_cameras,
            input_size=cfg.data.train.input_size,
            feat_strides=cfg.model.arch.fpn.out_strides,
            **bev_generator_cfg)
    elif name == 'BevDepthBEVGenerator':
        return BevDepthBEVGenerator(
            target_cameras=cfg.data.train.target_cameras,
            input_size=cfg.data.train.input_size,
            feat_strides=cfg.model.arch.fpn.out_strides,
            **bev_generator_cfg)
    else:
        raise NotImplementedError
