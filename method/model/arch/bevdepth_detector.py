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

import time
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

from ..backbone import build_backbone
from ..fpn import build_fpn
from ..head import build_head
from ..bev_generator import build_bev_generator
from pyquaternion import Quaternion
from .one_stage_detector import OneStageDetector

from ...memonger import SublinearSequential


class BevDepthDetector(OneStageDetector):
    def __init__(
            self,
            cfg
    ):
        nn.Module.__init__(self)
        backbone_cfg = cfg.model.arch.backbone
        fpn_cfg = cfg.model.arch.fpn if hasattr(cfg.model.arch, 'fpn') else None
        bev_generator_cfg = cfg.model.arch.bev_generator if hasattr(cfg.model.arch, 'bev_generator') else None
        head_cfg = cfg.model.arch.head
        self.head_cfg = head_cfg
        self.bev_generator_cfg = bev_generator_cfg

        self.backbone = build_backbone(backbone_cfg)

        if fpn_cfg is not None:
            self.fpn = build_fpn(fpn_cfg)
        if bev_generator_cfg is not None:
            self.bev_generator = build_bev_generator(cfg, bev_generator_cfg)

        if self.head_cfg is not None:
            self.head_names = list(self.head_cfg.keys())
            for head_name in self.head_names:
                if 'head' not in head_name:
                    continue
                setattr(self, head_name, build_head(cfg, head_cfg[head_name]))
        else:
            raise Exception('Invalid head_cfg')
        self.epoch = 0
        if 'save_memory_mode' in cfg.model.arch and cfg.model.arch.save_memory_mode:
            self.save_memory_mode()

    def save_memory_mode(self):
        self.backbone = SublinearSequential(
            *list(self.backbone.children())
        )

    def get_head_names(self):
        return self.head_names

    def get_head_types(self):
        return [self.head_cfg[head_name].name for head_name in self.head_names]

    def prepare_inputs(self, head_type, fpn_feats, meta):
        device = fpn_feats[0][0].device
        nsweeps = len(fpn_feats)
        if head_type in ['BevDepthBEVGenerator']:
            if meta is None:
                return {'feats': fpn_feats,
                        'rots': None,
                        'trans': None,
                        'intrins': None,
                        'warps': None,
                        'post_trans': None,
                        'post_rots': None,
                        'bev_aug_args': None}
            inputs = {
                'feats': fpn_feats,
                'rots': [],
                'trans': [],
                'intrins': [],
                'warps': [],
                'post_trans': [],
                'post_rots': [],
                'bev_aug_args': []
            }

            # input_stages = self.head_cfg[head_name].input_stages
            camera_types = meta['camera_types'][0]
            # feats = [f for i, f in enumerate(fpn_feats) if i in input_stages]
            feats = fpn_feats
            rots = []
            trans = []
            intrins = []
            warps = []
            post_trans = []
            post_rots = []
            for camera_type in camera_types:
                calibration = meta['ann_info']['camera_calibration'][camera_type]
                translation = torch.from_numpy(np.array(calibration['translation'], dtype=np.float32)).to(device)
                rotation = np.array([[Quaternion(sweep_q).rotation_matrix for sweep_q in batch_q] for batch_q in calibration['rotation']], np.float32)
                rotation = torch.from_numpy(rotation).to(device)
                intrinsic = torch.from_numpy(np.array(calibration['camera_intrinsic'], dtype=np.float32)).to(device)
                warp_matrix = torch.from_numpy(np.array(meta['warp_matrix'][camera_type], dtype=np.float32)).to(device)
                post_tran = None
                if 'post_tran' in meta:
                    post_tran = torch.from_numpy(np.array(meta['post_tran'][camera_type], dtype=np.float32)).to(device)
                post_rot = None
                if 'post_rot' in meta:
                    post_rot = torch.from_numpy(np.array(meta['post_rot'][camera_type], dtype=np.float32)).to(device)
                rots.append(rotation)
                trans.append(translation)
                warps.append(warp_matrix)
                intrins.append(intrinsic)
                post_trans.append(post_tran)
                post_rots.append(post_rot)
            rots = torch.stack(rots).permute(1, 2, 0, 3, 4)
            trans = torch.stack(trans).permute(1, 2, 0, 3)
            intrins = torch.stack(intrins).permute(1, 2, 0, 3, 4)
            warps = torch.stack(warps).permute(1, 2, 0, 3, 4)
            if 'post_rot' in meta:
                post_rots = torch.stack(post_rots).permute(1, 2, 0, 3, 4)
            if 'post_tran' in meta:
                post_trans = torch.stack(post_trans).permute(1, 2, 0, 3)

            inputs['rots'] = rots
            inputs['trans'] = trans
            inputs['intrins'] = intrins
            inputs['warps'] = warps
            inputs['post_trans'] = post_trans if 'post_tran' in meta else None
            inputs['post_rots'] = post_rots if 'post_rot' in meta else None
            inputs['bev_aug_args'] = meta['bev_aug_args'] if 'bev_aug_args' in meta else None

            feats = []
            for vals in zip(*inputs['feats']):
                feats_single_scale = torch.stack(vals, dim=1)
                feats.append(feats_single_scale)

            inputs['feats'] = feats

            return inputs
        else:
            raise NotImplementedError

    def forward(self, x, meta=None):
        b, nsweeps, ncams, c, h, w = x.size()

        feats = []
        for i in range(nsweeps):
            x_ = x[:, i, ...].reshape(-1, c, h, w)
            if i == 0:
                x_ = self.backbone(x_)
                if hasattr(self, "fpn"):
                    x_ = self.fpn(x_)
            else:
                with torch.no_grad():
                    x_ = self.backbone(x_)
                    if hasattr(self, "fpn"):
                        x_ = self.fpn(x_)
            feats.append(x_)

        if hasattr(self, "bev_generator"):
            input = self.prepare_inputs(self.bev_generator_cfg.name, feats, meta)
            bev_feat = self.bev_generator(**input)

        outs = {}
        for head_name in self.head_names:
            head_func = eval("self.%s" % (head_name))
            out = head_func(bev_feat)
            outs[head_name[:-5]] = out

        if torch.onnx.is_in_onnx_export():
            return self._forward_onnx(outs)
        return outs

    def _forward_onnx(self, outs):
        outputs = []

        if hasattr(self, "moving_obstacle_head"):
            outputs.extend(outs['moving_obstacle'])

        if hasattr(self, "face_head"):
            outputs.extend(outs['face'])

        return outputs

    def device_query(self):
        tensor = next(self.parameters())
        return tensor.device

    def inference(self, meta, warp=True):
        results = {}
        device = self.device_query()
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            preds = self(meta["imgs"].to(device), meta)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # ------------ post process for each head -------------- #
            for head_name in self.head_names:
                head_func = eval("self.%s" % (head_name))
                res = head_func.post_process(preds[head_name[:-5]], meta, warp=warp)
                results[head_name[:-5]] = res
            # ------------ post process for each head -------------- #

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        return results

    def forward_train(self, gt_meta, branch='sup'):
        """
        Args:
            gt_meta:
            branch:  sup: 给多任务正常训练使用, unsup: ateacher中的
        Returns:
        """
        assert branch in ['sup', 'unsup']
        device = gt_meta["imgs"].device
        preds = self(gt_meta["imgs"], gt_meta)
        total_loss = 0.0
        total_loss_states = {}

        for head_name in self.head_names:
            head_func = eval("self.%s" % (head_name))
            loss, loss_states = head_func.calc_loss(preds[head_name[:-5]], gt_meta)
            total_loss += loss
            for loss_name, loss_val in loss_states.items():
                total_loss_states["%s_%s" % (head_name[:-5], loss_name)] = loss_val

        return preds, total_loss, total_loss_states

    def set_epoch(self, epoch):
        self.epoch = epoch
