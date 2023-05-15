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

import torch

from ..head import build_head
from .one_stage_detector import OneStageDetector


class NanoDetPlus(OneStageDetector):
    def __init__(
            self,
            cfg,
            backbone,
            fpn,
            aux_head,
            head,
            detach_epoch=0,
    ):
        super(NanoDetPlus, self).__init__(
            cfg=cfg,
        )
        self.head_cfg = head
        self.aux_fpn = copy.deepcopy(self.fpn)
        self.aux_head_names = aux_head.need_aux_heads
        for aux_name in aux_head.need_aux_heads:
            aux_head['num_classes'] = eval('self.%s' % aux_name).num_classes
            setattr(self, 'aux_%s' % (aux_name), build_head(aux_head))
        self.detach_epoch = detach_epoch

    def forward_train(self, gt_meta, branch='sup'):
        img = gt_meta["img"]
        device = img.device
        feat = self.backbone(img)
        fpn_feat = self.fpn(feat)
        total_loss = 0.0
        total_loss_states = {}
        outs = {}
        if self.epoch >= self.detach_epoch:
            aux_fpn_feat = self.aux_fpn([f.detach() for f in feat])
            dual_fpn_feat = [
                torch.cat([f.detach(), aux_f], dim=1)
                for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            ]
        else:
            aux_fpn_feat = self.aux_fpn(feat)
            dual_fpn_feat = [
                torch.cat([f, aux_f], dim=1) for f, aux_f in zip(fpn_feat, aux_fpn_feat)
            ]

        for aux_head_name in self.aux_head_names:
            valid_idx = self.get_valid_idx(gt_meta, aux_head_name, device, branch)
            if not valid_idx.shape[0]:
                continue
            aux_head = eval('self.aux_%s' % aux_head_name)
            head = eval('self.%s' % aux_head_name)
            head_out = head(fpn_feat)
            outs[aux_head_name.replace('_head', '')] = head_out
            aux_head_out = aux_head(dual_fpn_feat)
            loss, loss_states = head.loss(head_out,
                                          gt_meta,
                                          valid_idx=valid_idx,
                                          key_name_pair=('gt_bboxes', 'gt_labels') if aux_head_name != 'face_head' \
                                              else ('gt_face_bboxes', 'gt_face_labels'),
                                          aux_preds=aux_head_out)
            total_loss += loss
            for loss_name, val in loss_states.items():
                total_loss_states['%s_%s' % (aux_head_name, loss_name)] = val
        return outs, total_loss, total_loss_states
