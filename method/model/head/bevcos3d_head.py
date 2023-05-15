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

import torch
import torch.nn as nn
import math
from torch.cuda.amp import autocast
import torch.nn.functional as F

from ..module.conv import ConvModule, DepthwiseConvModule
from ..module.init_weights import normal_init
from .gfl_head import GFLHead
from .assigner.atss_assigner import ATSSAssigner
from ..loss.gfocal_loss import QualityFocalLoss
from tools.nuscene.data_check import is_target_cls
from pyquaternion import Quaternion
import numpy as np
from method.util.mps_tools import adaptive_mps_unique
from method.util import (
    images_to_levels,
    multi_apply,
    rotated_box_ops
)
from method.data.transform.warp import BevTransform

from scipy.spatial.transform import Rotation as R
from mmdet3d.models.dense_heads.centerpoint_head import circle_nms

CIRCLE_NMS_THRESH = {
    'car': 4,
    'truck': 12,
    'construction_vehicle': 12,
    'bus': 10,
    'trailer': 10,
    'barrier': 1,
    'motorcycle': 0.85,
    'bicycle': 0.85,
    'pedestrian': 0.175,
    'traffic_cone': 0.175
}


class BevCos3DHead(nn.Module):
    """
    在此方法中，我们将bev特征图当成是传统2d目标检测检测头的输出，选取物体3d属性中的x,y,w,l来当成是目标检测的2d框，用上述属性来选取正负例，
    正负例的规则和ATSS一致，然后正例用来预测物体完整的2d属性（x,y,z,w,l,h,sin_alpha,cos_alpha）
    """

    def __init__(self,
                 loss,
                 input_channel,
                 target_names,
                 target_maps,
                 xbound,
                 ybound,
                 zbound,
                 dbound,
                 conf_thresh,
                 nms_thresh,
                 stacked_convs=2,
                 octave_base_scale=5,
                 conv_type="DWConv",
                 conv_cfg=None,
                 norm_cfg=dict(type="BN"),
                 share_cls_reg=False,
                 activation="ReLU",
                 feat_channels=256,
                 yaw_format=None,
                 with_velocity=False,
                 **kwargs):
        super().__init__()
        self.share_cls_reg = share_cls_reg
        self.activation = activation
        self.ConvModule = ConvModule if conv_type == "Conv" else DepthwiseConvModule
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.featmap_size = [math.ceil((xbound[1] - xbound[0]) / xbound[2]),
                             math.ceil((ybound[1] - ybound[0]) / ybound[2])]  # img height width
        self.in_channels = input_channel
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.grid_cell_scale = octave_base_scale
        # x, y, z, l, w, h, [sin(alpha), cos(alpha)| radian]
        self.yaw_format = yaw_format
        self.box_dim = 8 if yaw_format.name == 'sincos' else 7
        self.with_velocity = with_velocity
        if self.with_velocity:
            self.box_dim += 2
        self.yaw_use_sigmoid = yaw_format.use_sigmoid if 'use_sigmoid' in yaw_format else False
        self.target_names = target_names
        self.target_maps = target_maps
        self.class_names = []
        for map in self.target_maps:
            if map not in self.class_names:
                self.class_names.append(map)  # append by maps' order
        self.n_classes = len(self.class_names)
        self.loss_cfg = loss
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.assigner = ATSSAssigner(topk=1, xbound=xbound, ybound=ybound)
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh

        self._init_layers()
        self.init_weights()

        if loss.loss_box.name == 'SmoothL1':
            self.box_loss = torch.nn.SmoothL1Loss(reduction='none')
        elif loss.loss_box.name == 'L1':
            self.box_loss = torch.nn.L1Loss(reduction='none')
        elif loss.loss_box.name == 'L2':
            self.box_loss = torch.nn.MSELoss(reduction='none')
        else:
            raise NotImplementedError
        self.loss_box_weight = loss.loss_box.loss_weight
        self.weights_cls_to_box = loss.loss_box.weights_cls_to_box
        self.with_depth_supervised = True if 'depth_supervised' in loss else False
        if self.with_depth_supervised:
            self.depth_channel = loss.depth_supervised.depth_channel
            self.downsample_factor = loss.depth_supervised.downsample_factor

        if loss.loss_cls.name == 'QualityFocalLoss':
            self.cls_loss = QualityFocalLoss(
                use_sigmoid=True,
                beta=loss.loss_cls.beta,
                loss_weight=loss.loss_cls.loss_weight,
            )
        else:
            raise NotImplementedError

    def _init_layers(self):
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for _ in [0]:
            cls_convs, reg_convs = self._buid_not_shared_head()
            self.cls_convs.append(cls_convs)
            self.reg_convs.append(reg_convs)

        self.gfl_cls = nn.ModuleList(
            [
                nn.Conv2d(
                    self.feat_channels,
                    self.n_classes + self.box_dim
                    if self.share_cls_reg
                    else self.n_classes,
                    1,
                    padding=0,
                )
                for _ in [0]
            ]
        )
        # TODO: if
        self.gfl_reg = nn.ModuleList(
            [
                nn.Conv2d(self.feat_channels, self.box_dim, 1, padding=0)
                for _ in [0]
            ]
        )

    def _buid_not_shared_head(self):
        cls_convs = nn.ModuleList()
        reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            cls_convs.append(
                self.ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    norm_cfg=self.norm_cfg,
                    bias=self.norm_cfg is None,
                    activation=self.activation,
                )
            )
            if not self.share_cls_reg:
                reg_convs.append(
                    self.ConvModule(
                        chn,
                        self.feat_channels,
                        3,
                        stride=1,
                        padding=1,
                        norm_cfg=self.norm_cfg,
                        bias=self.norm_cfg is None,
                        activation=self.activation,
                    )
                )

        return cls_convs, reg_convs

    def init_weights(self):
        for m in self.cls_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        for m in self.reg_convs.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)
        # init cls head with confidence = 0.01
        bias_cls = -4.595
        for i in range(1):
            normal_init(self.gfl_cls[i], std=0.01, bias=bias_cls)
            normal_init(self.gfl_reg[i], std=0.01)
        print("Finish initialize NanoDet Head.")

    @autocast(False)
    def forward(self, feats):
        if isinstance(feats, dict):
            depths = feats['depths']
            feats = feats['bev_feats']
        outputs = []
        if not isinstance(feats, list):
            feats = [feats]

        for x, cls_convs, reg_convs, gfl_cls, gfl_reg in zip(
                feats, self.cls_convs, self.reg_convs, self.gfl_cls, self.gfl_reg
        ):
            cls_feat = x
            reg_feat = x
            for cls_conv in cls_convs:
                cls_feat = cls_conv(cls_feat)
            for reg_conv in reg_convs:
                reg_feat = reg_conv(reg_feat)
            if self.share_cls_reg:
                output = gfl_cls(cls_feat)
            else:
                cls_score = gfl_cls(cls_feat)
                bbox_pred = gfl_reg(reg_feat)
                output = torch.cat([cls_score, bbox_pred], dim=1)
            outputs.append(output.flatten(start_dim=2))
        outputs = torch.cat(outputs, dim=2).permute(0, 2, 1)

        if self.yaw_format.name == 'sincos' and self.yaw_use_sigmoid:
            outputs[..., -2:] = torch.sigmoid(outputs[..., -2:])

        return outputs, None if not self.with_depth_supervised else depths

    def in_range(self, translation, xbound, ybound, zbound):
        def is_in_range(val, low, high):
            return low <= val and high >= val

        x, y, z = translation
        return is_in_range(x, xbound[0], xbound[1]) \
               and is_in_range(y, ybound[0], ybound[1]) \
               and is_in_range(z, zbound[0], zbound[1])

    def generate_gt(self, gt_meta, device):
        gts = {
            'gt_boxes_3d': [],
            'gt_labels_3d': [],
        }
        boxes3d = gt_meta['boxes3d']
        for batch_idx, batch_boxes in enumerate(boxes3d):
            gt_boxes_batch = []
            gt_labels_batch = []
            for box in batch_boxes:
                if not is_target_cls(box['category_name'], self.target_names):
                    # 不属于目标物体
                    continue
                if box['num_pts'] == 0:
                    # 没有lidar radar点
                    continue
                if not self.in_range(box['translation'], self.xbound, self.ybound, self.zbound):
                    # 不在感知范围
                    # 不在感知范围
                    continue

                gt_label = None
                for cls_idx, cls_name in enumerate(self.target_names):
                    if cls_name in box['category_name']:
                        idx = self.target_names.index(cls_name)
                        gt_name = self.target_maps[idx]
                        gt_label = self.class_names.index(gt_name)
                if gt_label is None:
                    raise ValueError("something must be wrong")

                w, l, h = box['size']
                x, y, z = box['translation']
                vx, vy = box['velocity']
                quaternion = Quaternion(box['rotation'])
                yaw = quaternion.yaw_pitch_roll[0]

                # do bev augmentation transform
                if 'bev_aug_args' in gt_meta and gt_meta['bev_aug_args']:
                    gt_box = torch.from_numpy(np.array([[x, y, z, w, l, h, yaw, vx, vy]], dtype=np.float32))
                    # gt_box_raw = gt_box.clone()
                    boxes3d_trans = BevTransform.transform_boxes3d(*gt_meta['bev_aug_args'][batch_idx], boxes3d=gt_box)
                    x, y, z, w, l, h, yaw, vx, vy = boxes3d_trans[0].cpu().numpy()

                # 使用底部框中点作为translate
                if self.yaw_format.name == 'sincos':
                    sin_alpha = np.sin(yaw)
                    cos_alpha = np.cos(yaw)
                    box = [x, y, z - h / 2., w, l, h, sin_alpha, cos_alpha]
                elif self.yaw_format.name == 'radian':
                    box = [x, y, z - h / 2., w, l, h, yaw]
                else:
                    raise NotImplementedError
                if self.with_velocity:
                    box.extend([vx, vy])
                gt_boxes_batch.append(box)
                gt_labels_batch.append(gt_label)
            if len(gt_boxes_batch):
                gts['gt_boxes_3d'].append(np.array(gt_boxes_batch, dtype=np.float32))
                gts['gt_labels_3d'].append(np.array(gt_labels_batch, dtype=np.int64))
            else:
                gts['gt_boxes_3d'].append(np.zeros((0, self.box_dim), dtype=np.float32))
                gts['gt_labels_3d'].append(np.array([], dtype=np.int64))
        # gt_meta.update(gts)
        return gts

    def get_single_level_center_point(
            self, featmap_size, stride, dtype, device, flatten=True
    ):
        """
        Generate pixel centers of a single stage feature map.
        :param featmap_size: height and width of the feature map
        :param stride: down sample stride of the feature map
        :param dtype: data type of the tensors
        :param device: device of the tensors
        :param flatten: flatten the x and y tensors
        :return: y and x of the center points
        """
        h, w = featmap_size
        x_range = (torch.arange(w, dtype=dtype, device=device) + 0.5) * stride
        y_range = (torch.arange(h, dtype=dtype, device=device) + 0.5) * stride
        y, x = torch.meshgrid(y_range, x_range)
        if flatten:
            y = y.flatten()
            x = x.flatten()
        return y, x

    def get_grid_cells(self, featmap_size, scale, stride, dtype, device):
        """
        Generate grid cells of a feature map for target assignment.
        :param featmap_size: Size of a single level feature map.
        :param scale: Grid cell scale.
        :param stride: Down sample stride of the feature map.
        :param dtype: Data type of the tensors.
        :param device: Device of the tensors.
        :return: Grid_cells xyxy position. Size should be [feat_w * feat_h, 4]
        """
        cell_size = stride * scale
        y, x = self.get_single_level_center_point(
            featmap_size, stride, dtype, device, flatten=True
        )
        grid_cells = torch.stack(
            [
                x - 0.5 * cell_size,
                y - 0.5 * cell_size,
                x + 0.5 * cell_size,
                y + 0.5 * cell_size,
            ],
            dim=-1,
        )
        return grid_cells

    def sample(self, assign_result, gt_bboxes):
        pos_inds = (
            torch.nonzero(assign_result.gt_inds > 0, as_tuple=False)
                .squeeze(-1)
                .unique()
        )
        neg_inds = (
            torch.nonzero(assign_result.gt_inds == 0, as_tuple=False)
                .squeeze(-1)
                .unique()
        )
        pos_assigned_gt_inds = assign_result.gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)
            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds, :]
        return pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds

    def target_assign_single_img(
            self, grid_cells, num_level_cells, gt_bboxes_3d, gt_bboxes_ignore, gt_labels_3d
    ):
        """
        Using ATSS Assigner to assign target on one image.
        :param grid_cells: Grid cell boxes of all pixels on feature map
        :param num_level_cells: numbers of grid cells on each level's feature map
        :param gt_bboxes_3d: Ground truth boxes
        :param gt_bboxes_ignore: Ground truths which are ignored
        :param gt_labels_3d: Ground truth labels
        :return: Assign results of a single image
        """
        device = grid_cells.device
        gt_bboxes_3d = torch.from_numpy(gt_bboxes_3d).to(device)
        gt_labels_3d = torch.from_numpy(gt_labels_3d).to(device)
        n_bboxes_3d_attrs = gt_bboxes_3d.size()[1]
        n_grid_celss = grid_cells.size()[0]

        assign_result = self.assigner.assign(
            grid_cells, num_level_cells, gt_bboxes_3d, gt_bboxes_ignore, gt_labels_3d
        )

        pos_inds, neg_inds, pos_gt_bboxes, pos_assigned_gt_inds = self.sample(
            assign_result, gt_bboxes_3d
        )

        num_cells = grid_cells.shape[0]
        bbox_targets = torch.zeros(size=(n_grid_celss, n_bboxes_3d_attrs), dtype=torch.float32).to(device)
        bbox_weights = torch.zeros_like(bbox_targets)
        labels = grid_cells.new_full((num_cells,), self.n_classes, dtype=torch.long)
        label_weights = grid_cells.new_zeros(num_cells, dtype=torch.float)

        if len(pos_inds) > 0:
            pos_bbox_targets = pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0
            if gt_labels_3d is None:
                # Only rpn gives gt_labels_3d as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels_3d[pos_assigned_gt_inds]

            label_weights[pos_inds] = 1.0
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        return (
            grid_cells,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def target_assign(
            self,
            cls_preds,
            reg_preds,
            featmap_size,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            device,
    ):
        batch_size = cls_preds.shape[0]
        # get grid cells of one image
        multi_level_grid_cells = self.get_grid_cells(
            featmap_size,
            self.grid_cell_scale,
            stride=1,
            dtype=torch.float32,
            device=device,
        )
        mlvl_grid_cells_list = [multi_level_grid_cells for i in range(batch_size)]

        # pixel cell number of multi-level feature maps
        num_level_cells = [mlvl_grid_cells_list[0].size()[0]]
        num_level_cells_list = [num_level_cells] * batch_size
        # concat all level cells and to a single tensor
        # for i in range(batch_size):
        #     mlvl_grid_cells_list[i] = torch.cat(mlvl_grid_cells_list[i])
        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(batch_size)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(batch_size)]
        # target assign on all images, get list of tensors
        # list length = batch size
        # tensor first dim = num of all grid cell
        (
            all_grid_cells,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self.target_assign_single_img,
            mlvl_grid_cells_list,
            num_level_cells_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
        )
        # no valid cells
        if any([labels is None for labels in all_labels]):
            return None
        # sampled cells of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # merge list of targets tensors into one batch then split to multi levels
        mlvl_cls_preds = images_to_levels([c for c in cls_preds], num_level_cells)
        mlvl_reg_preds = images_to_levels([r for r in reg_preds], num_level_cells)
        mlvl_grid_cells = images_to_levels(all_grid_cells, num_level_cells)
        mlvl_labels = images_to_levels(all_labels, num_level_cells)
        mlvl_label_weights = images_to_levels(all_label_weights, num_level_cells)
        mlvl_bbox_targets = images_to_levels(all_bbox_targets, num_level_cells)
        mlvl_bbox_weights = images_to_levels(all_bbox_weights, num_level_cells)
        return (
            mlvl_cls_preds,
            mlvl_reg_preds,
            mlvl_grid_cells,
            mlvl_labels,
            mlvl_label_weights,
            mlvl_bbox_targets,
            mlvl_bbox_weights,
            num_total_pos,
            num_total_neg,
        )

    def bev_nms(self, batches, scores, bboxes):
        batch_idxs = adaptive_mps_unique(batches)
        bboxes_ = bboxes.clone()
        scores_ = scores.clone()
        keep = torch.zeros_like(scores[:, 0]).bool()
        assert keep[0] == False
        last_batch_size = 0
        for i in batch_idxs:
            mask = batches == i
            boxes_per_batch = bboxes_[mask]
            scores_per_batch = scores_[mask]
            scores_per_batch_max_cls, _ = scores_per_batch.max(dim=-1)
            x, y, w, l = boxes_per_batch[:, 0].clone(), boxes_per_batch[:, 1].clone(), \
                         boxes_per_batch[:, 3].clone(), boxes_per_batch[:, 4].clone()

            if self.yaw_format.name == 'sincos':
                s, c = boxes_per_batch[:, 6].clone(), boxes_per_batch[:, 7].clone()
                if self.yaw_use_sigmoid:
                    s, c = s.sigmoid(), c.sigmoid()
            elif self.yaw_format.name == 'radian':
                s, c = boxes_per_batch[:, 6].sin().clone(), boxes_per_batch[:, 6].cos().clone()
            else:
                raise NotImplementedError
            rotated_boxes = torch.stack([x, y, l, w, c, s, scores_per_batch_max_cls], dim=-1)
            keep_idx = rotated_box_ops.nms(rotated_boxes, self.nms_thresh)
            # print()
            # b = torch.stack([x1, y1, x2, y2, scores_per_batch_max_cls], dim=-1)
            # keep_idx = nms(b.cpu().numpy(), self.nms_thresh)
            # keep_idx = torch.from_numpy(np.array(keep_idx, np.int64)).to(scores.device)
            keep[keep_idx + last_batch_size] = True
            last_batch_size += mask.long().sum()

        return keep

    def to_bev_size(self, bboxes):
        """
        Args:
            bboxes (torch.Tensor): shape (n, 8) x, y, z, w, l, h, sin_alpha, cos_alpha,
        """
        # 将尺度转换成鸟瞰图大小200x200，然后方向变成鸟瞰图坐标系方向(lss中，鸟瞰图高度增加方向为x正向)
        # 注：如果后续有别的方法鸟瞰图的坐标方向和lss中不一致，这里就需要调整
        bboxes_ = bboxes.clone()
        x, y, w, l = bboxes_[:, 0].clone(), bboxes_[:, 1].clone(), bboxes_[:, 3].clone(), bboxes_[:, 4].clone()
        bboxes_[:, 0] = (y - self.ybound[0]) / self.ybound[2]
        bboxes_[:, 1] = (x - self.xbound[0]) / self.xbound[2]
        bboxes_[:, 3] = w / self.ybound[2]
        bboxes_[:, 4] = l / self.xbound[2]
        return bboxes_

    def to_world_size(self, bboxes):
        # 将bev尺寸转化为真实世界尺寸
        bboxes_ = bboxes.clone()
        x, y, w, l = bboxes_[:, 0].clone(), bboxes_[:, 1].clone(), bboxes_[:, 3].clone(), bboxes_[:, 4].clone()
        bboxes_[:, 0] = y * self.ybound[2] + self.ybound[0]
        bboxes_[:, 1] = x * self.xbound[2] + self.xbound[0]
        bboxes_[:, 3] = w * self.ybound[2]
        bboxes_[:, 4] = l * self.xbound[2]
        return bboxes_

    def loss_single(
            self,
            grid_cells,
            cls_score,
            bbox_pred,
            labels,
            label_weights,
            bbox_targets,
            num_total_samples,
    ):
        grid_cells = grid_cells.reshape(-1, 4)
        cls_score = cls_score.reshape(-1, self.n_classes)
        bbox_pred = bbox_pred.reshape(-1, self.box_dim)
        bbox_targets = bbox_targets.reshape(-1, self.box_dim)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.n_classes
        pos_inds = torch.nonzero(
            (labels >= 0) & (labels < bg_class_ind), as_tuple=False
        ).squeeze(1)

        # 分类分数会通过qfl逼近这个分数，先置为1
        score = label_weights.new_zeros(labels.shape)
        score[pos_inds] = 1.

        if len(pos_inds) > 0:
            pos_bbox_gts = bbox_targets[pos_inds]
            pos_bbox_targets = self.to_bev_size(pos_bbox_gts)
            pos_grid_cells = grid_cells[pos_inds][:, :2]  # 取网格左上角做偏移起始点
            pos_bbox_targets[:, 0] -= pos_grid_cells[:, 0]
            pos_bbox_targets[:, 1] -= pos_grid_cells[:, 1]
            # xy_offset = pos_bbox_targets[:, :2]
            # xy_offset = xy_offset.cpu().numpy()

            pos_bbox_pred = bbox_pred[pos_inds]

            # 分类分数高说明越可能是物体，因此分配更高的权重
            if self.weights_cls_to_box:
                weight_targets = cls_score.detach().sigmoid()
                weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            else:
                weight_targets = torch.ones_like(pos_inds, dtype=torch.float32)

            # regression loss
            x_loss = self.box_loss(pos_bbox_targets[:, 0], pos_bbox_pred[:, 0])
            y_loss = self.box_loss(pos_bbox_targets[:, 1], pos_bbox_pred[:, 1])
            z_loss = self.box_loss(pos_bbox_targets[:, 2], pos_bbox_pred[:, 2])
            w_loss = self.box_loss(pos_bbox_targets[:, 3], pos_bbox_pred[:, 3])
            l_loss = self.box_loss(pos_bbox_targets[:, 4], pos_bbox_pred[:, 4])
            h_loss = self.box_loss(pos_bbox_targets[:, 5], pos_bbox_pred[:, 5])
            if self.with_velocity:
                vex_loss = self.box_loss(pos_bbox_targets[:, -2], pos_bbox_pred[:, -2])
                vey_loss = self.box_loss(pos_bbox_targets[:, -1], pos_bbox_pred[:, -1])
            else:
                vex_loss = torch.zeros_like(h_loss)
                vey_loss = torch.zeros_like(h_loss)
            if self.yaw_format.name == 'sincos':
                sin_alpha = self.box_loss(pos_bbox_targets[:, 6], pos_bbox_pred[:, 6])
                cos_alpha = self.box_loss(pos_bbox_targets[:, 7], pos_bbox_pred[:, 7])
                loss_box = (x_loss + y_loss + z_loss + w_loss + l_loss + h_loss \
                            + 10 * sin_alpha + 10 * cos_alpha + vex_loss + vey_loss) * weight_targets / self.box_dim
            elif self.yaw_format.name == 'radian':
                radian = self.box_loss(pos_bbox_targets[:, 6], pos_bbox_pred[:, 6])
                loss_box = (x_loss + y_loss + z_loss + w_loss + l_loss + h_loss \
                            + 3 * radian + vex_loss + vey_loss) * weight_targets / self.box_dim
            else:
                raise NotImplementedError
            loss_box = loss_box.mean()
            if self.yaw_format.name == 'sincos':
                loss_box_states = {
                    'box_x_loss': x_loss.mean(),
                    'box_y_loss': y_loss.mean(),
                    'box_z_loss': z_loss.mean(),
                    'box_w_loss': w_loss.mean(),
                    'box_l_loss': l_loss.mean(),
                    'box_h_loss': h_loss.mean(),
                    'box_sin_alpha': sin_alpha.mean(),
                    'box_cos_alpha': cos_alpha.mean(),
                    'box_vel_loss': vex_loss.mean() + vey_loss.mean(),
                }
            elif self.yaw_format.name == 'radian':
                loss_box_states = {
                    'box_x_loss': x_loss.mean(),
                    'box_y_loss': y_loss.mean(),
                    'box_z_loss': z_loss.mean(),
                    'box_w_loss': w_loss.mean(),
                    'box_l_loss': l_loss.mean(),
                    'box_h_loss': h_loss.mean(),
                    'box_alpha': radian.mean(),
                    'box_vel_loss': vex_loss.mean() + vey_loss.mean(),
                }
            else:
                raise NotImplementedError
        else:
            loss_box = bbox_pred.sum() * 0
            weight_targets = torch.tensor(0).to(cls_score.device)
            loss_box_states = {
                'box_nothing': loss_box,
            }

        # cls loss
        loss_cls = self.cls_loss(
            cls_score,
            (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples,
        )

        return loss_box, loss_cls, weight_targets.sum(), loss_box_states

    def calc_loss(self, preds, gt_meta):
        preds, depth_preds = preds
        device = preds.device
        gts = self.generate_gt(gt_meta, device)
        gt_boxes_3d = gts['gt_boxes_3d']
        gt_labels_3d = gts['gt_labels_3d']
        gt_bboxes_3d_ignore = None

        cls_scores, bbox_preds = preds.split(
            [self.n_classes, self.box_dim], dim=-1
        )

        featmap_size = self.featmap_size

        cls_reg_targets = self.target_assign(
            cls_scores,
            bbox_preds,
            featmap_size,
            gt_boxes_3d,
            gt_bboxes_3d_ignore,
            gt_labels_3d,
            device=device,
        )

        (
            cls_preds_list,
            reg_preds_list,
            grid_cells_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_samples = max(num_total_pos, 1.0)

        losses_bbox, losses_cls, avg_factor, box_states = multi_apply(
            self.loss_single,
            grid_cells_list,
            cls_preds_list,
            reg_preds_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            num_total_samples=num_total_samples,
        )

        # avg_factor = sum(avg_factor) # 感觉没必要
        avg_factor = 1
        if avg_factor <= 0:
            loss_bbox = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
            loss_cls = torch.tensor(0, dtype=torch.float32, requires_grad=True).to(
                device
            )
        else:
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            losses_cls = list(map(lambda x: x / avg_factor, losses_cls))

            loss_cls = sum(losses_cls)
            loss_bbox = sum(losses_bbox)

        loss = self.loss_box_weight * loss_bbox + loss_cls
        loss_states = dict(loss_bbox=loss_bbox, loss_cls=loss_cls)
        loss_states.update(box_states[0])

        if self.with_depth_supervised:
            camera_orders = gt_meta['camera_types'][0]
            depth_gts = [gt_meta['depth_imgs'][cam] for cam in camera_orders]
            depth_gts = torch.stack(depth_gts, dim=1)
            depth_loss = self.get_depth_loss(depth_gts, depth_preds)
            loss += depth_loss
            loss_states.update({'depth_loss': depth_loss})
        loss_states.update({'total_loss': loss})

        return loss, loss_states

    def get_depth_loss(self, depth_labels, depth_preds):
        depth_loss_total = 0.
        for i, depth_preds_single_scale in enumerate(depth_preds):
            depth_labels_single_scale = self.get_downsampled_gt_depth(depth_labels, self.downsample_factor[i])
            depth_preds_single_scale = depth_preds_single_scale.permute(0, 2, 3, 1).contiguous().view(
                -1, self.depth_channel)
            fg_mask = torch.max(depth_labels_single_scale, dim=1).values > 0.0

            with autocast(enabled=False):
                depth_loss = (F.binary_cross_entropy(
                    depth_preds_single_scale[fg_mask],
                    depth_labels_single_scale[fg_mask],
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum()))
            depth_loss_total += depth_loss
        depth_loss_total /= len(depth_preds)

        return 3.0 * depth_loss_total

    def get_downsampled_gt_depth(self, gt_depths, downsample_factor):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N,
            H // downsample_factor,
            downsample_factor,
            W // downsample_factor,
            downsample_factor,
            1,
        )
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(
            -1, downsample_factor * downsample_factor)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample_factor,
                                   W // downsample_factor)

        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channel + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channel + 1).view(
            -1, self.depth_channel + 1)[:, 1:]

        return gt_depths.float()

    def post_process(self, preds, meta, warp=True):
        # preds = torch.sigmoid(preds)
        # # preds = (preds > 0.2).float()
        preds, _ = preds
        preds = preds.to('cpu')
        device = preds.device
        bs, featsize, n_attr = preds.size()

        multi_level_grid_cells = self.get_grid_cells(
            self.featmap_size,
            self.grid_cell_scale,
            stride=1,
            dtype=torch.float32,
            device=device,
        )[None, :, :2].repeat(bs, 1, 1)

        cls_scores, bbox_preds = preds.split(
            [self.n_classes, self.box_dim], dim=-1
        )

        shape = cls_scores.size()
        cls_scores = cls_scores.reshape(-1)
        cls_scores = cls_scores.sigmoid()
        cls_scores = cls_scores.reshape(shape)
        cls_max_scores, cls_idxs = torch.max(cls_scores, dim=-1)
        conf_thresh = torch.from_numpy(np.array(self.conf_thresh, dtype=np.float32)).to(device)
        conf_thresh_each = conf_thresh[cls_idxs]

        pos_mask = cls_max_scores > conf_thresh_each
        n_pos = pos_mask.float().sum()

        boxes3d = {}
        if n_pos:
            pos_scores = cls_scores[pos_mask]
            pos_classes = cls_idxs[pos_mask]
            pos_boxes = bbox_preds[pos_mask]
            pos_xy_offsets = multi_level_grid_cells[pos_mask]
            batches = torch.arange(0, bs)[:, None].to(device).repeat(1, featsize)[pos_mask]

            pos_boxes[:, :2] += pos_xy_offsets
            pos_boxes = self.to_world_size(pos_boxes)

            #     keep = self.bev_nms(batches, pos_scores, pos_boxes)
            #
            #     batches = batches[keep]
            #     pos_scores = pos_scores[keep]
            #     pos_classes = pos_classes[keep]
            #     pos_boxes = pos_boxes[keep]
            #
            #     # traverse data
            #     for batch_idx, score, cls_idx, boxes_pred in zip(batches, pos_scores, pos_classes, pos_boxes):
            #         if batch_idx.item() not in boxes3d:
            #             boxes3d[batch_idx.item()] = []
            #         cls_idx = cls_idx.item()
            #         boxes_pred = boxes_pred.cpu().numpy()
            #         xy = boxes_pred[:2]
            #         z = boxes_pred[2]
            #         wlh = boxes_pred[3:6]
            #         z += wlh[-1] / 2.
            #         if self.with_velocity:
            #             velocity = boxes_pred[-2:]
            #         else:
            #             velocity = np.array([np.nan, np.nan])
            #
            #         if self.yaw_format.name == 'sincos':
            #             alpha = boxes_pred[6:8]
            #             yaw = math.atan2(alpha[0], alpha[1])
            #         elif self.yaw_format.name == 'radian':
            #             alpha = boxes_pred[6]
            #             yaw = alpha.item()
            #         else:
            #             raise NotImplementedError
            #
            #         r = R.from_euler('zyx', [yaw, 0., 0.], degrees=False)
            #         xr, yr, zr, w = r.as_quat()
            #         rotation = np.array([w, xr, yr, zr], dtype=np.float32)
            #
            #         box = {
            #             'conf': score[cls_idx].item(),
            #             'class_name': self.class_names[cls_idx],
            #             'class_idx': cls_idx,
            #             'translation': np.array([xy[0], xy[1], z], dtype=np.float32),
            #             'size': wlh,
            #             'rotation': rotation,
            #             'velocity': velocity
            #         }
            #         if self.with_velocity:
            #             box.update({'velocity': np.array([boxes_pred[-2].item(), boxes_pred[-1].item()])})
            #         boxes3d[batch_idx.item()].append(box)
            #
            #     for batch_idx, boxes in boxes3d.items():
            #         boxes = sorted(boxes, key=lambda x: x['conf'], reverse=True)
            #         boxes3d[batch_idx] = boxes[:100]

            for batch_idx in range(bs):
                boxes3d[batch_idx] = []
                b_keep = batches == batch_idx
                for cls_idx in pos_classes.unique():
                    cls_name = self.class_names[cls_idx.item()]
                    boxes = pos_boxes[b_keep][pos_classes[b_keep] == cls_idx]
                    scores = pos_scores[b_keep][pos_classes[b_keep] == cls_idx][:, cls_idx]
                    centers = boxes[:, [0, 1]]
                    centers = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(circle_nms(
                        centers.detach().cpu().numpy(),
                        CIRCLE_NMS_THRESH[cls_name],
                        post_max_size=50),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes = boxes[keep]
                    scores = scores[keep]

                    for box_pred, score in zip(boxes, scores):
                        boxes_pred = box_pred.cpu().numpy()
                        xy = boxes_pred[:2]
                        z = boxes_pred[2]
                        wlh = boxes_pred[3:6]
                        z += wlh[-1] / 2.
                        if self.with_velocity:
                            velocity = boxes_pred[-2:]
                        else:
                            velocity = np.array([np.nan, np.nan])

                        if self.yaw_format.name == 'sincos':
                            alpha = boxes_pred[6:8]
                            yaw = math.atan2(alpha[0], alpha[1])
                        elif self.yaw_format.name == 'radian':
                            alpha = boxes_pred[6]
                            yaw = alpha.item()
                        else:
                            raise NotImplementedError

                        r = R.from_euler('zyx', [yaw, 0., 0.], degrees=False)
                        xr, yr, zr, w = r.as_quat()
                        rotation = np.array([w, xr, yr, zr], dtype=np.float32)

                        box = {
                            'conf': score.item(),
                            'class_name': cls_name,
                            'class_idx': cls_idx.item(),
                            'translation': np.array([xy[0], xy[1], z], dtype=np.float32),
                            'size': wlh,
                            'rotation': rotation,
                            'velocity': velocity
                        }
                        if self.with_velocity:
                            box.update({'velocity': np.array([boxes_pred[-2].item(), boxes_pred[-1].item()])})
                        boxes3d[batch_idx].append(box)

                if len(boxes3d[batch_idx]) == 0:
                    boxes3d[batch_idx].append({
                        'conf': 0.01,
                        'class_name': 'car',
                        'class_idx': 1,
                        'translation': np.array([0, 0, 0], dtype=np.float32),
                        'size': np.array([0, 0, 0], dtype=np.float32),
                        'rotation': np.array([1, 0, 0, 0], dtype=np.float32),
                        'velocity': np.array([0, 0], dtype=np.float32)
                    })

        for i in range(bs):
            if i not in boxes3d:
                boxes3d[i] = [
                    {
                        'conf': 0.01,
                        'class_name': 'car',
                        'class_idx': 1,
                        'translation': np.array([0, 0, 0], dtype=np.float32),
                        'size': np.array([0, 0, 0], dtype=np.float32),
                        'rotation': np.array([1, 0, 0, 0], dtype=np.float32),
                        'velocity': np.array([0, 0], dtype=np.float32)
                    }
                ]

        return boxes3d
