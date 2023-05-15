import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tools.nuscene.data_check import is_target_cls
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
import cv2
from copy import deepcopy
from collections import OrderedDict

class LSSSegmLoss(nn.Module):
    r"""
    """

    def __init__(self, segm_size, bx, dx, targets, maps, pos_weight=2.13):
        super(LSSSegmLoss, self).__init__()
        self.segm_size = segm_size
        self.targets = targets
        self.moving_obstacle_targets = self.get_moving_obstacle_targets(targets)
        self.map_targets = self.get_map_targets(targets)
        self.maps = maps
        self.map_dict = {}
        for target, map in zip(self.targets, self.maps):
            self.map_dict[target] = map
        self.class_names = []
        for map in self.maps:
            if map not in self.class_names:
                self.class_names.append(map)   # append by maps' order
        self.n_classes = len(self.class_names)
        self.bx = bx.copy()
        self.dx = dx.copy()
        self.n_cls = len(self.targets)
        self.cls_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def get_moving_obstacle_targets(self, targets):
        moving_obstacle_types = ['vehicle', 'human']
        m_targets = []
        for m_type in moving_obstacle_types:
            if m_type in targets:
                m_targets.append(m_type)
        return m_targets

    def is_ploy_element(self, target):
        ploy_types = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                     'walkway', 'stop_line', 'carpark_area', 'traffic_light']
        return target in ploy_types

    def is_line_element(self, target):
        line_types = ['road_divider', 'lane_divider']
        return target in line_types

    def get_map_targets(self, targets):
        map_types = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                     'walkway', 'stop_line', 'carpark_area', 'traffic_light',
                     'road_divider', 'lane_divider']
        m_targets = []
        for map_type in map_types:
            if map_type in targets:
                m_targets.append(map_type)
        return m_targets


    def generate_gt(self, gt_meta, device):
        gts_all = {}
        for target in self.moving_obstacle_targets:
            gts = []
            boxes3d = gt_meta['boxes3d']
            for batch_boxes in boxes3d:
                img = np.zeros((self.segm_size[0], self.segm_size[1]))
                for box in batch_boxes:
                    if not is_target_cls(box['category_name'], [target]):
                        continue

                    if box['num_pts'] == 0:
                        # 没有lidar radar点
                        continue

                    box = Box(box['translation'], box['size'], Quaternion(box['rotation']))

                    pts = box.bottom_corners()[:2].T
                    pts = np.round(
                        (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    cv2.fillPoly(img, [pts], 1.0)
                gts.append(img)
            gts = torch.from_numpy(np.array(gts, dtype=np.float32)).to(device)
            gts_all[target] = gts


        for target in self.map_targets:
            # if target not in ['drivable_area']:
            #     continue
            gts = []
            map_elemts = gt_meta[target]
            for batch in map_elemts:
                img = np.zeros((self.segm_size[0], self.segm_size[1]))
                for each_elem in batch:
                    pts = np.round(
                        (each_elem - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
                    ).astype(np.int32)
                    pts[:, [1, 0]] = pts[:, [0, 1]]
                    if self.is_ploy_element(target):
                        cv2.fillPoly(img, [pts], 1.0)
                    elif self.is_line_element(target):
                        for i in range(len(pts) - 1):
                            pt1 = int(pts[i][0]), int(pts[i][1])
                            pt2 = int(pts[i + 1][0]), int(pts[i + 1][1])
                            img = cv2.line(img, pt1=pt1, pt2=pt2, color=(1.0), thickness=1)
                    else:
                        raise NotImplementedError
                gts.append(img)
                # cv2.imshow('map_elemt_%s' % (target), img)
                # cv2.waitKey(0)
            gts = torch.from_numpy(np.array(gts, dtype=np.float32)).to(device)
            gts_all[target] = gts

        gt_fusion = {}
        for target, gt in gts_all.items():
            if self.map_dict[target] not in gt_fusion:
                gt_fusion[self.map_dict[target]] = gt
            else:
                gt_fusion[self.map_dict[target]] += gt

        # for debug
        # for elemt_type, gts in gt_fusion.items():
        #     img = gts[0].cpu().numpy()
        #     cv2.imshow('map_elemt_%s' % (elemt_type), img)
        # cv2.waitKey(0)
        # for debug

        gts = []
        for cls_name in self.class_names:
            gts.append(gt_fusion[cls_name])

        gts = torch.stack(gts, dim=1)
        return gts

    def forward(self, pred, gt):
        loss = self.cls_loss(pred, gt)
        return loss
