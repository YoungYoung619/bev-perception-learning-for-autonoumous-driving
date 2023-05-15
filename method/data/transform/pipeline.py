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

import functools
import warnings
from typing import Dict, Tuple
from copy import deepcopy
import torchvision
import cv2
from torch.utils.data import Dataset
import numpy as np

from .warp import (
    ShapeTransform,
    warp_and_resize,
    LssBasedTransform,
    BevTransform,
)
from .color import (
    color_aug_and_norm,
    norm,
    GaussianBlur,
    ColorTransform
)
import functools
import torchvision.transforms as transforms
import torch


class LegacyPipeline:
    def __init__(self, cfg, keep_ratio):
        warnings.warn(
            "Deprecated warning! Pipeline from method v0.x has been deprecated,"
            "Please use new Pipeline and update your config!"
        )
        self.warp = functools.partial(
            warp_and_resize, warp_kwargs=cfg, keep_ratio=keep_ratio
        )
        self.color = functools.partial(color_aug_and_norm, kwargs=cfg)

    def __call__(self, meta, dst_shape):
        meta = self.warp(meta, dst_shape=dst_shape)
        meta = self.color(meta=meta)
        return meta


class Pipeline:
    """Data process pipeline. Apply augmentation and pre-processing on
    meta_data from dataset.

    Args:
        cfg (Dict): Data pipeline config.
        keep_ratio (bool): Whether to keep aspect ratio when resizing image.

    """

    def __init__(self, dataset: str, input_size: tuple, cfg: Dict, keep_ratio: bool = True):
        self.type = cfg.type if 'type' in cfg else 'normal'
        self.cfg = cfg
        self.dst_shape = input_size
        self.dataset = dataset
        assert self.type in ['normal', 'lss_based']
        if self.type in ['normal']:
            self.shape_transform = ShapeTransform(keep_ratio, **cfg)
            self.norm = functools.partial(norm, kwargs=cfg)
            self.color_transform = ColorTransform(**cfg)
        elif self.type in ['lss_based']:
            self.transform = None
            mean = np.array(self.cfg.normalize[0][::-1], dtype=np.float32) / 255.
            std = np.array(self.cfg.normalize[1][::-1], dtype=np.float32) / 255.
            self.normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=mean.tolist(),
                                                 std=std.tolist()),
            ))
        else:
            raise NotImplementedError

    def normal_aug_callback(self, meta: Dict, dst_shape: Tuple[int, int]):
        meta = self.shape_transform(meta, dst_shape=dst_shape)
        meta = self.color_transform(meta)
        meta["warp_imgs"] = {}
        for camera_type, img in meta['imgs'].items():
            img = np.clip(img, 0., 255.)
            meta["warp_imgs"][camera_type] = img.astype(np.uint8).copy()

            # for debug
            # show = img.astype(np.uint8)
            # cv2.imshow('test', show)
            # cv2.waitKey(0)

            img = self.norm(img)
            meta['imgs'][camera_type] = img
        return meta

    def lss_based_aug_callback(self, meta: Dict, dst_shape):
        assert 'train' in self.cfg
        if self.dataset == 'NuScenesDataset':
            raw_shape = (1600, 900)  # w h
        else:
            raise NotImplementedError

        meta["warp_imgs"] = {}
        meta["warp_matrix"] = {}
        meta["post_tran"] = {}
        meta["post_rot"] = {}
        img_aug = self.cfg.img_aug if 'img_aug' in self.cfg else self.cfg
        bev_aug = self.cfg.bev_aug if 'bev_aug' in self.cfg else None
        for camera_type, camera_imgs in meta['imgs'].items():
            nsweeps = len(camera_imgs) if isinstance(camera_imgs, list) else 1
            resize, resize_dims, crop, flip, rotate = LssBasedTransform.sample_augmentation(self.cfg.train,
                                                                                            raw_shape,
                                                                                            dst_shape,
                                                                                            img_aug)
            if 'depth_imgs' in meta:
                depth_img = meta['depth_imgs'][camera_type]
                meta['depth_imgs'][camera_type] = LssBasedTransform.depth_transform(depth_img,
                                                                                    resize,
                                                                                    dst_shape[::-1],
                                                                                    crop,
                                                                                    flip,
                                                                                    rotate)

            camera_imgs = [camera_imgs] if not isinstance(camera_imgs, list) else camera_imgs
            for i, camera_img in enumerate(camera_imgs):   # nsweeps, first is key frame
                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)
                img, post_rot2, post_tran2 = LssBasedTransform.img_transform(camera_img, post_rot, post_tran,
                                                                             resize=resize,
                                                                             resize_dims=resize_dims,
                                                                             crop=crop,
                                                                             flip=flip,
                                                                             rotate=rotate)

                img_tensor = torchvision.transforms.ToTensor()(img).permute(1, 2, 0)
                warp_img = (img_tensor.numpy() * 255).astype(np.uint8)
                if camera_type not in meta['warp_imgs'] and nsweeps > 1:
                    meta['warp_imgs'][camera_type] = []

                if nsweeps == 1:
                    meta['warp_imgs'][camera_type] = warp_img
                else:
                    meta['warp_imgs'][camera_type].append(warp_img)

                # for convenience, make augmentation matrices 3x3
                warp_matrix = torch.eye(3)
                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2
                warp_matrix[:2, 2] = post_tran2
                warp_matrix[:2, :2] = post_rot2

                if camera_type not in meta['post_tran'] and nsweeps > 1:
                    meta['post_tran'][camera_type] = []
                if nsweeps > 1:
                    meta['post_tran'][camera_type].append(post_tran.numpy())
                else:
                    meta['post_tran'][camera_type] = post_tran.numpy()

                if camera_type not in meta['post_rot'] and nsweeps > 1:
                    meta['post_rot'][camera_type] = []
                if nsweeps > 1:
                    meta['post_rot'][camera_type].append(post_rot.numpy())
                else:
                    meta['post_rot'][camera_type] = post_rot.numpy()

                if camera_type not in meta['warp_matrix'] and nsweeps > 1:
                    meta['warp_matrix'][camera_type] = []
                if nsweeps > 1:
                    meta['warp_matrix'][camera_type].append(warp_matrix.numpy())
                else:
                    meta['warp_matrix'][camera_type] = warp_matrix.numpy()

                img = self.normalize_img(img)
                if nsweeps > 1:
                    meta['imgs'][camera_type][i] = img.numpy().transpose(1, 2, 0)
                else:
                    meta['imgs'][camera_type] = img.numpy().transpose(1, 2, 0)

        if bev_aug:
            # 这里只是产生bev aug的参数，实际的增强动作发生在检测头的generate_gt和bev generator的bev特征转换中
            bev_aug_args = BevTransform.sample_augmentation(self.cfg.train, bev_aug)
            meta['bev_aug_args'] = bev_aug_args

        # debug
        # for camera_type, img in meta['warp_imgs'].items():
        #     show = img.copy()
        #     if 'depth_imgs' in meta:
        #         ys, xs = torch.where(meta['depth_imgs'][camera_type]!=0)
        #         pts = meta['depth_imgs'][camera_type][ys, xs]
        #
        #         for y, x, d in zip(ys, xs, pts):
        #             dis_split = [0, 8, 22, 45]
        #             x = int(x.item())
        #             y = int(y.item())
        #             depth = float(d.item())
        #             if depth >= dis_split[0] and depth <= dis_split[1]:
        #                 color = (int(depth / (dis_split[1] - dis_split[0]) * 255), 0, 0)
        #             elif depth > dis_split[1] and depth <= dis_split[2]:
        #                 color = (0, int((depth - dis_split[1]) / (dis_split[2] - dis_split[1]) * 255), 0)
        #             elif depth > dis_split[2] and depth < dis_split[3]:
        #                 color = (0, 0, int((depth - dis_split[2]) / (dis_split[3] - dis_split[2]) * 255))
        #             else:
        #                 color = (0, 0, 255)
        #             show = cv2.circle(show, (x, y), radius=1, thickness=-1, color=color)
        #     cv2.imshow(camera_type, show)
        # cv2.waitKey(0)
        return meta

    def __call__(self, meta: Dict, dst_shape: Tuple[int, int]):
        if self.type in ['normal']:
            return self.normal_aug_callback(meta, dst_shape)
        elif self.type in ['lss_based']:
            return self.lss_based_aug_callback(meta, dst_shape)
        else:
            raise NotImplementedError
