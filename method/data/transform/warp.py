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

import math
import random
from typing import Dict, Optional, Tuple
import torch
import cv2
import numpy as np
from PIL import Image


def get_flip_matrix(prob=0.5):
    F = np.eye(3)
    if random.random() < prob:
        F[0, 0] = -1
    return F


def get_perspective_matrix(perspective=0.0):
    """

    :param perspective:
    :return:
    """
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective)  # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective)  # y perspective (about x)
    return P


def get_rotation_matrix(degree=0.0):
    """

    :param degree:
    :return:
    """
    R = np.eye(3)
    a = random.uniform(-degree, degree)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=1)
    return R


def get_scale_matrix(ratio=(1, 1)):
    """

    :param ratio:
    """
    Scl = np.eye(3)
    scale = random.uniform(*ratio)
    Scl[0, 0] *= scale
    Scl[1, 1] *= scale
    return Scl


def get_stretch_matrix(width_ratio=(1, 1), height_ratio=(1, 1)):
    """

    :param width_ratio:
    :param height_ratio:
    """
    Str = np.eye(3)
    Str[0, 0] *= random.uniform(*width_ratio)
    Str[1, 1] *= random.uniform(*height_ratio)
    return Str


def get_shear_matrix(degree):
    """

    :param degree:
    :return:
    """
    Sh = np.eye(3)
    Sh[0, 1] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # x shear (deg)
    Sh[1, 0] = math.tan(
        random.uniform(-degree, degree) * math.pi / 180
    )  # y shear (deg)
    return Sh


def get_translate_matrix(translate, width, height):
    """

    :param translate:
    :return:
    """
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * width  # x translation
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * height  # y translation
    return T

def get_translate_matrix_v2(translate_x, translate_y, width, height):
    """

    :param translate:
    :return:
    """
    T = np.eye(3)
    T[0, 2] = translate_x * width  # x translation
    T[1, 2] = translate_y * height  # y translation
    return T


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs


def warp_and_resize(
    meta: Dict,
    warp_kwargs: Dict,
    dst_shape: Tuple[int, int],
    keep_ratio: bool = True,
):
    # TODO: background, type
    raw_img = meta["img"]
    height = raw_img.shape[0]  # shape(h,w,c)
    width = raw_img.shape[1]

    # center
    C = np.eye(3)
    C[0, 2] = -width / 2
    C[1, 2] = -height / 2

    # do not change the order of mat mul
    if "perspective" in warp_kwargs and random.randint(0, 1):
        P = get_perspective_matrix(warp_kwargs["perspective"])
        C = P @ C
    if "scale" in warp_kwargs and random.randint(0, 1):
        Scl = get_scale_matrix(warp_kwargs["scale"])
        C = Scl @ C
    if "stretch" in warp_kwargs and random.randint(0, 1):
        Str = get_stretch_matrix(*warp_kwargs["stretch"])
        C = Str @ C
    if "rotation" in warp_kwargs and random.randint(0, 1):
        R = get_rotation_matrix(warp_kwargs["rotation"])
        C = R @ C
    if "shear" in warp_kwargs and random.randint(0, 1):
        Sh = get_shear_matrix(warp_kwargs["shear"])
        C = Sh @ C
    if "flip" in warp_kwargs:
        F = get_flip_matrix(warp_kwargs["flip"])
        C = F @ C
    if "translate" in warp_kwargs and random.randint(0, 1):
        if isinstance(warp_kwargs['translate'], list):
            translate_x = random.uniform(0.5 + warp_kwargs['translate'][0][0], 0.5 + warp_kwargs['translate'][0][1])
            translate_y = random.uniform(0.5 + warp_kwargs['translate'][1][0], 0.5 + warp_kwargs['translate'][1][1])
            T = get_translate_matrix_v2(translate_x, translate_y, width, height, )
        else:
            T = get_translate_matrix(warp_kwargs["translate"], width, height)
    else:
        T = get_translate_matrix(0, width, height)
    M = T @ C
    # M = T @ Sh @ R @ Str @ P @ C
    ResizeM = get_resize_matrix((width, height), dst_shape, keep_ratio)
    M = ResizeM @ M
    img = cv2.warpPerspective(raw_img, M, dsize=tuple(dst_shape))
    meta["img"] = img
    meta["warp_matrix"] = M
    if "gt_bboxes" in meta:
        boxes = meta["gt_bboxes"]
        meta["gt_bboxes"] = warp_boxes(boxes, M, dst_shape[0], dst_shape[1])
    if "gt_masks" in meta:
        for i, mask in enumerate(meta["gt_masks"]):
            meta["gt_masks"][i] = cv2.warpPerspective(mask, M, dsize=tuple(dst_shape))

    # TODO: keypoints
    # if 'gt_keypoints' in meta:

    return meta


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def warp_keypoints(keypoints, M, width, height):
    """超出图像外的kp将被裁剪到图像内"""
    n = len(keypoints)
    if n:
        # warp points
        xy = np.ones((n, 3), dtype=np.float32)
        xy[:, :2] = keypoints  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 2)  # rescale

        good_mask0 = np.logical_and(xy[:, 0] < width, xy[:, 1] < height)
        good_mask1 = np.logical_and(xy[:, 0] > 0, xy[:, 1] > 0)
        xy[:, 0] = xy[:, 0].clip(0, width)
        xy[:, 1] = xy[:, 1].clip(0, height)
        return np.logical_and(good_mask0, good_mask1), xy.astype(np.float32)


def warp_keypoints_v2(keypoints, M, width, height):
    """图像外的kp是不合理的，返回mask"""
    n = len(keypoints)
    if n:
        # warp points
        xy = np.ones((n, 3), dtype=np.float32)
        xy[:, :2] = keypoints  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 2)  # rescale

        xy[:, 0] = xy[:, 0]
        xy[:, 1] = xy[:, 1]
        good_mask0 = np.logical_and(xy[:, 0] < width, xy[:, 1] < height)
        good_mask1 = np.logical_and(xy[:, 0] > 0, xy[:, 1] > 0)
        return np.logical_and(good_mask0, good_mask1), xy.astype(np.float32)


def get_minimum_dst_shape(
    src_shape: Tuple[int, int],
    dst_shape: Tuple[int, int],
    divisible: Optional[int] = None,
) -> Tuple[int, int]:
    """Calculate minimum dst shape"""
    src_w, src_h = src_shape
    dst_w, dst_h = dst_shape

    if src_w / src_h < dst_w / dst_h:
        ratio = dst_h / src_h
    else:
        ratio = dst_w / src_w

    dst_w = int(ratio * src_w)
    dst_h = int(ratio * src_h)

    if divisible and divisible > 0:
        dst_w = max(divisible, int((dst_w + divisible - 1) // divisible * divisible))
        dst_h = max(divisible, int((dst_h + divisible - 1) // divisible * divisible))
    return dst_w, dst_h


class ShapeTransform:
    """Shape transforms including resize, random perspective, random scale,
    random stretch, random rotation, random shear, random translate,
    and random flip.

    Args:
        keep_ratio: Whether to keep aspect ratio of the image.
        divisible: Make image height and width is divisible by a number.
        perspective: Random perspective factor.
        scale: Random scale ratio.
        stretch: Width and height stretch ratio range.
        rotation: Random rotate degree.
        shear: Random shear degree.
        translate: Random translate ratio.
        flip: Random flip probability.
    """

    def __init__(
        self,
        keep_ratio: bool,
        divisible: int = 0,
        perspective: float = 0.0,
        scale: Tuple[int, int] = (1, 1),
        stretch: Tuple = ((1, 1), (1, 1)),
        rotation: float = 0.0,
        shear: float = 0.0,
        translate: float = 0.0,
        flip: float = 0.0,
        **kwargs
    ):
        self.keep_ratio = keep_ratio
        self.divisible = divisible
        self.perspective = perspective
        self.scale_ratio = scale
        self.stretch_ratio = stretch
        self.rotation_degree = rotation
        self.shear_degree = shear
        self.flip_prob = flip
        self.translate_ratio = translate

    def __call__(self, meta_data, dst_shape):
        # meta_data["warp_imgs"] = {}
        for camera, img in meta_data["imgs"].items():
            raw_img = img
            height = raw_img.shape[0]  # shape(h,w,c)
            width = raw_img.shape[1]

            # center
            C = np.eye(3)
            C[0, 2] = -width / 2
            C[1, 2] = -height / 2

            P = get_perspective_matrix(self.perspective)
            C = P @ C

            Scl = get_scale_matrix(self.scale_ratio)
            C = Scl @ C

            Str = get_stretch_matrix(*self.stretch_ratio)
            C = Str @ C

            R = get_rotation_matrix(self.rotation_degree)
            C = R @ C

            Sh = get_shear_matrix(self.shear_degree)
            C = Sh @ C

            F = get_flip_matrix(self.flip_prob)
            C = F @ C

            if isinstance(self.translate_ratio, list):
                translate_x = random.uniform(0.5 + self.translate_ratio[0][0], 0.5 + self.translate_ratio[0][1])
                translate_y = random.uniform(0.5 + self.translate_ratio[1][0], 0.5 + self.translate_ratio[1][1])
                T = get_translate_matrix_v2(translate_x, translate_y, width, height, )
            else:
                T = get_translate_matrix(self.translate_ratio, width, height)
            M = T @ C
            post_matrix = M.copy()

            # if self.keep_ratio:
            #     dst_shape = get_minimum_dst_shape(
            #         (width, height), dst_shape, self.divisible
            #     )

            ResizeM = get_resize_matrix((width, height), dst_shape, self.keep_ratio)
            M = ResizeM @ M
            img = cv2.warpPerspective(raw_img, M, dsize=tuple(dst_shape))
            # meta_data["warp_imgs"][camera] = img.copy()
            meta_data["imgs"][camera] = img

            if "warp_matrix" not in meta_data:
                meta_data["warp_matrix"] = {}
            if "resize_matrix" not in meta_data:
                meta_data["resize_matrix"] = {}
            if "post_matrix" not in meta_data:
                meta_data["post_matrix"] = {}

            meta_data["warp_matrix"][camera] = M
            meta_data["resize_matrix"][camera] = ResizeM
            meta_data["post_matrix"][camera] = post_matrix

            # if "gt_bboxes" in meta_data:
            #     boxes = meta_data["gt_bboxes"]
            #     meta_data["gt_bboxes"] = warp_boxes(boxes, M, dst_shape[0], dst_shape[1])
            # if "gt_masks" in meta_data:
            #     for i, mask in enumerate(meta_data["gt_masks"]):
            #         meta_data["gt_masks"][i] = cv2.warpPerspective(
            #             mask, M, dsize=tuple(dst_shape)
            #         )

        return meta_data

class LssBasedTransform():
    def __init__(self):
        pass

    @staticmethod
    def get_rot(h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    @staticmethod
    def img_transform(img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = LssBasedTransform.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    @staticmethod
    def sample_augmentation(is_train, raw_shape, dst_shape, pipeline_cfg):
        H, W = raw_shape[1], raw_shape[0]
        fH, fW = dst_shape[1], dst_shape[0]
        if is_train:
            resize = np.random.uniform(*pipeline_cfg['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*pipeline_cfg['bot_pct_lim'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if pipeline_cfg['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*pipeline_cfg['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(pipeline_cfg['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    @staticmethod
    def depth_transform(cam_depth, resize, resize_dims, crop, flip, rotate):
        """Transform depth based on ida augmentation configuration.

        Args:
            cam_depth (np array): Nx3, 3: x,y,d.
            resize (float): Resize factor.
            resize_dims (list): Final dimension.
            crop (list): x1, y1, x2, y2
            flip (bool): Whether to flip.
            rotate (float): Rotation value.

        Returns:
            np array: [h/down_ratio, w/down_ratio, d]
        """

        H, W = resize_dims
        cam_depth[:, :2] = cam_depth[:, :2] * resize
        cam_depth[:, 0] -= crop[0]
        cam_depth[:, 1] -= crop[1]
        if flip:
            cam_depth[:, 0] = resize_dims[1] - cam_depth[:, 0]

        cam_depth[:, 0] -= W / 2.0
        cam_depth[:, 1] -= H / 2.0

        h = rotate / 180 * np.pi
        rot_matrix = [
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ]
        cam_depth[:, :2] = np.matmul(rot_matrix, cam_depth[:, :2].T).T

        cam_depth[:, 0] += W / 2.0
        cam_depth[:, 1] += H / 2.0

        depth_coords = cam_depth[:, :2].astype(np.int16)

        depth_map = np.zeros(resize_dims)
        valid_mask = ((depth_coords[:, 1] < resize_dims[0])
                      & (depth_coords[:, 0] < resize_dims[1])
                      & (depth_coords[:, 1] >= 0)
                      & (depth_coords[:, 0] >= 0))
        depth_map[depth_coords[valid_mask, 1],
                  depth_coords[valid_mask, 0]] = cam_depth[valid_mask, 2]

        # 这里返回的是稀疏的深度
        return torch.Tensor(depth_map)

class BevTransform():
    def __init__(self):
        pass

    @staticmethod
    def sample_augmentation(is_train, pipeline_cfg):
        """Generate bda augmentation values based on bda_config."""
        if is_train:
            rotate_bda = np.random.uniform(*pipeline_cfg['rot_lim'])
            scale_bda = np.random.uniform(*pipeline_cfg['scale_lim'])
            flip_dx = np.random.uniform() < pipeline_cfg['flip_dx_ratio']
            flip_dy = np.random.uniform() < pipeline_cfg['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            flip_dx = False
            flip_dy = False
        return rotate_bda, scale_bda, flip_dx, flip_dy

    @staticmethod
    def transform_matrix(rotate_angle, scale_ratio, flip_dx, flip_dy):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        return rot_mat

    @staticmethod
    def transform_boxes3d(rotate_angle, scale_ratio, flip_dx, flip_dy, boxes3d):
        """
        boxes3d: np.array
        """
        rot_mat = BevTransform.transform_matrix(rotate_angle, scale_ratio, flip_dx, flip_dy)
        if boxes3d.shape[0] > 0:
            boxes3d[:, :3] = (rot_mat @ boxes3d[:, :3].unsqueeze(-1)).squeeze(-1)
            boxes3d[:, 3:6] *= scale_ratio
            boxes3d[:, 6] += (rotate_angle / 180 * np.pi)
            if flip_dx:
                boxes3d[:, 6] = 2 * torch.asin(torch.tensor(1.0)) - boxes3d[:, 6]
            if flip_dy:
                boxes3d[:, 6] = -boxes3d[:, 6]
            if boxes3d.shape[1] > 7:  # velocity
                boxes3d[:, 7:] = (
                        rot_mat[:2, :2] @ boxes3d[:, 7:].unsqueeze(-1)).squeeze(-1)
        return boxes3d

    def transfrom_map_elements(self, rotate_angle, scale_ratio, flip_dx, flip_dy, elements):
        raise NotImplementedError