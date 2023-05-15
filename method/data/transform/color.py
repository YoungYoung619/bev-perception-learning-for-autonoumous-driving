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

import random
import torch
import cv2
import numpy as np
from PIL import ImageFilter
import torchvision.transforms as transforms

def bgr_to_hsv(bgr):
    # Translated from source of colorsys.rgb_to_hsv
    # r,g,b should be a numpy arrays with values between 0 and 255
    # rgb_to_hsv returns an array of floats between 0.0 and 1.0.
    bgr = bgr.astype('float')
    hsv = np.zeros_like(bgr)
    # in case an RGBA array was passed, just copy the A channel
    hsv[..., 3:] = bgr[..., 3:]
    b, g, r = bgr[..., 0], bgr[..., 1], bgr[..., 2]
    maxc = np.max(bgr[..., :3], axis=-1)
    minc = np.min(bgr[..., :3], axis=-1)
    hsv[..., 2] = maxc
    mask = maxc != minc
    hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
    rc = np.zeros_like(r)
    gc = np.zeros_like(g)
    bc = np.zeros_like(b)
    rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
    gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
    bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
    hsv[..., 0] = np.select(
        [r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
    hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
    return hsv


def hsv_to_bgr(hsv):
    # Translated from source of colorsys.hsv_to_rgb
    # h,s should be a numpy arrays with values between 0.0 and 1.0
    # v should be a numpy array with values between 0.0 and 255.0
    # hsv_to_rgb returns an array of uints between 0 and 255.
    bgr = np.empty_like(hsv)
    bgr[..., 3:] = hsv[..., 3:]
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = (h * 6.0).astype('uint8')
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
    bgr[..., 2] = np.select(conditions, [v, q, p, p, t, v], default=v)
    bgr[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
    bgr[..., 0] = np.select(conditions, [v, p, t, v, v, q], default=p)
    return bgr.astype('uint8')


def hueChange(bgr, hue):
    hsv = bgr_to_hsv(bgr)
    hsv[..., 0] = hue
    bgr = hsv_to_bgr(hsv)
    return bgr


def random_brightness(img, delta):
    img += random.uniform(-delta, delta)
    return img


def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img


def normalize(meta, mean, std):
    img = meta["img"].astype(np.float32)
    mean = np.array(mean, dtype=np.float64).reshape(1, -1)
    stdinv = 1 / np.array(std, dtype=np.float64).reshape(1, -1)
    cv2.subtract(img, mean, img)
    cv2.multiply(img, stdinv, img)
    meta["img"] = img
    return meta


def _normalize(img, mean, std):
    mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
    std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
    img = (img - mean) / std
    return img


def random_gamma(img, gamma_low, gamma_up):
    gamma = random.uniform(gamma_low, gamma_up)
    return img ** gamma


def color_aug_and_norm(meta, kwargs):
    for camera, img in meta["imgs"].items():
        if 'hue' in kwargs and random.randint(0, 1):
            img = hueChange(img, random.uniform(*kwargs['hue']))

        if 'gray' in kwargs and random.uniform(0, 1) < kwargs['gray']:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if 'reverse_channel' in kwargs and random.uniform(0, 1) < kwargs['reverse_channel']:
            img = img[..., ::-1]

        img = img.astype(np.float32) / 255

        if 'gamma' in kwargs and random.randint(0, 1):
            img = random_gamma(img, *kwargs['gamma'])

        if "brightness" in kwargs and random.randint(0, 1):
            img = random_brightness(img, kwargs["brightness"])

        if "contrast" in kwargs and random.randint(0, 1):
            img = random_contrast(img, *kwargs["contrast"])

        if "saturation" in kwargs and random.randint(0, 1):
            img = random_saturation(img, *kwargs["saturation"])

        img = _normalize(img, *kwargs["normalize"])
        meta["imgs"][camera] = img
    return meta


def norm(img, kwargs):
    img = img.astype(np.float32) / 255
    img = _normalize(img, *kwargs['normalize'])
    return img


class GaussianBlur:
    """
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
    Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    Adapted from MoCo:
    https://github.com/facebookresearch/moco/blob/master/moco/loader.py
    Note that this implementation does not seem to be exactly the same as
    described in SimCLR.
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class ColorTransform:
    def __init__(self, color_transform=False, **kwargs):
        self.color_transform = color_transform
        augmentation = []
        if self.color_transform:
            augmentation.append(
                transforms.ToPILImage()
            )

            augmentation.append(
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.5)
            )
            augmentation.append(transforms.RandomGrayscale(p=0.2))
            augmentation.append(transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1))
            augmentation.append(transforms.ToTensor())
            randcrop_transform = transforms.Compose(
                [
                    transforms.RandomErasing(
                        p=0.3, scale=(0.01, 0.02), ratio=(0.3, 2), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.15, scale=(0.02, 0.05), ratio=(0.1, 2), value="random"
                    ),
                    transforms.RandomErasing(
                        p=0.075, scale=(0.02, 0.05), ratio=(0.05, 2), value="random"
                    ),
                ]
            )
            augmentation.append(randcrop_transform)
        self.color = transforms.Compose(augmentation)

    def __call__(self, meta_data):
        for camera_type, img in meta_data['imgs'].items():
            if self.color_transform:
                meta_data['imgs'][camera_type] = (self.color(img) * 255).permute(1, 2, 0).cpu().numpy()
            # min_val, max_val = np.min(meta_data['imgs'][camera_type]), np.max(meta_data['imgs'][camera_type])
            # print(camera_type, min_val, max_val)
        return meta_data
