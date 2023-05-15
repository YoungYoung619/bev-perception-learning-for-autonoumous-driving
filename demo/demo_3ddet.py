import sys
import time
import cv2
import os
import torch
import math
import argparse
from method.util import cfg, load_config, Logger
from method.model.arch import build_model
from method.util import load_model_weight
from method.data.transform import Pipeline
from method.data.dataset import build_dataset
from method.data.collate import collate_function
from method.model.bev_generator.lift_splate_shoot_generator import gen_dx_bx
from method.data.dataset.nuscenes import NuScenesDataset
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from scipy.spatial.transform import Rotation as R

from tools.nuscene.data_check import (
    is_box_in_image,
    concat_camera_imgs,
    show_img,
    corners_3d_box,
    draw_corners
)
from method.memonger import SublinearSequential


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', type=str, default='image', help='demo type, eg. image, video and webcam')
    parser.add_argument('--viz_train', type=bool, help='vis train or val dataset',
                        default=False)
    parser.add_argument('--cfg_file', help='model config file path',
                        type=str,
                        default='/Users/lvanyang/ADAS/bev-perception-learning-for-autonoumous-driving/config/bevdepth/centerhead_3ddet.yml'
                        )
    parser.add_argument('--model_file', help='ckpt file',
                        type=str,
                        default='/Users/lvanyang/Downloads/bevdepth_3ddet.ckpt'
                        )
    parser.add_argument('--device', type=str, help='inference device name (e.g., cpu, cuda, mps)',
                        default='cpu'
                        )
    args = parser.parse_args()
    return args


class Predictor(object):
    def __init__(self, cfg, model_path, logger, device):
        self.cfg = cfg
        self.device = device
        model = build_model(cfg)
        ckpt = torch.load(model_path, map_location=lambda storage, loc: storage)
        load_model_weight(model, ckpt, logger)
        self.model = model.to(device).eval()
        self.pipeline = Pipeline(cfg.data.val.name,
                                 cfg.data.val.input_size,
                                 cfg.data.val.pipeline,
                                 cfg.data.val.keep_ratio)

        head_names = self.model.get_head_names()
        self.class_names = {}
        for head_name in head_names:
            self.class_names[head_name] = []
        for head_name in head_names:
            for target_name in cfg.model.arch.head[head_name].target_maps:
                if target_name in self.class_names[head_name]:
                    continue
                self.class_names[head_name].append(target_name)

        xbound = [-50.0, 50.0, 0.5]
        ybound = [-50.0, 50.0, 0.5]
        zbound = [-10.0, 10.0, 20.0]
        dx, bx, nx = gen_dx_bx(xbound, ybound, zbound)
        self.dx = dx.cpu().numpy()
        self.bx = bx.cpu().numpy()

    def inference(self, gt_meta):
        outs = self.model.inference(gt_meta)
        return outs

    def combine_segm_maps(self, imgs):
        # method-1
        # colors = {
        #     'drivable_area': (int(0.31 * 255), int(0.5 * 255), int(255)),
        #     'vehicle': (232, 140, 55),
        #     'laneline': (255, 0, 159),
        # }
        # n_cls, h, w = imgs.shape
        # # maps = (imgs * 255).astype(np.uint8)
        # show_img = np.ones(shape=(h, w, 3), dtype=np.float32) * 255
        # for i, target_name in enumerate(self.class_names):
        #     map = imgs[i]
        #     map = map[:, :, None] * np.array(colors[target_name])[None, None, :]
        #     show_img = map * 0.3 + show_img * 0.7
        # show_img = show_img.astype(np.uint8)

        # method-2
        colors = {
            'drivable_area': (int(0.31 * 255), int(0.5 * 255), int(255)),
            'vehicle': (200, 100, 105),
            'laneline': (123, 0, 159),
        }
        confs = {
            'drivable_area': 0.5,
            'vehicle': 0.6,
            'laneline': 0.6,
        }
        occupations = {
            'drivable_area': 0.4,
            'vehicle': 1.0,
            'laneline': 0.8,
        }
        segm_head_names = self.get_segm_head_names()
        bs, n_cls, h, w = imgs[segm_head_names[0][:-5]].shape
        # maps = (imgs * 255).astype(np.uint8)
        show_img = np.ones(shape=(h, w, 3), dtype=np.float32) * 255
        plot_orders = ['drivable_area', 'laneline', 'vehicle']
        for order in plot_orders:
            target_head = None
            idx = -1
            for segm_head, head_classes in self.class_names.items():
                if order not in head_classes:
                    continue
                target_head = segm_head
                idx = self.class_names[segm_head].index(order)
            if target_head:
                map = imgs[target_head[:-5]][0][idx]
                mask = map > confs[order]
                oc = occupations[order]
                show_img[mask] = (1. - oc) * show_img[mask] + oc * np.array(colors[order])[None, :]

        # plot a ego car
        h, w, _ = show_img.shape
        ego_length = 3
        egeo_width = 1.3
        longitude = 50
        lateral = 50
        length = int(h / longitude * ego_length)
        width = int(w / lateral * egeo_width)
        x1, y1, x2, y2 = int(w / 2 - width / 2), int(h / 2 - length / 2), int(w / 2 + width / 2), int(
            h / 2 + length / 2)
        cv2.rectangle(show_img, (x1, y1), (x2, y2), color=(123, 212, 45), thickness=-1)

        show_img = show_img.astype(np.uint8)
        return show_img

    def get_segm_head_names(self):
        head_names = self.model.get_head_names()
        segm_heads = [head for head in head_names if 'segm_head' in head]
        return segm_heads

    def get_box3d_head_names(self):
        head_names = self.model.get_head_names()
        heads = [head for head in head_names if 'box3d' in head]
        return heads

    def get_box3d_head_types(self):
        head_names = self.model.get_head_names()
        idxes = [i for i, head in enumerate(head_names) if 'box3d' in head]
        head_types = self.model.get_head_types()
        head_types = [head_types[i] for i in idxes]
        return head_types

    def visualize_vehicle_segm_gt(self, gt_meta, waitKey=False):
        segm_heads = self.get_segm_head_names()
        gts = {}
        for head in segm_heads:
            head_func = eval('self.model.%s' % (head))
            gt = head_func.generate_gt(gt_meta, 'cpu').cpu().numpy()
            gts[head[:-5]] = gt
        show = self.combine_segm_maps(gts)
        cv2.imshow('segm_gt', show)
        if waitKey:
            cv2.waitKey(0)
        return show

    def visualize_vehicle_segm(self, outs, waitKey=False):
        out = outs
        show = self.combine_segm_maps(out)
        cv2.imshow('segm_pred', show)
        if waitKey:
            cv2.waitKey(0)
        return show

    def visualize_3dboxes_gt(self, meta, contents, win_name, waitKey=False):
        target_heads = self.get_box3d_head_names()
        head_types = self.get_box3d_head_types()
        gts = {'box3d': [[]]}
        for head_type, head in zip(head_types, target_heads):
            head_func = eval('self.model.%s' % (head))
            gt = head_func.generate_gt(meta, 'cpu')
            boxes, labels = gt['gt_boxes_3d'], gt['gt_labels_3d']
            for box, label in zip(boxes[0], labels[0]):
                if head_type == 'BevCos3DHead':
                    if head_func.yaw_format.name == 'sincos':
                        x, y, gz, w, l, h, sin_alpha, cos_alpha = box[:8]
                        yaw = math.atan2(sin_alpha, cos_alpha)
                        z = gz + h / 2.
                    elif head_func.yaw_format.name == 'radian':
                        x, y, gz, w, l, h, yaw = box[:7]
                        z = gz + h / 2.
                    else:
                        raise NotImplementedError
                elif head_type == 'BEVCenter3DHead':
                    x, y, z, l, w, h, yaw, _, _ = box
                else:
                    raise NotImplementedError
                r = R.from_euler('zyx', [yaw, 0., 0.], degrees=False)
                xr, yr, zr, w_ = r.as_quat()
                rotation = np.array([w_, xr, yr, zr], dtype=np.float32)
                size = np.array([w, l, h], dtype=np.float32)
                translation = np.array([x, y, z], dtype=np.float32)
                cls_name = head_func.class_orders[label] if hasattr(head_func, 'class_orders') \
                    else head_func.class_names[label]
                box = {
                    'class_idx': int(label),
                    'class_name': cls_name,
                    'size': size,
                    'translation': translation,
                    'rotation': rotation,
                }
                gts['box3d'][0].append(box)

        return gts, self.visualize_3dboxes(gts, contents, win_name, waitKey)

    def visualize_3dboxes(self, outs, meta, win_name, waitKey=False):
        # combine outs from different box3d heads
        if 'box3d' not in outs:
            target_heads = self.get_box3d_head_names()
            preds = {'box3d': [[]]}
            for head in target_heads:
                head = head[:-5]
                head_out = outs[head][0]
                preds['box3d'][0].extend(head_out)
        else:
            preds = outs

        img_key = 'warp_imgs'
        warp_matrix_key = 'warp_matrix'

        colors = {
            'car': (255, 144, 30),
            'pedestrian': (45, 200, 125),
            'barrier': (45, 125, 200),
            'bicycle': (125, 200, 45),
            'bus': (123, 80, 190),
            'construction_vehicle': (200, 23, 125),
            'motorcycle': (173, 255, 47),
            'traffic_cone': (0, 200, 100),
            'trailer': (255, 0, 255),
            'truck': (255, 215, 0),
        }
        ignore_cls = []
        draw_imgs = {}
        for camera_type, img in meta[img_key].items():
            camera_translation = np.array(meta['ann_info']['camera_calibration'][camera_type]['translation'])
            if camera_translation.ndim == 2:
                camera_translation = camera_translation[0]
            rotation = np.array(meta['ann_info']['camera_calibration'][camera_type]['rotation'])
            if rotation.ndim == 2:
                rotation = rotation[0]
            camera_quaternion = Quaternion(rotation)
            camera_rotation = camera_quaternion.rotation_matrix
            camera_intrinsic = np.array(meta['ann_info']['camera_calibration'][camera_type]['camera_intrinsic'])
            if camera_intrinsic.ndim == 3:
                camera_intrinsic = camera_intrinsic[0]
            warp_matrix = np.array(meta[warp_matrix_key][camera_type])
            if warp_matrix.ndim == 3:
                warp_matrix = warp_matrix[0]

            img = np.array(meta[img_key][camera_type])
            if img.ndim == 4:
                img = img[0]
            img = img.astype(np.uint8).copy()
            for box3d in preds['box3d'][0]:
                cls_name = box3d['class_name']
                if cls_name in ignore_cls:
                    continue
                cls_idx = box3d['class_idx']
                size = box3d['size']
                translation = box3d['translation']
                quaternion = Quaternion(box3d['rotation'])
                rotation = quaternion.rotation_matrix

                translation_ = camera_rotation.transpose() @ (translation - camera_translation)
                quaternion_ = camera_quaternion.inverse * quaternion
                rotation_ = quaternion_.rotation_matrix

                box_corners = corners_3d_box(translation_, quaternion_, size)  # 相机坐标下

                pixel_points = (warp_matrix @ camera_intrinsic @ box_corners).transpose()
                pixel_points = pixel_points[:, :2] / pixel_points[:, 2][:, None]

                h, w, _ = img.shape
                if is_box_in_image(pixel_points, box_corners.transpose(), [w, h], 'any'):
                    draw_corners(img, pixel_points.astype(np.int32), colors[cls_name], 1)

            draw_imgs[camera_type] = img

        img = concat_camera_imgs(draw_imgs)
        show_img(win_name, img.astype(np.uint8), waitKey, width=1400)
        return img.astype(np.uint8)

    def visualize_3dboxes_bev(self, outs, meta, win_name, waitKey=False):
        segm_size = (200, 200)
        img = np.ones((segm_size[0], segm_size[1], 3), dtype=np.uint8) * 255
        colors = {
            'car': (255, 144, 30),
            'pedestrian': (45, 200, 125),
            'barrier': (45, 125, 200),
            'bicycle': (125, 200, 45),
            'bus': (123, 80, 190),
            'construction_vehicle': (200, 23, 125),
            'motorcycle': (173, 255, 47),
            'traffic_cone': (0, 200, 100),
            'trailer': (255, 0, 255),
            'truck': (255, 215, 0),
        }
        ignore_cls = []

        if 'box3d' not in outs:
            target_heads = self.get_box3d_head_names()
            preds = {'box3d': [[]]}
            for head in target_heads:
                head = head[:-5]
                head_out = outs[head][0]
                preds['box3d'][0].extend(head_out)
        else:
            preds = outs

        for box3d in preds['box3d'][0]:
            cls_name = box3d['class_name']
            if cls_name in ignore_cls:
                continue
            box = Box(box3d['translation'], box3d['size'], Quaternion(box3d['rotation']))

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2] / 2.) / self.dx[:2]
            ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], colors[cls_name])

        # plot a ego car
        h, w, _ = img.shape
        ego_length = 3
        egeo_width = 1.3
        longitude = 50
        lateral = 50
        length = int(h / longitude * ego_length)
        width = int(w / lateral * egeo_width)
        x1, y1, x2, y2 = int(w / 2 - width / 2), int(h / 2 - length / 2), int(w / 2 + width / 2), int(
            h / 2 + length / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(123, 212, 45), thickness=-1)

        cv2.imshow(win_name, img)
        if waitKey:
            cv2.waitKey(0)
        return img


def vis_img_in_bev(dataset, meta, waitkey):
    dataset.vis_bev_imgs(meta, 'warp_imgs', 'warp_matrix', waitkey=waitkey)


def vis_boxes3d_in_imgs(dataset, meta, waitkey, vis_boxes3d):
    show = dataset.vis_boxes3d_in_imgs(meta, 'warp_imgs', 'warp_matrix', waitkey=waitkey, vis_boxes3d=vis_boxes3d)
    return show


def has_segm(outs):
    for segm_out_name in list(outs.keys()):
        if 'segm' in segm_out_name:
            return True
    return False


def has_box3d(outs):
    for out_name in list(outs.keys()):
        if 'box3d' in out_name:
            return True
    return False


def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    current_time = time.localtime()
    load_config(cfg, args.cfg_file)

    logger = Logger(-1, cfg.save_dir, use_tensorboard=False)
    predictor = Predictor(cfg, args.model_file, logger, device=args.device)  # cuda:0 ; cpu; mps(macm1 gpu)

    dataset = build_dataset(cfg.data.val, "train" if args.viz_train else 'val', logger)

    cur_idx = 0
    # while cur_idx < len(dataset):
    for contents in dataset:
        # contents = dataset[cur_idx]
        # vis_img_in_bev(dataset, contents, False)
        # show1 = vis_boxes3d_in_imgs(dataset, contents, False, vis_boxes3d=True)
        # sh, sw, _ = show1.shape

        gt_meta = collate_function([contents])
        outs = predictor.inference(gt_meta)
        if has_box3d(outs):
            gts, box3d_gt_img = predictor.visualize_3dboxes_gt(gt_meta, contents, win_name='box3d_gt', waitKey=False)
            box3d_pred_img = predictor.visualize_3dboxes(outs, contents, win_name='box3d_pred', waitKey=False)

            bev_pred_img = predictor.visualize_3dboxes_bev(outs, contents, win_name='bev_pred', waitKey=False)
            bev_gt_img = predictor.visualize_3dboxes_bev(gts, contents, win_name='bev_gt', waitKey=False)

            # h, w, _ = box3d_pred_img.shape
            # eh, ew, _ = bev_pred_img.shape
            # ratio = h / eh
            # bev = cv2.resize(bev_pred_img, dsize=None, fx=ratio, fy=ratio)
            # show_img = np.concatenate([box3d_pred_img, bev], axis=1)
            # show_img = box3d_pred_img
            #
            # cv2.imshow('concat', show_img)

            # save_root = '/Volumes/KINGSTON0/visualization/test2'
            # current_time = time.time()
            # save_file = os.path.join(save_root, "%d.jpg" % current_time)
            # cv2.imwrite(save_file, show_img)
            # cv2.imwrite(save_file, cv2.cvtColor(show_img, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(0)


if __name__ == '__main__':
    args = parse_args()
    main()
