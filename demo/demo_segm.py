import sys
import time
import cv2
import os
import torch
import argparse
from method.util import cfg, load_config, Logger
from method.model.arch import build_model
from method.util import load_model_weight
from method.data.transform import Pipeline
from method.data.dataset import build_dataset
from method.data.collate import collate_function
from method.data.dataset.nuscenes import NuScenesDataset
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--viz_train', type=bool, help='vis train or val dataset',
                        default=False)
    parser.add_argument('--cfg_file', help='model config file path',
                        type=str,
                        default='/Users/lvanyang/ADAS/ADMultiTaskPerception/config/lift_splat_shoot/lss_segm.yml'
                        )
    parser.add_argument('--model_file', help='ckpt file',
                        type=str,
                        # default='/Users/lvanyang/ADAS/Model/Face/20220923142716/model_last_init.onnx'
                        # default='/Users/lvanyang/Downloads/model_last.ckpt'
                        default='/Users/lvanyang/Downloads/model_best.ckpt'
                        )
    parser.add_argument('--dataset', type=str, help='dataset name',
                        default='nuscenes'
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
            'vehicle': (255, 144, 30),
            'laneline': (123, 0, 159),
        }
        confs = {
            'drivable_area': 0.5,
            'vehicle': 0.5,
            'laneline': 0.5,
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

def main():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    current_time = time.localtime()
    load_config(cfg, args.cfg_file)

    logger = Logger(-1, cfg.save_dir, use_tensorboard=False)
    predictor = Predictor(cfg, args.model_file, logger, device='mps')  # cuda:0 ; cpu; mps(macm1 gpu)

    dataset = build_dataset(cfg.data.val, "train" if args.viz_train else 'val', logger)
    cur_idx = 0
    # while cur_idx < len(dataset):
    #     contents = dataset[cur_idx]
    for contents in dataset:
        # vis_img_in_bev(dataset, contents, False)
        show1 = vis_boxes3d_in_imgs(dataset, contents, False, vis_boxes3d=False)
        sh, sw, _ = show1.shape

        gt_meta = collate_function([contents])
        outs = predictor.inference(gt_meta)
        timestamp = gt_meta['ann_info']['timestamp'].item()
        if has_segm(outs):
            gt = predictor.visualize_vehicle_segm_gt(gt_meta, waitKey=False)
            out = predictor.visualize_vehicle_segm(outs, waitKey=False)

            gt_pred = np.concatenate([gt, out], axis=0)
            h, w, _ = gt_pred.shape
            scale = sh / h
            gt_pred = cv2.resize(gt_pred, dsize=None, fx=scale, fy=scale)
            show = np.concatenate([show1, gt_pred], axis=1)
            cv2.imshow('concat', show)
            # cv2.waitKey(0)
            # save_root = '/Volumes/KINGSTON0/visualization/bev_segm'
            # save_file = os.path.join(save_root, "%d.jpg" % timestamp)
            # cv2.imwrite(save_file, show)

        key = cv2.waitKey(0)

        # if key == ord('.'):
        #     cur_idx += 1
        # elif key == ord(','):
        #     cur_idx -= 1



if __name__ == '__main__':
    args = parse_args()
    main()
