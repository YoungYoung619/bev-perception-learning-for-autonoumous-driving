import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from torchvision.models.resnet import resnet18

from ..loss.segm_loss import LSSSegmLoss
from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from method.util.mps_tools import (
    adaptive_mps_inverse,
    adaptive_mps_matmul,
    adaptive_mps_argsort,
    adaptive_mps_cumsum
)
from ..bev_generator.lift_splate_shoot_generator import gen_dx_bx
from ..bev_generator.lift_splate_shoot.utils import SimpleConvEncoder
from torch.cuda.amp import autocast

class BEVSegmHead(nn.Module):
    """the detection head used in lift splat shoot"""

    def __init__(
            self,
            input_channel,
            target_names,
            target_maps,
            xbound,
            ybound,
            zbound,
            dbound,
            loss,
            **kwargs,
    ):
        super(BEVSegmHead, self).__init__()

        self.dx, self.bx, self.nx = gen_dx_bx(xbound,
                                              ybound,
                                              zbound,
                                              )
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.use_quickcumsum = True
        self.target_names = target_names
        self.target_maps = target_maps
        self.class_names = []
        for map in self.target_maps:
            if map not in self.class_names:
                self.class_names.append(map)  # append by maps' order
        self.n_classes = len(self.class_names)

        self.loss_func = eval(loss.name)(segm_size=self.nx[:2],
                                         targets=self.target_names,
                                         maps=self.target_maps,
                                         bx=self.bx.cpu().numpy(),
                                         dx=self.dx.cpu().numpy())
        self.segm_conv = SimpleConvEncoder(input_channel,
                                           input_channel,
                                           self.n_classes
                                           )
        self.with_depth_supervised = True if 'depth_supervised' in loss else False
        if self.with_depth_supervised:
            self.depth_channel = loss.depth_supervised.depth_channel
            self.downsample_factor = loss.depth_supervised.downsample_factor

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

    @autocast(False)
    def forward(self, bev_feat):
        out = self.segm_conv(bev_feat if not isinstance(bev_feat, dict) else bev_feat['bev_feats'])
        return out,  None if not self.with_depth_supervised else bev_feat['depths']

    def post_process(self, preds, meta, warp=True):
        preds, _ = preds
        preds = torch.sigmoid(preds)
        # preds = (preds > 0.2).float()
        return preds.cpu().numpy()

    def generate_gt(self, gt_meta, device):
        return self.loss_func.generate_gt(gt_meta, device)

    def calc_loss(self, preds, gt_meta):
        preds, depth_preds = preds
        segm_gt = self.generate_gt(gt_meta, preds.device)

        # for gt in segm_gt:
        #     img = gt.numpy()
        #     # cv2.imshow('gt', img)
        #     # cv2.waitKey(0)
        #     cv2.imwrite('/Users/lvanyang/ADAS/ADMultiTaskPerception/test.jpg', (img*255).astype(np.uint8))
        loss_states = {}
        total_loss = 0
        for i, cls_name in enumerate(self.class_names):
            loss = self.loss_func(preds[:, i, :, :], segm_gt[:, i, :, :])
            loss_states['%s_loss' % cls_name] = loss
            factor = 0.2 if cls_name == 'drivable_area' else 1.0
            total_loss += factor * loss

        if self.with_depth_supervised:
            camera_orders = gt_meta['camera_types'][0]
            depth_gts = [gt_meta['depth_imgs'][cam] for cam in camera_orders]
            depth_gts = torch.stack(depth_gts, dim=1)
            depth_loss = self.get_depth_loss(depth_gts, depth_preds)
            total_loss += depth_loss
            loss_states.update({'depth_loss': depth_loss})

        return total_loss, loss_states
