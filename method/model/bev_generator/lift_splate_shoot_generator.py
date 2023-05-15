import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.resnet import resnet18
from method.data.transform.warp import BevTransform

from ..module.conv import ConvModule
from ..module.init_weights import normal_init
from method.util.mps_tools import (
    adaptive_mps_inverse,
    adaptive_mps_matmul,
    adaptive_mps_argsort,
    adaptive_mps_cumsum
)

from .lift_splate_shoot.utils import (
    gen_dx_bx,
    cumsum_trick,
    QuickCumsum,
    BevEncode,
    SimpleConvEncoder
)
from torch.cuda.amp import autocast

class LiftSplatShootBEVGenerator(nn.Module):
    """the bev features generator used in lift splat shoot"""
    def __init__(
            self,
            target_cameras,
            input_stages,
            input_channel,
            output_channel,
            image_channel,
            depth_channel,
            input_size,
            feat_strides,
            xbound,
            ybound,
            zbound,
            dbound,
            bev_xy_transpose=True,
            **kwargs,
    ):
        super(LiftSplatShootBEVGenerator, self).__init__()
        self.image_channel = image_channel
        self.depth_channel = depth_channel
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.target_cameras = target_cameras
        self.n_camera = len(self.target_cameras)
        self.input_stages = input_stages
        self.input_size = input_size
        self.feat_strides = feat_strides
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self._build_cam_encoder(input_stages, input_channel, image_channel)
        self._build_depth_encoder(input_stages, input_channel, depth_channel)
        self.frustums = self.create_frustum(input_size, input_stages, feat_strides)
        dx, bx, nx = gen_dx_bx(self.xbound,
                               self.ybound,
                               self.zbound,
                               )
        # self.dx = nn.Parameter(dx, requires_grad=False)
        # self.bx = nn.Parameter(bx, requires_grad=False)
        # self.nx = nn.Parameter(nx, requires_grad=False)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.use_quickcumsum = True
        self.bev_xy_transpose = bev_xy_transpose

        self.bevencode = BevEncode(inC=self.image_channel, outC=self.output_channel)

        self.multi_scale_voxel_fusion = nn.Conv2d(
            self.image_channel * len(self.input_stages), self.image_channel, 1, padding=0
        )
        self.init_weights()

    def init_weights(self):
        bias_cls = -4.595
        normal_init(self.multi_scale_voxel_fusion, std=0.01, bias=bias_cls)

    def create_frustum(self, input_size, input_stages, feat_strides):
        # make grid in image plane
        frustums = []
        for input_stage_idx in input_stages:
            downsample = feat_strides[input_stage_idx]
            ogfW, ogfH = input_size
            fH, fW = ogfH // downsample, ogfW // downsample
            ds = torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
            D, _, _ = ds.shape
            xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
            ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

            # D x H x W x 3
            frustum = torch.stack((xs, ys, ds), -1)
            # frustum = nn.Parameter(frustum, requires_grad=False)
            frustums.append(frustum)

        return frustums

    def _build_cam_encoder(self, input_stages, input_channel, out_channel):
        for stage_idx in input_stages:
            setattr(self, 'cam_encoder_%d' % (stage_idx), SimpleConvEncoder(
                input_channel=input_channel,
                feat_channel=256,
                last_channel=out_channel,
            ))

    def _build_depth_encoder(self, input_stages, input_channel, out_channel):
        for stage_idx in input_stages:
            setattr(self, 'depth_encoder_%d' % (stage_idx), SimpleConvEncoder(
                input_channel=input_channel,
                feat_channel=256,
                last_channel=out_channel,
                stacked_convs=1,
                kernel_size=1,
            ))

    def _get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_geometry(self, rots, trans, intrins, post_warps, post_trans, post_rots, bev_aug_args):
        geoms = []
        B, N, _ = trans.size()
        warp_inv = adaptive_mps_inverse(post_warps)

        device = warp_inv.device
        for frustum in self.frustums:
            frustum = frustum.to(device).clone()

            if post_trans is not None and post_rots is not None:
                points = frustum - post_trans.view(B, N, 1, 1, 1, 3)
                points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))
            else:
                points = torch.ones_like(frustum, device=device)
                points[..., :2] = frustum[..., :2]
                # undo post-transformation
                points = warp_inv.view(B, N, 1, 1, 1, 3, 3) @ points.unsqueeze(-1)
                points = points[..., 0] / points[..., 2, 0].unsqueeze(-1)

                # attach depth to points undo
                points[..., 2] = frustum[..., 2]

            # aa = points[0, 0, 0, :, :, :2].squeeze().numpy()

            # debug
            # a = (points[0, 0, 0, :, :, :2, 0]).cpu().numpy()
            # b = (points[0, 0, 2, :, :, :2, 0]).cpu().numpy()
            # xs = a[..., 0]
            # ys = a[..., 1]
            # xs1 = b[..., 0]
            # ys1 = b[..., 1]

            points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                                points[:, :, :, :, :, 2:3]
                                ), 5)
            if post_trans is None and post_rots is None:
                points = points.unsqueeze(-1)
            inv = adaptive_mps_inverse(intrins)
            combine = adaptive_mps_matmul(rots, inv)
            # points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
            points = adaptive_mps_matmul(combine.view(B, N, 1, 1, 1, 3, 3), points).squeeze(-1)
            points += trans.view(B, N, 1, 1, 1, 3)

            # a = points[0, 0, 0, :, :, :]
            # b = points[0, 1, 0, :, :, :]
            # c = points[0, 2, 0, :, :, :]
            # d = points[0, 3, 0, :, :, :]
            # e = points[0, 4, 0, :, :, :]
            # f = points[0, 5, 0, :, :, :]

            if bev_aug_args is not None:
                ones = torch.ones_like(points[..., 0]).unsqueeze(dim=-1)
                points_homo = torch.cat([points, ones], dim=-1).unsqueeze(dim=-1)
                rot_mats = torch.eye(4).to(device)[None, :, :].repeat(B, 1, 1)
                rot_mats_3x3 = torch.stack([BevTransform.transform_matrix(*args) for args in bev_aug_args]).to(device)
                rot_mats[:, :3, :3] = rot_mats_3x3

                rot_mats = rot_mats.unsqueeze(1).repeat(1, N, 1, 1).view(B, N, 1, 1, 1, 4, 4)
                points = (rot_mats @ points_homo).squeeze(-1)[..., :3]

            geoms.append(points)
        return geoms

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        # sorts = ranks.argsort()
        sorts = adaptive_mps_argsort(ranks)
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        # self.use_quickcumsum = False
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        if self.bev_xy_transpose:
            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x
        else:
            # griddify (B x C x Z x Y x X)
            final = torch.zeros((B, C, self.nx[2], self.nx[1], self.nx[0]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def vis_depth(self, depth):
        import cv2
        import numpy
        depth = depth.permute(0, 2, 3, 1).to('cpu')
        depth_dist = torch.arange(4, 45, 1).float()[None, None, None, :]
        depth = (depth * depth_dist).sum(dim=-1)
        for i, depth_img in enumerate(depth):
            depth_img = depth_img / 44.
            depth_img = depth_img.numpy()[:, :, None]
            depth_img = cv2.resize(depth_img, dsize=None, fx=10, fy=10)
            cv2.imshow('img%d' % i, depth_img)
        # cv2.waitKey(0)
        pass

    @autocast(False)
    def forward(self, feats, rots, trans, intrins, warps, post_trans=None, post_rots=None, bev_aug_args=None):
        device = feats[0].device
        if self.dx.device != device:
            self.dx = self.dx.to(device)
            self.bx = self.bx.to(device)
            self.nx = self.nx.to(device)
        geoms = None
        if rots is not None:
            geoms = self.get_geometry(rots, trans, intrins, warps, post_trans, post_rots, bev_aug_args)

        # image feature
        img_feats = []
        for stage_idx in self.input_stages:
            cam_encoder = eval("self.cam_encoder_%d" % stage_idx)
            img_feat = cam_encoder(feats[stage_idx])
            img_feats.append(img_feat)

        # depth feature
        depth_feats = []
        for stage_idx in self.input_stages:
            depth_encoder = eval("self.depth_encoder_%d" % stage_idx)
            depth_feat = depth_encoder(feats[stage_idx])
            depth_feats.append(depth_feat)

        context_feats = []
        debug = False
        for i, (img_feat, depth_feat) in enumerate(zip(img_feats, depth_feats)):
            depth = self._get_depth_dist(depth_feat)

            if i == 0 and debug:
                self.vis_depth(depth)

            context_feat = depth.unsqueeze(1) * img_feat.unsqueeze(2)
            _, ic, d, h, w = context_feat.size()
            context_feat = context_feat.view(-1, self.n_camera, ic, d, h, w)
            context_feat = context_feat.permute(0, 1, 3, 4, 5, 2)
            context_feats.append(context_feat)

        if geoms is not None:
            voxel_feats = []
            for geom, context_feat in zip(geoms, context_feats):
                voxel_feat = self.voxel_pooling(geom, context_feat)
                # bs = voxel_feat.size()[0]
                # for i in range(0, bs):
                #     aa = voxel_feat[i].detach().numpy()
                #     img = aa[0]
                #     print(np.max(img), np.min(img))
                #     img = (img - np.min(img)) / (np.max(img) - np.min(img))
                #     cv2.imshow("test", img)
                #     cv2.waitKey(0)
                voxel_feats.append(voxel_feat)

            voxel_feats = torch.cat(voxel_feats, dim=1)
            voxel_feats = self.multi_scale_voxel_fusion(voxel_feats)

            bev_feats = self.bevencode(voxel_feats)
        else:
            # only for flops calculation
            if torch.onnx.is_in_onnx_export():
                raise NotImplementedError

            bs = feats[0].size()[0]
            bev_feats = torch.rand(size=(bs, self.output_channel, self.nx[0], self.nx[1]), dtype=torch.float32)
        return bev_feats
