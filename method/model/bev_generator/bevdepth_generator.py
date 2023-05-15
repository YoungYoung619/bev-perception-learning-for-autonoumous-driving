from functools import reduce

import torch
import torch.nn as nn

from .bevdepth.utils import DepthNet
from .lift_splate_shoot.utils import (
    gen_dx_bx,
)
from .bevdepth.utils import BevEncode, DepthAggregation
from torch.cuda.amp import autocast
from .lift_splate_shoot_generator import LiftSplatShootBEVGenerator


class BevDepthBEVGenerator(LiftSplatShootBEVGenerator):
    """the bev features generator used in bevdepth"""

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
            bev_xy_transpose,
            enable_pos_embedding,
            multi_frame=False,
            use_da=False,
            **kwargs,
    ):
        nn.Module.__init__(self)
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
        # self._build_cam_encoder(input_stages, input_channel, image_channel)
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

        self.bevencode = BevEncode(
            self.image_channel,
            dx,
            bx,
            nx,
            self.bev_xy_transpose,
            multi_frame=multi_frame,
            enable_pos_embedding=enable_pos_embedding)

        self.use_da = use_da
        if use_da:
            self.depth_aggregation_net = DepthAggregation(self.image_channel, self.image_channel, self.image_channel)

    def _build_cam_encoder(self, input_stages, input_channel, out_channel):
        raise NotImplementedError

    def _build_depth_encoder(self, input_stages, input_channel, out_channel):
        # 旧，不支持多input stages
        # for stage_idx in input_stages:
        #     setattr(self, 'depth_encoder_%d'%(stage_idx), self.build_bevdepth_depthnet(
        #         in_channel=input_channel,
        #         mid_channel=512,
        #         img_feat_channel=self.image_channel,
        #         depth_channel=self.depth_channel,
        #     ))

        # 新，支持多input stages
        for stage_idx in input_stages[0:1]:
            setattr(self, 'depth_encoder', self.build_bevdepth_depthnet(
                in_channel=input_channel,
                mid_channel=512,
                img_feat_channel=self.image_channel,
                depth_channel=self.depth_channel,
            ))

    def build_bevdepth_depthnet(self, in_channel, mid_channel, img_feat_channel, depth_channel):
        return DepthNet(in_channel, mid_channel, img_feat_channel, depth_channel)

    def vis_depth(self, depth):
        import cv2
        import numpy
        depth = depth.permute(0, 2, 3, 1).to('cpu')
        depth_dist = torch.arange(2, 58, 0.5).float()[None, None, None, :]
        depth = (depth * depth_dist).sum(dim=-1)
        for i, depth_img in enumerate(depth):
            depth_img = depth_img / 58.
            depth_img = depth_img.numpy()[:, :, None]
            depth_img = cv2.resize(depth_img, dsize=None, fx=10, fy=10)
            cv2.imshow('img%d' % i, depth_img)
        # cv2.waitKey(0)
        pass

    def _forward_voxel_net(self, img_feat_with_depth):
        if self.use_da:
            # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
            img_feat_with_depth = img_feat_with_depth.permute(
                0, 3, 1, 4,
                2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
            n, h, c, w, d = img_feat_with_depth.shape
            img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
            img_feat_with_depth = (
                self.depth_aggregation_net(img_feat_with_depth).view(
                    n, h, c, w, d).permute(0, 2, 4, 1, 3).contiguous().float())
        return img_feat_with_depth

    @autocast(False)
    def forward(self,
                feats,
                rots,
                trans,
                intrins,
                warps,
                post_trans=None,
                post_rots=None,
                bev_aug_args=None):
        # assert len(self.input_stages) == 1  # 旧代码不支持多个input stages
        if rots is None and trans is None:
            # only for flops calculation
            if torch.onnx.is_in_onnx_export():
                raise NotImplementedError

            bs = 1
            out_c = sum(self.bevencode.bev_neck_conf['out_channels'])
            bev_feats = torch.rand(size=(bs, out_c, self.nx[0], self.nx[1]), dtype=torch.float32)
            return {'bev_feats': bev_feats, 'depths': None}

        if rots.ndim == 4:
            bs, n_cams, _, _ = rots.size()
            n_sweeps = 1
        elif rots.ndim == 5:
            bs, n_sweeps, n_cams, _, _ = rots.size()
        else:
            raise ValueError

        device = feats[0].device
        if self.dx.device != device:
            self.dx = self.dx.to(device)
            self.bx = self.bx.to(device)
            self.nx = self.nx.to(device)

        geoms = []
        if rots is not None:
            if rots.ndim == 4:
                # bs, n_cams, ...
                geom_one_sweep = self.get_geometry(rots, trans, intrins, warps, post_trans, post_rots, bev_aug_args)
                geoms.append(geom_one_sweep)
            elif rots.ndim == 5:
                # bs, n_sweeps, n_cams, ...
                for i in range(n_sweeps):
                    geom_one_sweep = self.get_geometry(rots[:, i], trans[:, i], intrins[:, i], warps[:, i],
                                                       post_trans[:, i], post_rots[:, i], bev_aug_args)
                    geoms.append(geom_one_sweep)
            else:
                raise ValueError

        # image feature
        # img_feats = [[]] * n_sweeps
        # for stage_idx in self.input_stages:
        #     cam_encoder = eval("self.cam_encoder_%d" % stage_idx)
        #     img_feat = cam_encoder(feats[stage_idx])
        #     _, c, h, w = img_feat.size()
        #     img_feat = img_feat.view(bs, n_sweeps, n_cams, c, h, w)
        #     for i in range(n_sweeps):
        #         img_feats[i].append(img_feat[:, i])

        # depth feature
        contexts = [[] for i in range(n_sweeps)]
        depths = []
        for stage_idx in self.input_stages:
            # depth_encoder = eval("self.depth_encoder_%d"%(stage_idx))  # 旧
            depth_encoder = eval("self.depth_encoder")
            for sweep_idx in range(n_sweeps):
                if feats[stage_idx].ndim == 4:
                    depth_img_feat = depth_encoder(feats[stage_idx],
                                                   rots,
                                                   trans,
                                                   intrins,
                                                   warps,
                                                   bev_aug_args)
                elif feats[stage_idx].ndim == 5:
                    if sweep_idx == 0:
                        depth_img_feat = depth_encoder(feats[stage_idx][:, sweep_idx],
                                                       rots[:, sweep_idx],
                                                       trans[:, sweep_idx],
                                                       intrins[:, sweep_idx],
                                                       warps[:, sweep_idx],
                                                       bev_aug_args)
                    else:
                        with torch.no_grad():
                            depth_img_feat = depth_encoder(feats[stage_idx][:, sweep_idx],
                                                           rots[:, sweep_idx],
                                                           trans[:, sweep_idx],
                                                           intrins[:, sweep_idx],
                                                           warps[:, sweep_idx],
                                                           bev_aug_args)
                else:
                    raise ValueError
                _, c, h, w = depth_img_feat.size()
                # depth_img_feat = depth_img_feat.view(bs, n_sweeps, n_cams, c, h, w)
                depth = depth_img_feat[:, :self.depth_channel].softmax(1)
                img_feat_with_depth = depth.unsqueeze(
                    1) * depth_img_feat[:, self.depth_channel:(
                        self.depth_channel + self.image_channel)].unsqueeze(2)

                if sweep_idx == 0:
                    img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)
                else:
                    with torch.no_grad():
                        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

                if sweep_idx == 0:
                    depths.append(depth)
                img_feat_with_depth = img_feat_with_depth.reshape(
                    bs,
                    n_cams,
                    img_feat_with_depth.shape[1],
                    img_feat_with_depth.shape[2],
                    img_feat_with_depth.shape[3],
                    img_feat_with_depth.shape[4],
                )

                context_feat = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
                contexts[sweep_idx].append(context_feat)

        voxel_feats = []
        for geom, context_feat in zip(geoms, contexts):
            voxel_feat_multi_scales = [self.voxel_pooling(g, c) for g, c in zip(geom, context_feat)]
            voxel_feat_multi_scales = torch.stack(voxel_feat_multi_scales)
            voxel_feat = torch.mean(voxel_feat_multi_scales, dim=0)
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
        # voxel_feats = torch.stack(voxel_feats)
        # voxel_feats = torch.sum(voxel_feats, dim=0)

        out = self.bevencode(voxel_feats)

        # self.vis_depth(depths[0])

        return {'bev_feats': out, 'depths': depths}
