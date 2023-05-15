"""reference from bevdepth"""
import numba
import numpy as np
import torch
from torch.cuda.amp import autocast
import torch.nn.functional as F

try:
    from mmdet3d.core import draw_heatmap_gaussian, gaussian_radius
    from mmdet3d.models import build_neck
    from mmdet3d.models.dense_heads.centerpoint_head import CenterHead, circle_nms
    from mmdet3d.models.utils import clip_sigmoid
    from mmdet.core import reduce_mean
    from mmdet.models import build_backbone
except:
    import warnings

    warnings.warn('Please install mmdet3d first')

from tools.nuscene.data_check import is_target_cls
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from method.data.transform.warp import BevTransform
__all__ = ['BEVCenter3DHead']

TASKS = [
    dict(num_class=1, class_names=['car']),
    dict(num_class=2, class_names=['truck', 'construction_vehicle']),
    dict(num_class=2, class_names=['bus', 'trailer']),
    dict(num_class=1, class_names=['barrier']),
    dict(num_class=2, class_names=['motorcycle', 'bicycle']),
    dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
]

COMMON_HEADS = dict(reg=(2, 2),
                    height=(1, 2),
                    dim=(3, 2),
                    rot=(2, 2),
                    vel=(2, 2))

SEPERATE_HEAD = dict(type='SeparateHead',
                     init_bias=-2.19,
                     final_kernel=3)

TRAIN_CFG = dict(
    point_cloud_range=[-51.2, -51.2, -5, 51.2, 51.2, 3],
    grid_size=[512, 512, 1],
    voxel_size=[0.2, 0.2, 8],
    out_size_factor=4,
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
    code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
)

TEST_CFG = dict(
    post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
    max_per_img=500,
    max_pool_nms=False,
    min_radius=[4, 12, 10, 1, 0.85, 0.175],
    score_threshold=0.1,
    out_size_factor=4,
    voxel_size=[0.2, 0.2, 8],
    nms_type='circle',
    pre_max_size=1000,
    post_max_size=83,
    nms_thr=0.2,
)

@numba.jit(nopython=True)
def size_aware_circle_nms(dets, thresh_scale, post_max_size=83):
    """Circular NMS.
    An object is only counted as positive if no other center
    with a higher confidence exists within a radius r using a
    bird-eye view distance metric.
    Args:
        dets (torch.Tensor): Detection results with the shape of [N, 3].
        thresh (float): Value of threshold.
        post_max_size (int): Max number of prediction to be kept. Defaults
            to 83
    Returns:
        torch.Tensor: Indexes of the detections to be kept.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    dx1 = dets[:, 2]
    dy1 = dets[:, 3]
    yaws = dets[:, 4]
    scores = dets[:, -1]
    order = scores.argsort()[::-1].astype(np.int32)  # highest->lowest
    ndets = dets.shape[0]
    suppressed = np.zeros((ndets), dtype=np.int32)
    keep = []
    for _i in range(ndets):
        i = order[_i]  # start with highest score box
        if suppressed[
            i] == 1:  # if any box have enough iou with this, remove it
            continue
        keep.append(i)
        for _j in range(_i + 1, ndets):
            j = order[_j]
            if suppressed[j] == 1:
                continue
            # calculate center distance between i and j box
            dist_x = abs(x1[i] - x1[j])
            dist_y = abs(y1[i] - y1[j])
            dist_x_th = (abs(dx1[i] * np.cos(yaws[i])) +
                         abs(dx1[j] * np.cos(yaws[j])) +
                         abs(dy1[i] * np.sin(yaws[i])) +
                         abs(dy1[j] * np.sin(yaws[j])))
            dist_y_th = (abs(dx1[i] * np.sin(yaws[i])) +
                         abs(dx1[j] * np.sin(yaws[j])) +
                         abs(dy1[i] * np.cos(yaws[i])) +
                         abs(dy1[j] * np.cos(yaws[j])))
            # ovr = inter / areas[j]
            if dist_x <= dist_x_th * thresh_scale / 2 and \
                    dist_y <= dist_y_th * thresh_scale / 2:
                suppressed[j] = 1
    return keep[:post_max_size]


class BEVCenter3DHead(CenterHead):
    def __init__(self,
                 loss,
                 input_channel,
                 target_names,
                 target_maps,
                 class_names,
                 bbox_coder,
                 xbound,
                 ybound,
                 zbound,
                 dbound,
                 conf_thresh,
                 nms_thresh
                 ):
        super(BEVCenter3DHead, self).__init__(
            in_channels=input_channel,
            tasks=TASKS,
            bbox_coder=bbox_coder,
            common_heads=COMMON_HEADS,
            loss_cls=loss.loss_cls,
            loss_bbox=loss.loss_box,
            separate_head=SEPERATE_HEAD,
        )
        self.target_names = target_names
        self.target_maps = target_maps
        self.class_orders = class_names
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.box_dim = 9  # [x, y, z, l, w, h, yaw, vx, vy]
        self.train_cfg = TRAIN_CFG
        self.test_cfg = TEST_CFG
        self.with_depth_supervised = True if 'depth_supervised' in loss else False
        if self.with_depth_supervised:
            self.depth_channel = loss.depth_supervised.depth_channel
            self.downsample_factor = loss.depth_supervised.downsample_factor
        pass

    @autocast(False)
    def forward(self, x):
        ret_values = super().forward([x] if not isinstance(x, dict) else [x['bev_feats']])
        return ret_values, None if not self.with_depth_supervised else x['depths']

    def loss(self, targets, preds_dicts, **kwargs):
        """Loss function for BEVDepthHead.

        Args:
            gt_bboxes_3d (list[:obj:`LiDARInstance3DBoxes`]): Ground
                truth gt boxes.
            gt_labels_3d (list[torch.Tensor]): Labels of boxes.
            preds_dicts (dict): Output of forward function.

        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        heatmaps, anno_boxes, inds, masks = targets
        return_loss = 0
        loss_dict = {}
        for task_id, preds_dict in enumerate(preds_dicts):
            # heatmap focal loss
            preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
            num_pos = heatmaps[task_id].eq(1).float().sum().item()
            cls_avg_factor = torch.clamp(reduce_mean(
                heatmaps[task_id].new_tensor(num_pos)),
                min=1).item()
            loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'],
                                         heatmaps[task_id],
                                         avg_factor=cls_avg_factor)
            target_box = anno_boxes[task_id]
            # reconstruct the anno_box from multiple reg heads
            if 'vel' in preds_dict[0].keys():
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot'],
                     preds_dict[0]['vel']),
                    dim=1,
                )
            else:
                preds_dict[0]['anno_box'] = torch.cat(
                    (preds_dict[0]['reg'], preds_dict[0]['height'],
                     preds_dict[0]['dim'], preds_dict[0]['rot']),
                    dim=1,
                )
            # Regression loss for dimension, offset, height, rotation
            num = masks[task_id].float().sum()
            ind = inds[task_id]
            pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
            pred = pred.view(pred.size(0), -1, pred.size(3))
            pred = self._gather_feat(pred, ind)
            mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
            num = torch.clamp(reduce_mean(target_box.new_tensor(num)),
                              min=1e-4).item()
            isnotnan = (~torch.isnan(target_box)).float()
            mask *= isnotnan
            code_weights = self.train_cfg['code_weights']
            bbox_weights = mask * mask.new_tensor(code_weights)
            loss_bbox = self.loss_bbox(pred,
                                       target_box,
                                       bbox_weights,
                                       avg_factor=num)
            if loss_bbox.isnan() or loss_heatmap.isnan():
                import warnings
                warnings.warn(
                    'NaN detected in loss function, which will be ignored automatically')
            else:
                return_loss += loss_bbox
                return_loss += loss_heatmap
            loss_dict['task%d_heatmap' % task_id] = loss_heatmap
            loss_dict['task%d_box' % task_id] = loss_bbox
        return return_loss, loss_dict

    def get_bboxes(self, preds_dicts, img_metas=None, img=None, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.

        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            num_class_with_bg = self.num_classes[task_id]
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_heatmap = preds_dict[0]['heatmap'].sigmoid()

            batch_reg = preds_dict[0]['reg']
            batch_hei = preds_dict[0]['height']

            if self.norm_bbox:
                batch_dim = torch.exp(preds_dict[0]['dim'])
            else:
                batch_dim = preds_dict[0]['dim']

            batch_rots = preds_dict[0]['rot'][:, 0].unsqueeze(1)
            batch_rotc = preds_dict[0]['rot'][:, 1].unsqueeze(1)

            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel']
            else:
                batch_vel = None
            temp = self.bbox_coder.decode(batch_heatmap.to(device),
                                          batch_rots.to(device),
                                          batch_rotc.to(device),
                                          batch_hei.to(device),
                                          batch_dim.to(device),
                                          batch_vel.to(device),
                                          reg=batch_reg.to(device),
                                          task_id=task_id)
            assert self.test_cfg['nms_type'] in [
                'size_aware_circle', 'circle', 'rotate'
            ]
            batch_reg_preds = [box['bboxes'] for box in temp]
            batch_cls_preds = [box['scores'] for box in temp]
            batch_cls_labels = [box['labels'] for box in temp]
            if self.test_cfg['nms_type'] == 'circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    centers = boxes3d[:, [0, 1]]
                    boxes = torch.cat([centers, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(circle_nms(
                        boxes.detach().cpu().numpy(),
                        self.test_cfg['min_radius'][task_id],
                        post_max_size=self.test_cfg['post_max_size']),
                        dtype=torch.long,
                        device=boxes.device)

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            elif self.test_cfg['nms_type'] == 'size_aware_circle':
                ret_task = []
                for i in range(batch_size):
                    boxes3d = temp[i]['bboxes']
                    scores = temp[i]['scores']
                    labels = temp[i]['labels']
                    boxes_2d = boxes3d[:, [0, 1, 3, 4, 6]]
                    boxes = torch.cat([boxes_2d, scores.view(-1, 1)], dim=1)
                    keep = torch.tensor(
                        size_aware_circle_nms(
                            boxes.detach().cpu().numpy(),
                            self.test_cfg['thresh_scale'][task_id],
                            post_max_size=self.test_cfg['post_max_size'],
                        ),
                        dtype=torch.long,
                        device=boxes.device,
                    )

                    boxes3d = boxes3d[keep]
                    scores = scores[keep]
                    labels = labels[keep]
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)
                    ret_task.append(ret)
                rets.append(ret_task)
            else:
                rets.append(
                    self.get_task_detections(num_class_with_bg,
                                             batch_cls_preds, batch_reg_preds,
                                             batch_cls_labels, img_metas))

        # Merge branches results
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            for k in rets[0][i].keys():
                if k == 'bboxes':
                    bboxes = torch.cat([ret[i][k] for ret in rets])
                elif k == 'scores':
                    scores = torch.cat([ret[i][k] for ret in rets])
                elif k == 'labels':
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    labels = torch.cat([ret[i][k].int() for ret in rets])
            ret_list.append([bboxes, scores, labels])
        return ret_list

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.

        Returns:
            tuple[list[torch.Tensor]]: Tuple of target including \
                the following results in order.

                - list[torch.Tensor]: Heatmap scores.
                - list[torch.Tensor]: Ground truth boxes.
                - list[torch.Tensor]: Indexes indicating the position \
                    of the valid boxes.
                - list[torch.Tensor]: Masks indicating which boxes \
                    are valid.
        """
        max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])

        feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

        # reorganize the gt_dict by tasks
        task_masks = []
        flag = 0
        for class_name in self.class_names:
            task_masks.append([
                torch.where(gt_labels_3d == class_name.index(i) + flag)
                for i in class_name
            ])
            flag += len(class_name)

        task_boxes = []
        task_classes = []
        flag2 = 0
        for idx, mask in enumerate(task_masks):
            task_box = []
            task_class = []
            for m in mask:
                task_box.append(gt_bboxes_3d[m])
                # 0 is background for each task, so we need to add 1 here.
                task_class.append(gt_labels_3d[m] + 1 - flag2)
            task_boxes.append(
                torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
            task_classes.append(
                torch.cat(task_class).long().to(gt_bboxes_3d.device))
            flag2 += len(mask)
        draw_gaussian = draw_heatmap_gaussian
        heatmaps, anno_boxes, inds, masks = [], [], [], []

        for idx, task_head in enumerate(self.task_heads):
            heatmap = gt_bboxes_3d.new_zeros(
                (len(self.class_names[idx]), feature_map_size[1],
                 feature_map_size[0]),
                device=self.device)

            anno_box = gt_bboxes_3d.new_zeros(
                (max_objs, len(self.train_cfg['code_weights'])),
                dtype=torch.float32,
                device=self.device)

            ind = gt_labels_3d.new_zeros((max_objs),
                                         dtype=torch.int64,
                                         device=self.device)
            mask = gt_bboxes_3d.new_zeros((max_objs),
                                          dtype=torch.uint8,
                                          device=self.device)

            num_objs = min(task_boxes[idx].shape[0], max_objs)

            for k in range(num_objs):
                cls_id = task_classes[idx][k] - 1

                width = task_boxes[idx][k][3]
                length = task_boxes[idx][k][4]
                width = width / voxel_size[0] / self.train_cfg[
                    'out_size_factor']
                length = length / voxel_size[1] / self.train_cfg[
                    'out_size_factor']

                if width > 0 and length > 0:
                    radius = gaussian_radius(
                        (length, width),
                        min_overlap=self.train_cfg['gaussian_overlap'])
                    radius = max(self.train_cfg['min_radius'], int(radius))

                    # be really careful for the coordinate system of
                    # your box annotation.
                    x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][
                        1], task_boxes[idx][k][2]

                    coor_x = (
                                     x - pc_range[0]
                             ) / voxel_size[0] / self.train_cfg['out_size_factor']
                    coor_y = (
                                     y - pc_range[1]
                             ) / voxel_size[1] / self.train_cfg['out_size_factor']

                    center = torch.tensor([coor_x, coor_y],
                                          dtype=torch.float32,
                                          device=self.device)
                    center_int = center.to(torch.int32)

                    # throw out not in range objects to avoid out of array
                    # area when creating the heatmap
                    if not (0 <= center_int[0] < feature_map_size[0]
                            and 0 <= center_int[1] < feature_map_size[1]):
                        continue

                    draw_gaussian(heatmap[cls_id], center_int, radius)

                    new_idx = k
                    x, y = center_int[0], center_int[1]

                    assert y * feature_map_size[0] + x < feature_map_size[
                        0] * feature_map_size[1]

                    ind[new_idx] = y * feature_map_size[0] + x
                    mask[new_idx] = 1
                    # TODO: support other outdoor dataset
                    if len(task_boxes[idx][k]) > 7:
                        vx, vy = task_boxes[idx][k][7:]
                    rot = task_boxes[idx][k][6]
                    box_dim = task_boxes[idx][k][3:6]
                    if self.norm_bbox:
                        box_dim = box_dim.log()
                    if len(task_boxes[idx][k]) > 7:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=self.device),
                            z.unsqueeze(0),
                            box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0),
                            vx.unsqueeze(0),
                            vy.unsqueeze(0),
                        ])
                    else:
                        anno_box[new_idx] = torch.cat([
                            center - torch.tensor([x, y], device=self.device),
                            z.unsqueeze(0), box_dim,
                            torch.sin(rot).unsqueeze(0),
                            torch.cos(rot).unsqueeze(0)
                        ])

            heatmaps.append(heatmap)
            anno_boxes.append(anno_box)
            masks.append(mask)
            inds.append(ind)
        return heatmaps, anno_boxes, inds, masks

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
                if box['num_pts'] <= 0:
                    # 没有lidar radar点
                    continue
                # if not self.in_range(box['translation'], self.xbound, self.ybound, self.zbound):
                #     # 不在感知范围
                #     # 不在感知范围
                #     continue

                gt_label = None
                for cls_idx, cls_name in enumerate(self.target_names):
                    if cls_name in box['category_name']:
                        idx = self.target_names.index(cls_name)
                        gt_name = self.target_maps[idx]
                        gt_label = self.class_orders.index(gt_name)
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

                gt_boxes_batch.append([x, y, z, l, w, h, yaw, vx, vy])
                gt_labels_batch.append(gt_label)
            if len(gt_boxes_batch):
                gts['gt_boxes_3d'].append(torch.from_numpy(np.array(gt_boxes_batch, dtype=np.float32)).to(device))
                gts['gt_labels_3d'].append(torch.from_numpy(np.array(gt_labels_batch, dtype=np.int64)).to(device))
            else:
                gts['gt_boxes_3d'].append(torch.from_numpy(np.zeros((0, self.box_dim), dtype=np.float32)).to(device))
                gts['gt_labels_3d'].append(torch.from_numpy(np.array([], dtype=np.int64)).to(device))
        return gts

    def calc_loss(self, preds, gt_meta):
        preds, depth_preds = preds
        device = preds[0][0]['reg'].device
        if not hasattr(self, 'device'):
            self.device = device
        gts = self.generate_gt(gt_meta, device)

        targets = self.get_targets(gts['gt_boxes_3d'], gts['gt_labels_3d'])
        loss, loss_dict = self.loss(targets, preds)
        loss_dict.update({'total_loss': loss})

        if self.with_depth_supervised:
            camera_orders = gt_meta['camera_types'][0]
            depth_gts = [gt_meta['depth_imgs'][cam] for cam in camera_orders]
            depth_gts = torch.stack(depth_gts, dim=1)
            depth_loss = self.get_depth_loss(depth_gts, depth_preds)
            loss += depth_loss
            loss_dict.update({'depth_loss': depth_loss})
        loss_dict.update({'total_loss': loss})

        return loss, loss_dict

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
        # import cv2
        # gts = self.generate_gt(meta, 'cpu')
        # if not hasattr(self, 'device'):
        #     self.device = 'cpu'
        # targets = self.get_targets(gts['gt_boxes_3d'], gts['gt_labels_3d'])
        # heatmap = targets[0][0]
        # for map in heatmap:
        #     img = map[0].cpu().numpy()
        #     cv2.imshow('gt_task1', img)
        #     # cv2.waitKey(0)
        #
        # heatmap = preds[0][0]['heatmap']
        # for map in heatmap:
        #     img = map[0].sigmoid().cpu().numpy()
        #     # mask = (img > 0.05).astype(np.float32)
        #     cv2.imshow('pred_task1', img)
        #     # cv2.waitKey(0)
        preds, _ = preds
        res = self.get_bboxes(preds)
        outs = {}
        for batch_idx, results in enumerate(res):
            bboxes, scores, labels = results
            pos = scores > self.conf_thresh
            bboxes = bboxes[pos]
            scores = scores[pos]
            labels = labels[pos]
            outs[batch_idx] = []
            for box, score, label in zip(bboxes, scores, labels):
                x, y, z, l, w, h, yaw, vx, vy = box.cpu().numpy()
                r = R.from_euler('zyx', [yaw, 0., 0.], degrees=False)
                xr, yr, zr, w_ = r.as_quat()
                rotation = np.array([w_, xr, yr, zr], dtype=np.float32)
                res = {
                    'translation': np.array([x, y, z], dtype=np.float32),
                    'class_name': self.class_orders[label.item()],
                    'class_idx': label.item(),
                    'size': np.array([w, l, h], dtype=np.float32),
                    'rotation': rotation,
                    'conf': score.item(),
                    'velocity': np.array([vx, vy], dtype=np.float32)
                }
                outs[batch_idx].append(res)
        return outs
