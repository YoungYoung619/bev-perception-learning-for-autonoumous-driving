"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import os
import numpy as np
import torch
import torchvision
from tqdm import tqdm
from pyquaternion import Quaternion
from PIL import Image
from functools import reduce
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.map_expansion.map_api import NuScenesMap


# def get_lidar_data(data_root, samples, cur_sample, nsweeps, min_distance):
#     """
#     Returns at most nsweeps of lidar in the ego frame.
#     Returned tensor is 5(x, y, z, reflectance, dt) x N
#     Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
#
#     nsweeps is used to aug the lidar point:
#         we will use the lidar points from past nsweeps timestamp to used as points at current timestamp
#     """
#     points = np.zeros((5, 0))
#     sensor_tag = 'LIDAR_TOP'
#     # Get reference pose and timestamp.
#     ref_pose_rec = cur_sample['ego_pose'][sensor_tag]
#     ref_time = 1e-6 * cur_sample['lidar_timestamp'][sensor_tag]
#
#     # Homogeneous transformation matrix from global to _current_ ego car frame.
#     car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
#                                         inverse=True)
#
#     # Aggregate current and previous sweeps.
#     current_sd_rec = cur_sample
#     for _ in range(nsweeps):
#         # Load up the pointcloud and remove points close to the sensor.
#         current_pc = LidarPointCloud.from_file(os.path.join(data_root, current_sd_rec['lidar_pcd_files'][sensor_tag]))
#         current_pc.remove_close(min_distance)
#
#         # Get past pose.
#         current_pose_rec = cur_sample['ego_pose'][sensor_tag]
#         global_from_car = transform_matrix(current_pose_rec['translation'],
#                                             Quaternion(current_pose_rec['rotation']), inverse=False)
#
#         # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
#         current_cs_rec = cur_sample['lidar_calibration'][sensor_tag]
#         car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
#                                             inverse=False)
#
#         # Fuse four transformation matrices into one and perform transform.
#         trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
#         current_pc.transform(trans_matrix)
#
#         # Add time vector which can be used as a temporal feature.
#         time_lag = ref_time - 1e-6 * current_sd_rec['lidar_timestamp'][sensor_tag]
#         times = time_lag * np.ones((1, current_pc.nbr_points()))
#
#         new_points = np.concatenate((current_pc.points, times), 0)
#         points = np.concatenate((points, new_points), 1)
#
#         # Abort if there are no previous sweeps.
#         if current_sd_rec['prev'] == None:
#             break
#         else:
#             current_sd_rec = samples[str(current_sd_rec['prev'])]
#
#     return points

def get_lidar_data(nusc, sample_rec, nsweeps, min_distance):
    """
    Returns at most nsweeps of lidar in the ego frame.
    Returned tensor is 5(x, y, z, reflectance, dt) x N
    Adapted from https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py#L56
    """
    points = np.zeros((5, 0))

    sensor_standard = 'LIDAR_TOP'
    # Get reference pose and timestamp.
    if sensor_standard == 'LIDAR_TOP':
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']
        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)
    elif sensor_standard == 'CAM':
        sensor_names = [name for name in sample_rec['data'].keys() if 'CAM' in name]
        translation = np.zeros(3, dtype=np.float32)
        rotation = np.zeros(4, dtype=np.float32)
        ref_time = 0
        for sensor_name in sensor_names:
            ref_sd_token = sample_rec['data'][sensor_name]
            ref_sd_rec = nusc.get('sample_data', ref_sd_token)
            ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
            translation += np.array(ref_pose_rec['translation'])
            rotation += np.array(ref_pose_rec['rotation'])
            ref_time += 1e-6 * ref_sd_rec['timestamp']
        translation /= len(sensor_names)
        rotation /= len(sensor_names)
        car_from_global = transform_matrix(translation, Quaternion(rotation.tolist()),
                                           inverse=True)
        ref_time /= len(sensor_names)
    else:
        raise NotImplementedError

    # Aggregate current and previous sweeps.
    sample_data_token = sample_rec['data']['LIDAR_TOP']
    current_sd_rec = nusc.get('sample_data', sample_data_token)
    for _ in range(nsweeps):
        # Load up the pointcloud and remove points close to the sensor.
        current_pc = LidarPointCloud.from_file(os.path.join(nusc.dataroot, current_sd_rec['filename']))
        current_pc.remove_close(min_distance)

        # Get past pose.
        current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)

        # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)

        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [car_from_global, global_from_car, car_from_current])
        current_pc.transform(trans_matrix)

        # Add time vector which can be used as a temporal feature.
        time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']
        times = time_lag * np.ones((1, current_pc.nbr_points()))

        new_points = np.concatenate((current_pc.points, times), 0)
        points = np.concatenate((points, new_points), 1)

        # Abort if there are no previous sweeps.
        if current_sd_rec['prev'] == '':
            break
        else:
            current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

    return points

def ego_to_cam(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    points = points - trans.unsqueeze(1)
    points = rot.permute(1, 0).matmul(points)

    points = intrins.matmul(points)
    points[:2] /= points[2:3]

    return points

def ego_to_cam_np(points, rot, trans, intrins):
    """Transform points (3 x N) from ego frame into a pinhole camera
    """
    pts = rot.transpose() @ (points - trans[:, None])
    pixel = (intrins @ pts)
    pixel[:2] /= pixel[2][None, :]

    return pixel


def cam_to_ego(points, rot, trans, intrins):
    """Transform points (3 x N) from pinhole camera with depth
    to the ego frame
    """
    points = torch.cat((points[:2] * points[2:3], points[2:3]))
    points = intrins.inverse().matmul(points)

    points = rot.matmul(points)
    points += trans.unsqueeze(1)

    return points


def get_only_in_img_mask(pts, H, W):
    """pts should be 3 x N
    """
    return (pts[2] > 0) &\
        (pts[0] > 1) & (pts[0] < W - 1) &\
        (pts[1] > 1) & (pts[1] < H - 1)


def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])


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
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


denormalize_img = torchvision.transforms.Compose((
            NormalizeInverse(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
            torchvision.transforms.ToPILImage(),
        ))


normalize_img = torchvision.transforms.Compose((
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
))


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]))

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)
        return loss


def get_batch_iou(preds, binimgs):
    """Assumes preds has NOT been sigmoided yet
    """
    with torch.no_grad():
        pred = (preds > 0)
        tgt = binimgs.bool()
        intersect = (pred & tgt).sum().float().item()
        union = (pred | tgt).sum().float().item()
    return intersect, union, intersect / union if (union > 0) else 1.0


def get_val_info(model, valloader, loss_fn, device, use_tqdm=False):
    model.eval()
    total_loss = 0.0
    total_intersect = 0.0
    total_union = 0
    print('running eval...')
    loader = tqdm(valloader) if use_tqdm else valloader
    with torch.no_grad():
        for batch in loader:
            allimgs, rots, trans, intrins, post_rots, post_trans, binimgs = batch
            preds = model(allimgs.to(device), rots.to(device),
                          trans.to(device), intrins.to(device), post_rots.to(device),
                          post_trans.to(device))
            binimgs = binimgs.to(device)

            # loss
            total_loss += loss_fn(preds, binimgs).item() * preds.shape[0]

            # iou
            intersect, union, _ = get_batch_iou(preds, binimgs)
            total_intersect += intersect
            total_union += union

    model.train()
    return {
            'loss': total_loss / len(valloader.dataset),
            'iou': total_intersect / total_union,
            }


def add_ego(bx, dx):
    # approximate rear axel
    W = 1.85
    pts = np.array([
        [-4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, W/2.],
        [4.084/2.+0.5, -W/2.],
        [-4.084/2.+0.5, -W/2.],
    ])
    pts = (pts - bx) / dx
    pts[:, [0,1]] = pts[:, [1,0]]
    plt.fill(pts[:, 0], pts[:, 1], '#76b900')


def get_nusc_maps(map_folder):
    nusc_maps = {map_name: NuScenesMap(dataroot=map_folder,
                map_name=map_name) for map_name in [
                    "singapore-hollandvillage", 
                    "singapore-queenstown",
                    "boston-seaport",
                    "singapore-onenorth",
                ]}
    return nusc_maps


def plot_nusc_map(rec, nusc_maps, nusc, scene2map, dx, bx):
    egopose = nusc.get('ego_pose', nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
    map_name = scene2map[nusc.get('scene', rec['scene_token'])['name']]

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    poly_names = ['road_segment', 'lane']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(nusc_maps[map_name], center,
                         50.0, poly_names, line_names)
    for name in poly_names:
        for la in lmap[name]:
            pts = (la - bx) / dx
            plt.fill(pts[:, 1], pts[:, 0], c=(1.00, 0.50, 0.31), alpha=0.2)
    for la in lmap['road_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(0.0, 0.0, 1.0), alpha=0.5)
    for la in lmap['lane_divider']:
        pts = (la - bx) / dx
        plt.plot(pts[:, 1], pts[:, 0], c=(159./255., 0.0, 1.0), alpha=0.5)


def get_local_map(nmap, center, stretch, layer_names, line_names):
    # need to get the map here...
    box_coords = (
        center[0] - stretch[0],
        center[1] - stretch[1],
        center[0] + stretch[0],
        center[1] + stretch[1],
    )

    polys = {}

    # polygons
    records_in_patch = nmap.get_records_in_patch(box_coords,
                                                 layer_names=layer_names+line_names,
                                                 mode='intersect')
    for layer_name in layer_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            poly_record = nmap.get(layer_name, token)
            if layer_name == 'drivable_area':
                polygon_tokens = poly_record['polygon_tokens']
            else:
                polygon_tokens = [poly_record['polygon_token']]

            for polygon_token in polygon_tokens:
                polygon = nmap.extract_polygon(polygon_token)
                polys[layer_name].append(np.array(polygon.exterior.xy).T)

    # lines
    for layer_name in line_names:
        polys[layer_name] = []
        for token in records_in_patch[layer_name]:
            line_record = nmap.get(layer_name, token)
            line = nmap.extract_line(line_record['line_token'])
            if line.is_empty:  # Skip lines without nodes
                continue
            xs, ys = line.xy

            polys[layer_name].append(
                np.array([xs, ys]).T
                )
    # for layer_name in line_names:
    #     polys[layer_name] = []
    #     for record in getattr(nmap, layer_name):
    #         print(1)
    #         token = record['token']
    #
    #         line = nmap.extract_line(record['line_token'])
    #         if line.is_empty:  # Skip lines without nodes
    #             continue
    #         xs, ys = line.xy
    #
    #         polys[layer_name].append(
    #             np.array([xs, ys]).T
    #             )

    # convert to local coordinates in place
    rot = get_rot(np.arctan2(center[3], center[2])).T
    for layer_name in polys:
        for rowi in range(len(polys[layer_name])):
            polys[layer_name][rowi] -= center[:2]
            polys[layer_name][rowi] = np.dot(polys[layer_name][rowi], rot)

    return polys

if __name__ == '__main__':
    import cv2
    from nuscenes.nuscenes import NuScenes
    map_folder = '/Volumes/KINGSTON0/nuscenes/nuScenes-map-expansion-v1.3'
    nusc_maps = get_nusc_maps(map_folder)

    version = 'mini'
    dataroot = '/Volumes/KINGSTON0/nuscenes'
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)

    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    for each_scene in nusc.scene:
        scene_name = each_scene['name']
        print('parsing scene %s' % scene_name)

        # sample parse
        n_samples, cur_sample, last_sample = each_scene['nbr_samples'], each_scene['first_sample_token'], each_scene[
            'last_sample_token']
        samples = {}
        while cur_sample != last_sample:
            plt.clf()
            # plt.axis('off')
            print('a')
            sample = nusc.get('sample', cur_sample)
            timestamp = sample['timestamp']
            dx = [1, 1]
            bx = [-49.75, -49.75]
            plot_nusc_map(sample, nusc_maps, nusc, scene2map, dx=dx, bx=bx)
            add_ego(bx, dx)
            plt.xlim((100, 0))
            plt.ylim((0, 100))
            img_name = "%s_map.jpg"%(timestamp)
            plt.savefig('./save/%s' % img_name, bbox_inches='tight')

            img = cv2.imread('./save/%s' % img_name)
            cv2.imshow('aa', img)
            cv2.waitKey(0)

            cur_sample = sample['next']
    pass