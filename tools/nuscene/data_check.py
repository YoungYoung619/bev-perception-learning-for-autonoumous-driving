import glob
import os
import json
import sys
sys.path.insert(0, '/Users/lvanyang/ADAS/ADMultiTaskPerception')
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
import cv2
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from torch.utils.data import Dataset

from tools.nuscene.utils import (
    get_local_map,
    get_lidar_data,
    ego_to_cam_np,
    get_only_in_img_mask
)


def draw_corners(img, points, color, thickness):
    front = points[:4]
    cv2.polylines(img, [front], isClosed=True, color=color, thickness=thickness)
    back = points[4:]
    cv2.polylines(img, [back], isClosed=True, color=color, thickness=thickness)

    # direction
    front_bottom = points[2:4]
    back_bottom = points[-2:]
    front = np.concatenate([front_bottom, back_bottom], axis=0)
    center = np.mean(front, axis=0).astype(np.int32)
    front_center = np.mean(front_bottom, axis=0).astype(np.int32)
    cv2.line(img, pt1=(center[0], center[1]), pt2=(front_center[0], front_center[1]), color=color,
             thickness=thickness)

    for i in range(0, 4):
        j = i + 4
        cv2.line(img, pt1=(points[i][0], points[i][1]), pt2=(points[j][0], points[j][1]), color=color,
                 thickness=thickness)


def is_box_in_image(corners_pixel, corners_3d, img_size, visible_level):
    """
    corners_3d: must be camera coordinate
    visible_level: all required all corner point in image, any required any corner point in image
    """
    assert visible_level in ['all', 'any']
    visible = np.logical_and(corners_pixel[:, 0] > 0, corners_pixel[:, 0] < img_size[0])
    visible = np.logical_and(visible, corners_pixel[:, 1] < img_size[1])
    visible = np.logical_and(visible, corners_pixel[:, 1] > 0)
    visible = np.logical_and(visible, corners_3d[:, 2] > 1)

    in_front = corners_3d[:, 2] > 0.
    return all(visible) if visible_level == 'all' else any(visible) and all(in_front)


def is_target_cls(cls_name, targets):
    if targets is None or 'all' in targets or targets == 'all':
        return True
    for target in targets:
        if target in cls_name:
            return True
    return False


def quaternion_to_list(quaternion: Quaternion):
    return [quaternion.w, quaternion.x, quaternion.y, quaternion]


def concat_camera_imgs(imgs):
    row1 = np.concatenate([imgs['CAM_FRONT_LEFT'], imgs['CAM_FRONT'], imgs['CAM_FRONT_RIGHT']], axis=1)
    row2 = np.concatenate([imgs['CAM_BACK_LEFT'], imgs['CAM_BACK'], imgs['CAM_BACK_RIGHT']], axis=1)
    all = np.concatenate([row1, row2], axis=0)
    return all


def show_img(win_name, img, wait_key, width=1280):
    h, w, _ = img.shape
    ratio = width / w
    img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
    cv2.imshow(win_name, img)
    if wait_key:
        return cv2.waitKey(0)


def quaternion_2_rotation(quaternion):
    qx, qy, qz, qw = quaternion[1], quaternion[2], quaternion[3], quaternion[0]
    matrix = np.array([[1. - 2 * qy ** 2 - 2 * qz ** 2, 2 * qy * qx + 2 * qw * qz, 2 * qx * qz - 2 * qw * qy],
                       [2 * qy * qx - 2 * qw * qz, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz + 2 * qw * qx],
                       [2 * qx * qz + 2 * qw * qy, 2 * qy * qz - 2 * qw * qx, 1 - 2 * qx ** 2 - 2 * qy ** 2]],
                      dtype=np.float32)
    return matrix.transpose()


def corners_3d_box(center,
                   orientation: Quaternion,
                   size,
                   wlh_factor: float = 1.0) -> np.ndarray:
    """
    Returns the bounding box corners.
    :param wlh_factor: Multiply w, l, h by a factor to scale the box.
    :return: <np.float: 3, 8>. First four corners are the ones facing forward.
        The last four are the ones facing backwards.
    """
    w, l, h = size[0] * wlh_factor, size[1] * wlh_factor, size[2] * wlh_factor

    # 3D bounding box corners. (Convention: x points forward, y to the left, z up.)
    x_corners = l / 2 * np.array([1, 1, 1, 1, -1, -1, -1, -1])
    y_corners = w / 2 * np.array([1, -1, -1, 1, 1, -1, -1, 1])
    z_corners = h / 2 * np.array([1, 1, -1, -1, 1, 1, -1, -1])
    corners = np.vstack((x_corners, y_corners, z_corners))

    # Rotate
    corners = np.dot(orientation.rotation_matrix, corners)

    # Translate
    x, y, z = center
    corners[0, :] = corners[0, :] + x
    corners[1, :] = corners[1, :] + y
    corners[2, :] = corners[2, :] + z

    return corners


def draw_center_perspective(p_img, p, color, thickness, radius):
    cv2.circle(p_img, (int(p[0]), int(p[1])), color=color, thickness=thickness, radius=radius)


def draw_corners_perspective(p_img, ps, color, thickness):
    cv2.line(p_img, pt1=(int(ps[0][0]), int(ps[0][1])), pt2=(int(ps[1][0]), int(ps[1][1])),
             color=color, thickness=thickness)
    cv2.line(p_img, pt1=(int(ps[2][0]), int(ps[2][1])), pt2=(int(ps[3][0]), int(ps[3][1])),
             color=color, thickness=thickness)
    cv2.line(p_img, pt1=(int(ps[2][0]), int(ps[2][1])), pt2=(int(ps[3][0]), int(ps[3][1])),
             color=color, thickness=thickness)
    cv2.line(p_img, pt1=(int(ps[0][0]), int(ps[0][1])), pt2=(int(ps[2][0]), int(ps[2][1])),
             color=color, thickness=thickness)
    cv2.line(p_img, pt1=(int(ps[1][0]), int(ps[1][1])), pt2=(int(ps[3][0]), int(ps[3][1])),
             color=color, thickness=thickness)


def get_camera_data(data_root, nusc):
    camera_data = {}
    for each_scene in nusc.scene:
        scene_name = each_scene['name']
        print('parsing scene %s' % scene_name)
        camera_data[scene_name] = {}

        # sample parse
        n_samples, cur_sample, last_sample = each_scene['nbr_samples'], each_scene['first_sample_token'], each_scene[
            'last_sample_token']
        while cur_sample != last_sample:
            sample = nusc.get('sample', cur_sample)
            timestamp = sample['timestamp']

            # 获取传感器信息 (camera  lidar  )
            sensor_names = list(sample['data'].keys())
            camera_names = [name for name in sensor_names if 'CAM' in name]
            camera_img_files = {}
            camera_calibration = {}
            ego_pose_for_each_sensor = {}
            camera_timestamp = {}
            for sensor_name, sensor_token in sample['data'].items():
                sample_data = nusc.get('sample_data', sensor_token)

                calibrated_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                if 'CAM' in sensor_name:
                    camera_timestamp[sensor_name] = sample_data['timestamp']
                    img_file = sample_data['filename']
                    camera_img_files[sensor_name] = os.path.join(data_root, img_file)
                    camera_calibration[sensor_name] = {
                        'translation': calibrated_sensor_data['translation'],
                        'rotation': calibrated_sensor_data['rotation'],
                        'camera_intrinsic': calibrated_sensor_data['camera_intrinsic'],
                    }
                    ego_pose_for_each_sensor[sensor_name] = {
                        'translation': ego_pose['translation'],
                        'rotation': ego_pose['rotation'],
                    }

            camera_data[scene_name][timestamp] = {
                'camera_timestamp': camera_timestamp,
                'camera_img_files': camera_img_files,
                'camera_calibration': camera_calibration,
                'ego_pose': ego_pose_for_each_sensor,
            }

            cur_sample = sample['next']
    return camera_data


def get_anns(nusc):
    sample_anns = {}
    samples = {}
    for each_scene in nusc.scene:
        scene_name = each_scene['name']
        print('parsing scene %s' % scene_name)
        sample_anns[scene_name] = {}

        # sample parse
        n_samples, cur_sample, last_sample = each_scene['nbr_samples'], each_scene['first_sample_token'], each_scene[
            'last_sample_token']
        while cur_sample != last_sample:
            sample = nusc.get('sample', cur_sample)
            timestamp = sample['timestamp']

            camera_calibration = {}
            ego_pose_for_each_sensor = {}
            for sensor_name, sensor_token in sample['data'].items():
                sample_data = nusc.get('sample_data', sensor_token)

                calibrated_sensor_data = nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
                ego_pose = nusc.get('ego_pose', sample_data['ego_pose_token'])
                if 'CAM' in sensor_name:
                    camera_calibration[sensor_name] = {
                        'translation': calibrated_sensor_data['translation'],
                        'rotation': calibrated_sensor_data['rotation'],
                        'camera_intrinsic': calibrated_sensor_data['camera_intrinsic'],
                    }
                    ego_pose_for_each_sensor[sensor_name] = {
                        'translation': ego_pose['translation'],
                        'rotation': ego_pose['rotation'],
                    }

            samples[timestamp] = {
                'camera_calibration': camera_calibration,
                'ego_pose': ego_pose_for_each_sensor,
            }

            # ann
            sensor_names = list(sample['data'].keys())
            camera_names = [name for name in sensor_names if 'CAM' in name]
            sample_anns[scene_name][timestamp] = []
            for ann_token in sample['anns']:
                ann_info = nusc.get('sample_annotation', ann_token)

                # 获取实例类别
                instance_token = ann_info['instance_token']
                instance = nusc.get('instance', instance_token)
                category = nusc.get('category', instance['category_token'])
                # category_name = ann_info['category_name']
                category_name = category['name']

                # 可见性
                visibility_token = ann_info['visibility_token']
                visibility = nusc.get('visibility', visibility_token)

                # 属性 (例如: 站立，行走)
                attribute_tokens = ann_info['attribute_tokens']
                attributes = []
                for attr_token in attribute_tokens:
                    attribute = nusc.get('attribute', attr_token)
                    attribute_ = {
                        'name': attribute['name'],
                        'description': attribute['description'],
                    }
                    attributes.append(attribute)

                # global frame
                global_translation = np.array(ann_info['translation'])
                global_quaternion = Quaternion(np.array(ann_info['rotation']))
                global_rotation = global_quaternion.rotation_matrix

                """
                每个摄像头的成像的时间戳不同，因此有不同的ego_pose (自车在全局坐标系下的)，
                因此转换到自身坐标系下时，同一个物体对于不同摄像头有不同的translation 和 rotation
                """
                ann = {
                    'global_frame': {
                        'translation': ann_info['translation'],
                        'rotation': ann_info['rotation'],
                    },
                    'size': ann_info['size'],
                    'category_name': category_name,
                    'visibility': visibility,
                    'attributes': attributes,
                }
                for camera_name in camera_names:
                    ego_translation = np.array(samples[timestamp]['ego_pose'][camera_name]['translation'])
                    ego_quaternion = Quaternion(np.array(samples[timestamp]['ego_pose'][camera_name]['rotation']))
                    ego_rotation = ego_quaternion.rotation_matrix
                    # global to ego frame
                    translation = ego_rotation.transpose() @ (global_translation - ego_translation)
                    quaternion = ego_quaternion.inverse * np.array(ann_info['rotation'])
                    rotation = ego_rotation.transpose() @ global_rotation
                    # assert b == rotation

                    # ego to camera frame
                    camera_translation = np.array(camera_calibration[camera_name]['translation'])
                    camera_quaternion = Quaternion(np.array(camera_calibration[camera_name]['rotation']))
                    camera_rotation = camera_quaternion.rotation_matrix
                    translation_ = camera_rotation.transpose() @ (translation - camera_translation)
                    quaternion_ = camera_quaternion.inverse * quaternion
                    rotation_ = quaternion_.rotation_matrix

                    # 获取角点的像素值
                    box_corners = corners_3d_box(translation_, quaternion_, ann_info['size'])  # 相机坐标下
                    box_corners_global = corners_3d_box(translation, quaternion, ann_info['size'])  # 全局坐标下

                    intrinsic = np.array(camera_calibration[camera_name]['camera_intrinsic'])
                    pixel_points = (intrinsic @ box_corners).transpose()
                    pixel_points = pixel_points[:, :2] / pixel_points[:, 2][:, None]

                    # 中心点像素值
                    center = translation_
                    pixel_center = (intrinsic @ center[:, None]).transpose()
                    pixel_center = pixel_center[:, :2] / pixel_center[:, 2][:, None]

                    ann[camera_name] = {
                        'ego_frame': {
                            'translation': translation.tolist(),
                            'rotation': quaternion_to_list(quaternion),
                            'corners': box_corners_global.transpose().tolist(),
                        },
                        'sensor_frame': {
                            'translation': translation_.tolist(),
                            'rotation': quaternion_to_list(quaternion_),
                            'corners': box_corners.transpose().tolist(),
                        },
                        'pixel_frame': {
                            'corners': pixel_points.tolist(),
                            'center': pixel_center[0].tolist(),
                        },
                    }
                    sample_anns[scene_name][timestamp].append(ann)

            cur_sample = sample['next']
    return sample_anns


def concat_perspective_imgs(imgs):
    new_img = np.zeros_like(imgs['CAM_FRONT'])
    for name, img in imgs.items():
        img_mask1 = new_img > 0
        img_mask2 = img > 0
        inter_mask = img_mask1 & img_mask2
        new_img[inter_mask] = (
                (new_img[inter_mask].astype(np.float32) + img[inter_mask].astype(np.float32)) / 2).astype(np.uint8)

        only_img_mask = (img_mask2.astype(np.uint8) - inter_mask.astype(np.uint8)).astype(np.bool)
        new_img[only_img_mask] = img[only_img_mask]

    return new_img


def get_z_homograph_matrix(translation, orientation, camera_intrinsic, z=0.):
    """获取图像坐标系到指定z平面世界坐标系的homograph矩阵
    后续：point_world (X, Y, Z=z) = H @ point_img
    """
    K = np.array(camera_intrinsic)
    R = Quaternion(np.array(orientation)).rotation_matrix
    T = np.array(translation)
    r = K @ R.transpose()
    t = -r @ T.transpose()
    r[..., -1] = r[..., -1] * z + t
    H = r
    H = np.linalg.inv(H)
    return H


def plot_nusc_map(egopose, map, map_range=[50, 25], img_size=[1000, 500]):
    bx = [-map_range[0] / 2, -map_range[1] / 2]
    dx = [map_range[0] / img_size[0], map_range[1] / img_size[1]]
    height = int(abs(map_range[0] / dx[0]))
    width = int(abs(map_range[1] / dx[1]))

    img = np.ones(shape=(height, width, 3), dtype=np.uint8) * 255

    rot = Quaternion(egopose['rotation']).rotation_matrix
    rot = np.arctan2(rot[1, 0], rot[0, 0])
    center = np.array([egopose['translation'][0], egopose['translation'][1], np.cos(rot), np.sin(rot)])

    # poly_names = ['drivable_area']
    # line_names = ['road_divider', 'lane_divider']
    # poly_names = ['drivable_area', 'road_segment',
    #               'road_block', 'lane', 'ped_crossing',
    #               'walkway', 'stop_line', 'carpark_area']
    poly_names = ['drivable_area', 'road_block']
    line_names = ['road_divider', 'lane_divider']
    lmap = get_local_map(map, center,
                         [map_range[0], map_range[1]], poly_names, line_names)
    for name in poly_names:
        show = img.copy()
        for la in lmap[name]:
            pts = (la - bx) / dx
            pts = pts[:, ::-1]
            pts[:, 0] = width - pts[:, 0]
            pts[:, 1] = height - pts[:, 1]
            show = cv2.fillPoly(show, [pts.astype(np.int32)], color=(int(0.31 * 255), int(0.5 * 255), int(255)))
            # print('la size: ', len(la))
            # cv2.imshow('aa', show)
            # cv2.waitKey(0)
        img = (0.2 * show + 0.8 * img).astype(np.uint8)

    if 'road_divider' in lmap:
        show = img.copy()
        for la in lmap['road_divider']:
            pts = (la - bx) / dx
            pts = pts[:, ::-1]
            pts[:, 0] = width - pts[:, 0]
            pts[:, 1] = height - pts[:, 1]
            for i in range(len(pts) - 1):
                pt1 = int(pts[i][0]), int(pts[i][1])
                pt2 = int(pts[i + 1][0]), int(pts[i + 1][1])
                show = cv2.line(show, pt1=pt1, pt2=pt2, color=(255, 0, 0), thickness=2)
        img = (0.5 * show + 0.5 * img).astype(np.uint8)

    if 'lane_divider' in lmap:
        show = img.copy()
        for la in lmap['lane_divider']:
            pts = (la - bx) / dx
            pts = pts[:, ::-1]
            pts[:, 0] = width - pts[:, 0]
            pts[:, 1] = height - pts[:, 1]
            for i in range(len(pts) - 1):
                pt1 = int(pts[i][0]), int(pts[i][1])
                pt2 = int(pts[i + 1][0]), int(pts[i + 1][1])
                show = cv2.line(show, pt1=pt1, pt2=pt2, color=(255, 0, 159), thickness=2)
        img = (0.5 * show + 0.5 * img).astype(np.uint8)

    # show = img.copy()
    # for la in lmap['stop_line']:
    #     pts = (la - bx) / dx
    #     pts = pts[:, ::-1]
    #     pts[:, 0] = width - pts[:, 0]
    #     pts[:, 1] = height - pts[:, 1]
    #     for i in range(len(pts) - 1):
    #         pt1 = int(pts[i][0]), int(pts[i][1])
    #         pt2 = int(pts[i + 1][0]), int(pts[i + 1][1])
    #         show = cv2.line(show, pt1=pt1, pt2=pt2, color=(255, 0, 159), thickness=2)
    # img = (0.5 * show + 0.5 * img).astype(np.uint8)
    return img


def vis_3d_detection(samples):
    """可视化3d检测数据，只含动态障碍物"""
    vis_targets = ['vehicle', 'human']

    imgs = {}
    camera_names = list(samples['camera_img_files'].keys())
    for camera_name, img_file in samples['camera_img_files'].items():
        if not os.path.exists(img_file):
            print('ignore %s' % (img_file))
            continue
        img = cv2.imread(img_file)
        imgs[camera_name] = img

    anns = samples['anns']
    for ann in anns:
        if not is_target_cls(ann['category_name'], vis_targets):
            continue
        for camera_name in camera_names:
            camera_ann = ann[camera_name]
            pixel_points = np.array(camera_ann['pixel_frame']['corners'], dtype=np.int32)
            points_camera = np.array(camera_ann['sensor_frame']['corners'], dtype=np.float32)
            img = imgs[camera_name]
            h, w, _ = img.shape
            if is_box_in_image(pixel_points, points_camera, [w, h], 'any'):
                draw_corners(img, pixel_points, (200, 123, 180), 5)

    img = concat_camera_imgs(imgs)
    show_img('3d_detection', img, wait_key=False, width=1800)


def vis_bev(samples, vis_img_height, vis_img_width, longitude, lateral):
    """"""
    vis_targets = ['vehicle', 'human']

    longitude /= 2
    lateral /= 2
    src_points = np.array([[longitude, lateral], [longitude, -lateral], [-longitude, -lateral], [-longitude, lateral]],
                          dtype="float32")
    dst_points = np.array([[0., 0.], [vis_img_width, 0.], [vis_img_width, vis_img_height], [0., vis_img_height]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    imgs = {}
    camera_names = list(samples['camera_img_files'].keys())
    camera_calibrations = samples['camera_calibration']
    perspective_imgs = {}
    for camera_name, img_file in samples['camera_img_files'].items():
        img = cv2.imread(img_file)
        imgs[camera_name] = img

        camera_calibration = camera_calibrations[camera_name]

        homograph_matrix = get_z_homograph_matrix(camera_calibration['translation'],
                                                  camera_calibration['rotation'],
                                                  camera_calibration['camera_intrinsic'],
                                                  z=0.2)

        # 消逝点以上的数据会被投影到对称区域，先置0
        half_img = img.copy()
        h, w, _ = img.shape
        half_img[: 180 * h // 350, ...] = 0

        # 通过M和homograph_matrix投影到假想的图像平面
        perspective_img = cv2.warpPerspective(half_img, M @ homograph_matrix, (vis_img_width, vis_img_height))
        perspective_imgs[camera_name] = perspective_img

    anns = samples['anns']
    for ann in anns:
        if not is_target_cls(ann['category_name'], vis_targets):
            continue
        for camera_name in camera_names:
            camera_ann = ann[camera_name]
            pixel_points = np.array(camera_ann['pixel_frame']['corners'], dtype=np.int32)
            points_camera = np.array(camera_ann['sensor_frame']['corners'], dtype=np.float32)
            img = imgs[camera_name]
            h, w, _ = img.shape

            center = np.array(camera_ann['ego_frame']['translation'])[None, :]
            center[0][2] = 1  # z没有用，得改成齐次坐标
            center_perspective = (M @ center.transpose()).transpose()[0]
            center_perspective = center_perspective[:2] / center_perspective[2]

            # 地面上的corners, 前两个是front， 后两个是back
            corners = np.array(
                [
                    camera_ann['ego_frame']['corners'][2],
                    camera_ann['ego_frame']['corners'][3],
                    camera_ann['ego_frame']['corners'][6],
                    camera_ann['ego_frame']['corners'][7],
                ]
            )
            corners[:, 2] = 1  # z没有用，得改成齐次坐标
            corners_perspective = (M @ corners.transpose()).transpose()
            corners_perspective = corners_perspective[:, :2] / corners_perspective[:, 2][:, None]

            if is_box_in_image(pixel_points, points_camera, [w, h], 'any'):
                if center_perspective[0] > 0 and center_perspective[0] < w and \
                        center_perspective[1] > 0 and center_perspective[1] < h:
                    draw_center_perspective(perspective_imgs[camera_name],
                                            center_perspective,
                                            color=(80, 120, 220),
                                            thickness=-1,
                                            radius=2)
                    draw_corners_perspective(perspective_imgs[camera_name],
                                             corners_perspective,
                                             color=(230, 45, 20),
                                             thickness=3)

    p_img = concat_perspective_imgs(perspective_imgs)
    cv2.imshow('bev', p_img)


def vis_lidar(samples):
    imgs = {}
    for camera_name, img_file in samples['camera_img_files'].items():
        img = cv2.imread(img_file)
        imgs[camera_name] = img

    points = samples['lidar_pts']
    pts = np.array(points)[:3, :]
    for camera_name, calibration in samples['camera_calibration'].items():
        cam_pts = ego_to_cam_np(pts,
                                np.array(Quaternion(calibration['rotation']).rotation_matrix),
                                np.array(calibration['translation']),
                                np.array(calibration['camera_intrinsic']))
        h, w, _ = imgs[camera_name].shape
        mask = get_only_in_img_mask(cam_pts, h, w)
        plot_pts = cam_pts.transpose()[mask]

        dis_split = [0, 8, 22, 45]
        for p in plot_pts:
            x = int(p[0])
            y = int(p[1])
            depth = int(p[2])
            if depth >= dis_split[0] and depth <= dis_split[1]:
                color = (int(depth / (dis_split[1] - dis_split[0]) * 255), 0, 0)
            elif depth > dis_split[1] and depth <= dis_split[2]:
                color = (0, int((depth - dis_split[1]) / (dis_split[2] - dis_split[1]) * 255), 0)
            elif depth > dis_split[2] and depth < dis_split[3]:
                color = (0, 0, int((depth - dis_split[2]) / (dis_split[3] - dis_split[2]) * 255))
            else:
                color = (0, 0, 255)
            cv2.circle(imgs[camera_name], (x, y), radius=3, thickness=-1, color=color)

    raw_img = concat_camera_imgs(imgs)

    cv2.imshow('lidar', raw_img)


def ann_test_lidar(samples):
    import open3d as o3d
    img_base = samples['timestamp']
    imgs = {}
    for camera_name, img_file in samples['camera_img_files'].items():
        img = cv2.imread(img_file)
        imgs[camera_name] = img

    calibrations = {}
    for camera_name, calibration in samples['camera_calibration'].items():
        rotation = np.array(Quaternion(calibration['rotation']).rotation_matrix)
        translation = np.array(calibration['translation'])
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = rotation
        extrinsic[:3, 3] = translation
        intrinsic = np.array(calibration['camera_intrinsic'])

        extrinsic = np.linalg.inv(extrinsic)

        intrinsic = np.reshape(intrinsic, newshape=-1).tolist()
        extrinsic = np.reshape(extrinsic, newshape=-1).tolist()
        calibrations[camera_name] = {
            'extrinsic': extrinsic,
            'intrinsic': intrinsic
        }

    points = samples['lidar_pts']
    pts = np.array(points)[:3, :]
    pts = pts.transpose(1, 0)

    save_root = '/Volumes/KINGSTON0/visualization/toann'
    calib_root = os.path.join(save_root, 'calib')
    lidar_root = os.path.join(save_root, 'lidar')
    camera_root = os.path.join(save_root, 'camera')
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    if not os.path.exists(camera_root):
        os.makedirs(calib_root)
    if not os.path.exists(lidar_root):
        os.makedirs(lidar_root)
    if not os.path.exists(calib_root):
        os.makedirs(calib_root)
    if not os.path.exists(os.path.join(calib_root, 'camera')):
        os.makedirs(os.path.join(calib_root, 'camera'))

    for camera_type, camera_img in imgs.items():
        if not os.path.exists(os.path.join(camera_root, camera_type)):
            os.makedirs(os.path.join(camera_root, camera_type))
        img_name = os.path.join(camera_root, camera_type, "%s.jpg"%(img_base))
        cv2.imwrite(img_name, camera_img)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    lidar_name = os.path.join(lidar_root, "%s.pcd"%(img_base))
    o3d.io.write_point_cloud(lidar_name, pcd)

    calib_files = glob.glob(os.path.join(calib_root, 'camera', '*json'))
    if len(calib_files) == 0:
        print('write camera calibration...')
        for camera_type, calibration in calibrations.items():
            with open(os.path.join(calib_root, 'camera', '%s.json'%(camera_type)), 'w') as f:
                json.dump(calibration, f)


def vis_3d_detection_with_map(
        nusc,
        samples,
        map_folder,
        vis_img_height,
        vis_img_width,
        longitude,
        lateral):
    from tools.nuscene.utils import get_nusc_maps

    # map_folder = '/Volumes/KINGSTON0/nuscenes/nuScenes-map-expansion-v1.3'
    nusc_maps = get_nusc_maps(map_folder)
    vis_targets = ['vehicle', 'human']

    scene2map = {}
    for rec in nusc.scene:
        log = nusc.get('log', rec['log_token'])
        scene2map[rec['name']] = log['location']

    vis_h = vis_img_height
    vis_w = vis_img_width
    src_points = np.array([[longitude // 2, lateral // 2],
                           [longitude // 2, -lateral // 2],
                           [-longitude // 2, -lateral // 2],
                           [-longitude // 2, lateral // 2]],
                          dtype="float32")
    dst_points = np.array([[0., 0.], [vis_w, 0.], [vis_w, vis_h], [0., vis_h]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)

    ego_pose = samples['ego_pose']['CAM_FRONT']
    map_img = plot_nusc_map(ego_pose,
                            nusc_maps[scene2map[samples['scene_name']]],
                            map_range=[longitude, lateral], img_size=[vis_h, vis_w])
    camera_names = list(samples['camera_calibration'].keys())
    anns = samples['anns']
    for ann in anns:
        if not is_target_cls(ann['category_name'], vis_targets):
            continue
        for camera_name in camera_names:
            camera_ann = ann[camera_name]
            pixel_points = np.array(camera_ann['pixel_frame']['corners'], dtype=np.int32)
            points_camera = np.array(camera_ann['sensor_frame']['corners'], dtype=np.float32)

            center = np.array(camera_ann['ego_frame']['translation'])[None, :]
            center[0][2] = 1  # z没有用，得改成齐次坐标
            center_perspective = (M @ center.transpose()).transpose()[0]
            center_perspective = center_perspective[:2] / center_perspective[2]

            # 地面上的corners, 前两个是front， 后两个是back
            corners = np.array(
                [
                    camera_ann['ego_frame']['corners'][2],
                    camera_ann['ego_frame']['corners'][3],
                    camera_ann['ego_frame']['corners'][6],
                    camera_ann['ego_frame']['corners'][7],
                ]
            )
            corners[:, 2] = 1  # z没有用，得改成齐次坐标
            corners_perspective = (M @ corners.transpose()).transpose()
            corners_perspective = corners_perspective[:, :2] / corners_perspective[:, 2][:, None]

            w = 1600
            h = 900
            if is_box_in_image(pixel_points, points_camera, [w, h], 'any'):
                if center_perspective[0] > 0 and center_perspective[0] < w and \
                        center_perspective[1] > 0 and center_perspective[1] < h:
                    draw_center_perspective(map_img,
                                            center_perspective,
                                            color=(80, 120, 220),
                                            thickness=-1,
                                            radius=2)
                    draw_corners_perspective(map_img,
                                             corners_perspective,
                                             color=(230, 45, 20),
                                             thickness=3
                                             )
    h, w, _ = map_img.shape
    ego_length = 4.3
    egeo_width = 1.7
    length = int(h / longitude * ego_length)
    width = int(w / lateral * egeo_width)
    x1, y1, x2, y2 = int(w/2 - width/2), int(h/2 - length/2), int(w/2 + width/2), int(h/2 + length/2)
    cv2.rectangle(map_img, (x1, y1), (x2, y2), color=(123, 212, 45), thickness=-1)
    cv2.imshow('map', map_img)


class NuscenesVisualizer(Dataset):
    def __init__(self, data_root, nusc, mode='train', with_depth=False):
        self.data_root = data_root
        self.nusc = nusc
        self.mode = mode
        self.samples_info = {}
        self.idx_2_scene_idx = []
        self.idx_2_sample_tokens = []
        self.timestamps = []
        self.scene_tokens = []

        target_scene = self.get_scenes()  # different between train and val
        for scene_idx, each_scene in enumerate(self.nusc.scene):
            scene_name = each_scene['name']
            if scene_name not in target_scene:
                continue
            # sample parse
            n_samples, cur_sample, last_sample = each_scene['nbr_samples'], each_scene['first_sample_token'], \
                                                 each_scene['last_sample_token']
            self.samples_info[scene_name] = n_samples
            temp = [scene_idx] * n_samples
            self.idx_2_scene_idx.extend(temp)

            while cur_sample != last_sample:
                self.idx_2_sample_tokens.append(cur_sample)
                sample = self.nusc.get('sample', cur_sample)
                self.timestamps.append(sample['timestamp'])
                self.scene_tokens.append(sample['scene_token'])
                cur_sample = sample['next']
            self.idx_2_sample_tokens.append(last_sample)
            sample = self.nusc.get('sample', last_sample)
            self.timestamps.append(sample['timestamp'])
            self.scene_tokens.append(sample['scene_token'])

        self.sort_samples()
        self.with_depth = with_depth

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.mode == 'train']

        scenes = create_splits_scenes()[split]

        return scenes

    def sort_samples(self):
        sample_contents = zip(self.idx_2_scene_idx, self.idx_2_sample_tokens, self.scene_tokens, self.timestamps)
        sample_contents = sorted(sample_contents, key=lambda x: (x[2], x[3]))
        sample_contents = zip(*sample_contents)
        self.idx_2_scene_idx, self.idx_2_sample_tokens, self.scene_tokens, self.timestamps = [list(x) for x in sample_contents]
        pass

    def __len__(self):
        return len(self.idx_2_sample_tokens)

    def __getitem__(self, idx):
        scene_idx = self.idx_2_scene_idx[idx]
        sample_token = self.idx_2_sample_tokens[idx]
        scene = self.nusc.scene[scene_idx]
        scene_name = scene['name']

        sample = self.nusc.get('sample', sample_token)

        timestamp = sample['timestamp']

        # 获取传感器信息 (camera  lidar  )
        sensor_names = list(sample['data'].keys())
        camera_names = [name for name in sensor_names if 'CAM' in name]
        camera_img_files = {}
        camera_calibration = {}
        ego_pose_for_each_sensor = {}
        camera_timestamp = {}
        samples = {}
        for sensor_name, sensor_token in sample['data'].items():
            sample_data = self.nusc.get('sample_data', sensor_token)

            calibrated_sensor_data = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            if 'CAM' in sensor_name:
                camera_timestamp[sensor_name] = sample_data['timestamp']
                img_file = os.path.join(self.data_root, sample_data['filename'])
                camera_img_files[sensor_name] = img_file
                camera_calibration[sensor_name] = {
                    'translation': calibrated_sensor_data['translation'],
                    'rotation': calibrated_sensor_data['rotation'],
                    'camera_intrinsic': calibrated_sensor_data['camera_intrinsic'],
                }
                ego_pose_for_each_sensor[sensor_name] = {
                    'translation': ego_pose['translation'],
                    'rotation': ego_pose['rotation'],
                }
        samples = {
            'scene_name': scene_name,
            'timestamp': timestamp,
            'camera_timestamp': camera_timestamp,
            'camera_img_files': camera_img_files,
            'camera_calibration': camera_calibration,
            'ego_pose': ego_pose_for_each_sensor,
        }

        # ann
        anns = []
        for ann_token in sample['anns']:
            ann_info = self.nusc.get('sample_annotation', ann_token)

            # 获取实例类别
            instance_token = ann_info['instance_token']
            instance = self.nusc.get('instance', instance_token)
            category = self.nusc.get('category', instance['category_token'])
            # category_name = ann_info['category_name']
            category_name = category['name']

            # 可见性
            visibility_token = ann_info['visibility_token']
            visibility = self.nusc.get('visibility', visibility_token)

            # 属性 (例如: 站立，行走)
            attribute_tokens = ann_info['attribute_tokens']
            attributes = []
            for attr_token in attribute_tokens:
                attribute = self.nusc.get('attribute', attr_token)
                attribute_ = {
                    'name': attribute['name'],
                    'description': attribute['description'],
                }
                attributes.append(attribute)

            # global frame
            global_translation = np.array(ann_info['translation'])
            global_quaternion = Quaternion(np.array(ann_info['rotation']))
            global_rotation = global_quaternion.rotation_matrix

            """
            每个摄像头的成像的时间戳不同，因此有不同的ego_pose (自车在全局坐标系下的)，
            因此转换到自身坐标系下时，同一个物体对于不同摄像头有不同的translation 和 rotation
            """
            ann = {
                'global_frame': {
                    'translation': ann_info['translation'],
                    'rotation': ann_info['rotation'],
                },
                'size': ann_info['size'],
                'category_name': category_name,
                'visibility': visibility,
                'attributes': attributes,
            }
            for camera_name in camera_names:
                ego_translation = np.array(samples['ego_pose'][camera_name]['translation'])
                ego_quaternion = Quaternion(np.array(samples['ego_pose'][camera_name]['rotation']))
                ego_rotation = ego_quaternion.rotation_matrix
                # global to ego frame
                translation = ego_rotation.transpose() @ (global_translation - ego_translation)
                quaternion = ego_quaternion.inverse * np.array(ann_info['rotation'])
                rotation = ego_rotation.transpose() @ global_rotation
                # assert b == rotation

                # ego to camera frame
                camera_translation = np.array(camera_calibration[camera_name]['translation'])
                camera_quaternion = Quaternion(np.array(camera_calibration[camera_name]['rotation']))
                camera_rotation = camera_quaternion.rotation_matrix
                translation_ = camera_rotation.transpose() @ (translation - camera_translation)
                quaternion_ = camera_quaternion.inverse * quaternion
                rotation_ = quaternion_.rotation_matrix

                # 获取角点的像素值
                box_corners = corners_3d_box(translation_, quaternion_, ann_info['size'])  # 相机坐标下
                box_corners_global = corners_3d_box(translation, quaternion, ann_info['size'])  # 全局坐标下

                intrinsic = np.array(camera_calibration[camera_name]['camera_intrinsic'])
                pixel_points = (intrinsic @ box_corners).transpose()
                pixel_points = pixel_points[:, :2] / pixel_points[:, 2][:, None]

                # 中心点像素值
                center = translation_
                pixel_center = (intrinsic @ center[:, None]).transpose()
                pixel_center = pixel_center[:, :2] / pixel_center[:, 2][:, None]

                ann[camera_name] = {
                    'ego_frame': {
                        'translation': translation.tolist(),
                        'rotation': quaternion_to_list(quaternion),
                        'corners': box_corners_global.transpose().tolist(),
                    },
                    'sensor_frame': {
                        'translation': translation_.tolist(),
                        'rotation': quaternion_to_list(quaternion_),
                        'corners': box_corners.transpose().tolist(),
                    },
                    'pixel_frame': {
                        'corners': pixel_points.tolist(),
                        'center': pixel_center[0].tolist(),
                    },
                }

            anns.append(ann)
        samples['anns'] = anns

        if self.with_depth:
            # get lidar point in ego frame
            pts = self.get_lidar_data(sample, 3)
            samples['lidar_pts'] = pts
        return samples

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                             nsweeps=nsweeps, min_distance=0.5)
        return pts


if __name__ == '__main__':
    def need_skip(samples):
        img_files = samples['camera_img_files']
        if os.path.exists(img_files['CAM_FRONT']):
            return False
        else:
            print('skip %s' % img_files['CAM_FRONT'])
            return True

    version = 'mini'
    dataroot = '/Volumes/KINGSTON0/nuscenes'
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)

    dataset = NuscenesVisualizer(os.path.join(dataroot, version),
                                 nusc,
                                 mode='val',
                                 with_depth=True)
    idx = 0
    while idx < len(dataset):
        samples = dataset[idx]

        if need_skip(samples):
            idx += 1
            continue

        # vis_3d_detection(samples)

        # vis_h = 500
        # vis_w = 500
        # longitude = 100
        # lateral = 100
        # # vis_bev(samples,
        # #         vis_h,
        # #         vis_w,
        # #         longitude,
        # #         lateral)
        # #
        # vis_3d_detection_with_map(nusc,
        #                           samples,
        #                           '/Volumes/KINGSTON0/nuscenes/nuScenes-map-expansion-v1.3',
        #                           vis_h,
        #                           vis_w,
        #                           longitude,
        #                           lateral)

        vis_lidar(samples)

        # ann_test_lidar(samples)

        key = cv2.waitKey(1)
        if key == ord('.'):
            idx += 1
        elif key == ord(','):
            idx -= 1
        # idx += 1
