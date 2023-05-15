import os
import json

import torch
import numpy as np
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.map_expansion.map_api import NuScenesMap
from tools.nuscene.utils import (
    get_nusc_maps,
    get_local_map
)
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
from functools import reduce
from tqdm import tqdm
from .base import BaseDataset
from ...util.geometry_utils import (
    corners_3d_box,
    quaternion_to_list,
)
from .load_groudtruth.nuscenes_gt import (
    load_boxes3d_groundtruth,
    load_map_groundtruth
)
from copy import deepcopy
from PIL import Image
from nuscenes.utils.data_classes import Box
from ..collate import collate_function

class NuScenesDataset(BaseDataset):

    def __init__(self,
                 data_root,
                 target_cameras,
                 version,
                 input_size,
                 with_depth=False,
                 nsweeps=1,
                 min_distance=0.0,
                 mode='train',
                 map_folder=None,
                 map_range=None,
                 map_extractor=[],
                 data_balance=False,
                 load_data_balance_cache=False,
                 multi_frame=False,
                 **kwargs):
        assert version in ['mini', 'trainval']
        assert mode in ['train', 'val']
        self.version = version
        self.mode = mode
        super().__init__(version=version,
                         data_root=data_root,
                         input_size=input_size,
                         with_depth=with_depth,
                         mode=mode,
                         **kwargs)
        self.nusc = NuScenes(version='v1.0-{}'.format(version),
                             dataroot=os.path.join(data_root, version),
                             verbose=False)
        self.target_cameras = target_cameras
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

        self.min_distance = min_distance
        # self.sort_samples()

        # build maps
        self.map_folder = map_folder
        self.map_range = map_range
        self.map_extractors = map_extractor
        self.nusc_maps = None
        if self.map_folder and len(self.map_extractors) > 0:
            self.nusc_maps = get_nusc_maps(map_folder)
            self.scene2map = {}
            for rec in self.nusc.scene:
                log = self.nusc.get('log', rec['log_token'])
                self.scene2map[rec['name']] = log['location']

        self.data_info = []

        # parse class info
        self.data_balance = data_balance
        self.load_cache_data_balance = load_data_balance_cache
        if self.data_balance:
            self.sample_idxs = self.cbgs_strategy()
            # self.data_balance_strategy()

        self.use_multi_frame = multi_frame

    def cbgs_strategy(self):
        ignore_cls_names = ['animal', 'human.pedestrian.personal_mobility',
                            'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
                            'movable_object.debris', 'movable_object.pushable_pullable',
                            'static_object.bicycle_rack', 'vehicle.emergency.ambulance',
                            'vehicle.emergency.police']

        cls_maps = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        cat2ids = set(cls_maps.values())
        class_sample_idxs = {cat_id: [] for cat_id in cat2ids}
        for idx in range(len(self.idx_2_sample_tokens)):
            scene_idx = self.idx_2_scene_idx[idx]
            sample_token = self.idx_2_sample_tokens[idx]
            scene = self.nusc.scene[scene_idx]
            scene_name = scene['name']
            sample = self.nusc.get('sample', sample_token)

            sample_rec = self.nusc.get('sample', sample_token)
            # sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            # pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
            #
            meet_sample = False

            sub_cls_names = [self.nusc.get('sample_annotation', ann_token)['category_name'] for ann_token in
                             sample['anns']]
            sub_cls_names = set(sub_cls_names)
            for cls_name in sub_cls_names:
                if cls_name in ignore_cls_names:
                    continue
                class_sample_idxs[cls_maps[cls_name]].append(idx)
            # for ann_token in sample['anns']:
            #     ann_info = self.nusc.get('sample_annotation', ann_token)
            #     # if ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] == 0:
            #     #     continue
            #     cls_name = ann_info['category_name']
            #     if cls_name in ignore_cls_names:
            #         continue
            #     # x, y = ann_info['translation'][0] - pose_record['translation'][0], \
            #     #        ann_info['translation'][1] - pose_record['translation'][1]
            #     # if x <= -51.2 or x >= 51.2 or y <= -51.2 or y >= 51.2:
            #     #     continue
            #     if idx not in class_sample_idxs[cls_maps[cls_name]]:
            #         class_sample_idxs[cls_maps[cls_name]].append(idx)

        duplicated_samples = sum(
            [len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples
            for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(cat2ids)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(cls_inds,
                                               int(len(cls_inds) *
                                                   ratio)).tolist()
        return sample_indices

    def data_balance_strategy(self):
        # raise NotImplementedError  # 弃用，使用cbgs代替
        import os
        import json
        flag_version = 'v0.1_202201051509'
        flag_save_name = "nuscenes_data_balance_flag_%s.json" % (flag_version)
        flag_savr_dir = "/cfs/cfs-j898vwqh/lvanyang/model/nuscenes"
        if os.path.exists(os.path.join(flag_savr_dir, flag_save_name)) and self.load_cache_data_balance:
            self.logger.info("load data balance flags from cache %s" % (os.path.join(flag_savr_dir, flag_save_name)))
            self.flags = json.load(open(os.path.join(flag_savr_dir, flag_save_name), 'r'))
            return

        flag_maps = {
            'movable_object.barrier': 1,
            'vehicle.bicycle': 2,
            'vehicle.bus.bendy': 3,
            'vehicle.bus.rigid': 3,
            'vehicle.car': 4,
            'vehicle.construction': 5,
            'vehicle.motorcycle': 6,
            'human.pedestrian.adult': 7,
            'human.pedestrian.child': 7,
            'human.pedestrian.construction_worker': 7,
            'human.pedestrian.police_officer': 7,
            'movable_object.trafficcone': 8,
            'vehicle.trailer': 9,
            'vehicle.truck': 10,
        }
        cls_maps = {
            'movable_object.barrier': 'barrier',
            'vehicle.bicycle': 'bicycle',
            'vehicle.bus.bendy': 'bus',
            'vehicle.bus.rigid': 'bus',
            'vehicle.car': 'car',
            'vehicle.construction': 'construction_vehicle',
            'vehicle.motorcycle': 'motorcycle',
            'human.pedestrian.adult': 'pedestrian',
            'human.pedestrian.child': 'pedestrian',
            'human.pedestrian.construction_worker': 'pedestrian',
            'human.pedestrian.police_officer': 'pedestrian',
            'movable_object.trafficcone': 'traffic_cone',
            'vehicle.trailer': 'trailer',
            'vehicle.truck': 'truck'
        }
        cls_infos = {}
        flag_infos = {}
        flag_samples = {}
        for idx in range(len(self.idx_2_sample_tokens)):
            scene_idx = self.idx_2_scene_idx[idx]
            sample_token = self.idx_2_sample_tokens[idx]
            scene = self.nusc.scene[scene_idx]
            scene_name = scene['name']
            sample = self.nusc.get('sample', sample_token)

            sample_rec = self.nusc.get('sample', sample_token)
            sd_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            pose_record = self.nusc.get('ego_pose', sd_record['ego_pose_token'])

            ignore_cls_names = ['animal', 'human.pedestrian.personal_mobility',
                                'human.pedestrian.stroller', 'human.pedestrian.wheelchair',
                                'movable_object.debris', 'movable_object.pushable_pullable',
                                'static_object.bicycle_rack', 'vehicle.emergency.ambulance',
                                'vehicle.emergency.police']
            cls_infos[sample_token] = []
            flag_infos[sample_token] = []
            meet_sample = False
            for ann_token in sample['anns']:
                ann_info = self.nusc.get('sample_annotation', ann_token)
                if ann_info['num_lidar_pts'] + ann_info['num_radar_pts'] == 0:
                    continue
                cls_name = ann_info['category_name']
                if cls_name in ignore_cls_names:
                    continue
                x, y = ann_info['translation'][0] - pose_record['translation'][0], \
                       ann_info['translation'][1] - pose_record['translation'][1]
                if x <= -51.2 or x >= 51.2 or y <= -51.2 or y >= 51.2:
                    continue
                cls_infos[sample_token].append(cls_maps[cls_name])
                flag_infos[sample_token].append(flag_maps[cls_name])
                if flag_maps[cls_name] not in flag_samples:
                    flag_samples[flag_maps[cls_name]] = []
                if sample_token not in flag_samples[flag_maps[cls_name]]:
                    meet_sample = True
                    flag_samples[flag_maps[cls_name]].append(sample_token)
            if not meet_sample:
                cls_infos[sample_token] = ['no_targets']
                flag_infos[sample_token] = [1]
                # 该sample不属于任何一个flag, 默认放在flag 1
                if 1 not in flag_samples:
                    flag_samples[1] = []
                flag_samples[1].append(sample_token)
            flag_infos[sample_token] = list(set(flag_infos[sample_token]))

        samples = []
        for i, val in flag_samples.items():
            samples.extend(val)
        samples = set(samples)
        assert len(samples) == len(self.idx_2_sample_tokens)

        flag_keys = list(flag_samples.keys())
        flag_keys = sorted(flag_keys, key=lambda x: len(flag_samples[x]))
        rate = 1 / len(list(set(flag_maps.values())))
        average_num = rate * len(self.idx_2_sample_tokens)
        import random
        random.seed(0)
        abandom_samples = []
        rest_n = len(self.idx_2_sample_tokens)
        for flag in flag_keys:
            sample_tokens = flag_samples[flag]
            sample_tokens = sorted(sample_tokens, key=lambda x: len(flag_infos[x]))
            keep = []  # 留下的sample
            total_size = len(sample_tokens)
            rest_rate = average_num / total_size
            sing_flag_samples_count = 0
            for sample_token in sample_tokens:
                # 决定留下或者扔掉
                # if random.uniform(0, 1) < (average_num - len(keep)) / rest_n:
                if random.uniform(0, 1) < (average_num - sing_flag_samples_count) / total_size:
                    # 留下
                    if sample_token in abandom_samples:
                        abandom_samples.pop(abandom_samples.index(sample_token))

                    keep.append(sample_token)
                    for other_flag in flag_keys:
                        if other_flag == flag:
                            continue
                        other_flag_samples = flag_samples[other_flag]
                        if sample_token in other_flag_samples:
                            other_flag_samples.pop(other_flag_samples.index(sample_token))
                    rest_n -= 1
                else:
                    # 丢弃，看下有没有其他任务有，有则不管，无则不能丢弃
                    flags = flag_infos[sample_token]
                    if len(flags) == 1 and flags[0] == flag:
                        keep.append(sample_token)
                        sing_flag_samples_count += 1
                        rest_n -= 1
                    elif len(flags) > 1:
                        if sample_token not in abandom_samples:
                            abandom_samples.append(sample_token)
                    else:
                        raise ValueError

            flag_samples[flag] = keep

        for abandom_sample in abandom_samples:
            flags = flag_samples.keys()
            flags = sorted(flags, key=lambda x: len(flag_samples[x]))

            for flag in flags:
                if flag in flag_infos[abandom_sample]:
                    flag_samples[flag].append(abandom_sample)
                    break

        flags_num = [len(samples) for key, samples in flag_samples.items()]
        flags_num = sum(flags_num)
        assert flags_num == len(self.idx_2_sample_tokens)

        flag_infos = {}
        for flag, samples in flag_samples.items():
            for sample in samples:
                flag_infos[sample] = flag

        def log_each_class_nums(cls_info):
            cls_nums = {}
            for sample_token, cls_names in cls_info.items():
                for cls_name in cls_names:
                    if cls_name not in cls_nums:
                        cls_nums[cls_name] = 1
                    else:
                        cls_nums[cls_name] += 1
            for cls_name, cls_num in cls_nums.items():
                self.logger.info("nums: %s -> %d" % (cls_name, cls_num))

        def log_each_class_cover(cls_info):
            cls_cover = {}
            for sample_token, cls_names in cls_info.items():
                cls_name_set = set(cls_names) if isinstance(cls_names, list) else [cls_names]
                for cls_name in cls_name_set:
                    if cls_name not in cls_cover:
                        cls_cover[cls_name] = 1
                    else:
                        cls_cover[cls_name] += 1
            for cls_name, cover_num in cls_cover.items():
                self.logger.info("cover: %s -> %f" % (cls_name, cover_num / len(cls_info)))
                pass

        log_each_class_cover(flag_infos)
        log_each_class_cover(cls_infos)
        log_each_class_nums(cls_infos)

        self.flags = []
        for sample_token in self.idx_2_sample_tokens:
            self.flags.append(flag_infos[sample_token])

        if not os.path.exists(os.path.join(flag_savr_dir, flag_save_name)) and self.load_cache_data_balance:
            self.logger.info('save data balance to %s' % (os.path.join(flag_savr_dir, flag_save_name)))
            file = open(os.path.join(flag_savr_dir, flag_save_name), 'w')
            json.dump(self.flags, file)

    def sort_samples(self):
        sample_contents = zip(self.idx_2_scene_idx, self.idx_2_sample_tokens, self.scene_tokens, self.timestamps)
        sample_contents = sorted(sample_contents, key=lambda x: (x[2], x[3]))
        sample_contents = zip(*sample_contents)
        self.idx_2_scene_idx, self.idx_2_sample_tokens, self.scene_tokens, self.timestamps = [list(x) for x in
                                                                                              sample_contents]
        pass

    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.mode == 'train']

        scenes = create_splits_scenes()[split]

        return scenes


    def get_sample(self, idx, key_idx_bia=0, is_key_frame=True):
        # print('flags:', self.flags[idx])
        if self.data_balance:
            idx = self.sample_idxs[idx]
        raw_idx = idx
        idx = idx + key_idx_bia
        if idx < 0:
            idx = raw_idx
        elif self.scene_tokens[raw_idx] != self.scene_tokens[idx]:
            idx = raw_idx
        scene_idx = self.idx_2_scene_idx[idx]
        sample_token = self.idx_2_sample_tokens[idx]
        scene = self.nusc.scene[scene_idx]
        scene_name = scene['name']

        sample = self.nusc.get('sample', sample_token)

        timestamp = sample['timestamp']

        # 获取传感器信息 (camera  lidar  )
        sensor_names = list(sample['data'].keys())
        camera_names = [name for name in sensor_names if 'CAM' in name]
        lidar_names = [name for name in sensor_names if 'LIDAR' in name]
        camera_img_files = {}
        camera_calibration = {}
        lidar_calibration = {}
        ego_pose_for_each_sensor = {}
        camera_timestamp = {}
        lidar_timestamp = {}
        sensor_names = self.target_cameras + ['LIDAR_TOP']
        for sensor_name in sensor_names:
            sensor_token = sample['data'][sensor_name]
            sample_data = self.nusc.get('sample_data', sensor_token)

            calibrated_sensor_data = self.nusc.get('calibrated_sensor', sample_data['calibrated_sensor_token'])
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            if 'CAM' in sensor_name:
                camera_timestamp[sensor_name] = sample_data['timestamp']
                img_file = os.path.join(self.data_root, self.version, sample_data['filename'])
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
            if 'LIDAR' in sensor_name:
                lidar_timestamp[sensor_name] = sample_data['timestamp']
                lidar_calibration[sensor_name] = {
                    'translation': calibrated_sensor_data['translation'],
                    'rotation': calibrated_sensor_data['rotation'],
                }
                ego_pose_for_each_sensor[sensor_name] = {
                    'translation': ego_pose['translation'],
                    'rotation': ego_pose['rotation'],
                }

        samples = {
            'sample_token': sample['token'],
            'scene_name': scene_name,
            'timestamp': timestamp,
            'camera_timestamp': camera_timestamp,
            'camera_img_files': camera_img_files,
            'camera_calibration': camera_calibration,
            'lidar_timestamp': lidar_timestamp,
            'lidar_calibration': lidar_calibration,
            'ego_pose': ego_pose_for_each_sensor,
        }

        if not is_key_frame:
            return samples
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

            velocity = self.nusc.box_velocity(ann_token)
            if np.any(np.isnan(velocity)):
                velocity = np.zeros(3)

            """
            每个摄像头的成像的时间戳不同，因此有不同的ego_pose (自车在全局坐标系下的)，
            因此转换到自身坐标系下时，同一个物体对于不同摄像头有不同的translation 和 rotation
            """
            ann = {
                'global_frame': {
                    'translation': np.array(ann_info['translation'], dtype=np.float32),
                    'rotation': np.array(ann_info['rotation'], dtype=np.float32),
                },
                'size': np.array(ann_info['size'], dtype=np.float32),
                # 'velocity': np.array(box_velocity, dtype=np.float32),
                'category_name': category_name,
                'visibility': visibility,
                'attributes': attributes,
                'num_pts': ann_info['num_lidar_pts'] + ann_info['num_radar_pts'],
            }
            target_sensor_names = camera_names + lidar_names
            for sensor_name in target_sensor_names:
                box = Box(ann_info['translation'],
                          ann_info['size'],
                          global_quaternion,
                          velocity=velocity)

                ego_translation = np.array(samples['ego_pose'][sensor_name]['translation'])
                ego_quaternion = Quaternion(np.array(samples['ego_pose'][sensor_name]['rotation']))
                ego_rotation = ego_quaternion.rotation_matrix
                # global to ego frame
                box.translate(-ego_translation)
                box.rotate(ego_quaternion.inverse)
                translation = box.center
                quaternion = box.orientation
                # rotation = ego_rotation.transpose() @ global_rotation
                # assert b == rotation

                # ego to camera frame
                calibrations = camera_calibration if 'CAM' in sensor_name else lidar_calibration
                sensor_translation = np.array(calibrations[sensor_name]['translation'])
                sensor_quaternion = Quaternion(np.array(calibrations[sensor_name]['rotation']))
                camera_rotation = sensor_quaternion.rotation_matrix
                translation_ = camera_rotation.transpose() @ (translation - sensor_translation)
                quaternion_ = sensor_quaternion.inverse * quaternion
                # rotation_ = quaternion_.rotation_matrix

                # 获取角点的像素值
                box_corners = corners_3d_box(translation_, quaternion_, ann_info['size'])  # 相机坐标下
                box_corners_global = corners_3d_box(translation, quaternion, ann_info['size'])  # 全局坐标下

                if 'CAM' in sensor_name:
                    # 只有摄像头才有像素坐标下的位置
                    intrinsic = np.array(calibrations[sensor_name]['camera_intrinsic'])
                    pixel_points = (intrinsic @ box_corners).transpose()
                    pixel_points = pixel_points[:, :2] / pixel_points[:, 2][:, None]

                    # 中心点像素值
                    center = translation_
                    pixel_center = (intrinsic @ center[:, None]).transpose()
                    pixel_center = pixel_center[:, :2] / pixel_center[:, 2][:, None]
                    pixel_points = pixel_points.tolist()
                    pixel_center = pixel_center[0].tolist()

                else:
                    pixel_center = [-1., -1.]
                    pixel_points = [-1., -1.] * 8

                ann[sensor_name] = {
                    'ego_frame': {
                        'translation': np.array(translation.tolist(), dtype=np.float32),
                        'rotation': np.array(quaternion_to_list(quaternion), dtype=np.float32),
                        'corners': np.array(box_corners_global.transpose().tolist(), dtype=np.float32),
                        'velocity': np.array(box.velocity[:2], dtype=np.float32)
                    },
                    'sensor_frame': {
                        'translation': np.array(translation_.tolist(), dtype=np.float32),
                        'rotation': np.array(quaternion_to_list(quaternion_), dtype=np.float32),
                        'corners': np.array(box_corners.transpose().tolist(), dtype=np.float32),
                    },
                    'pixel_frame': {
                        'corners': np.array(pixel_points, dtype=np.float32),
                        'center': np.array(pixel_center, dtype=np.float32),
                    },
                }

            anns.append(ann)
        samples['anns'] = anns

        if self.nusc_maps is not None:
            map_data = self.get_map_data(samples['ego_pose']['LIDAR_TOP'],
                                         self.nusc_maps[self.scene2map[samples['scene_name']]],
                                         self.map_range,
                                         self.map_extractors)
            samples.update(map_data)
        return samples

    def get_img_depth(self, lidar_pts, img_size, ego_poses, cam_calibs):
        """
        Args:
            lidar_pts: ego frame下点云
        """
        width, height = img_size
        ego_to_global_trans_lidar_frame = np.array(ego_poses['LIDAR_TOP']['translation'])
        ego_to_global_rot_lidar_frame = Quaternion(np.array(ego_poses['LIDAR_TOP']['rotation']))
        ego_to_global_lidar_frame = transform_matrix(ego_to_global_trans_lidar_frame, ego_to_global_rot_lidar_frame)

        res = {
            'depth_imgs': {}
        }
        for camera_type, cam_calib in cam_calibs.items():
            cam_to_ego_trans = np.array(cam_calib['translation'])
            cam_to_ego_rot = Quaternion(np.array(cam_calib['rotation']))
            ego_to_cam = transform_matrix(cam_to_ego_trans, cam_to_ego_rot, inverse=True)
            cam_intrinsic = np.array(cam_calib['camera_intrinsic'])

            ego_to_global_trans_cam_frame = np.array(ego_poses[camera_type]['translation'])
            ego_to_global_rot_cam_frame = Quaternion(np.array(ego_poses[camera_type]['rotation']))
            global_to_ego_cam_frame = transform_matrix(ego_to_global_trans_cam_frame, ego_to_global_rot_cam_frame,
                                                       inverse=True)

            # transform matrix which translates point (ego coord) in lidar frame to point (camera coord) in camera frame
            homo = ego_to_cam @ global_to_ego_cam_frame @ ego_to_global_lidar_frame
            pts = homo.dot(np.vstack((lidar_pts[:3, :], np.ones(lidar_pts.shape[1]))))[:3, :]

            depths = pts[2, :]
            coloring = depths

            points = view_points(pts[:3, :], cam_intrinsic, normalize=True)

            # Remove points that are either outside or behind the camera.
            # Leave a margin of 1 pixel for aesthetic reasons. Also make
            # sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera casing for non-keyframes
            # which are slightly out of sync.
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > self.min_distance)
            mask = np.logical_and(mask, points[0, :] > 1)
            mask = np.logical_and(mask, points[0, :] < width - 1)
            mask = np.logical_and(mask, points[1, :] > 1)
            mask = np.logical_and(mask, points[1, :] < height - 1)
            points = points[:, mask]
            coloring = coloring[mask]

            point_depth = np.concatenate([points[:2, :].T, coloring[:, None]],
                                         axis=1).astype(np.float32)
            res['depth_imgs'][camera_type] = point_depth
        return res

    def get_map_data(self, ego_pose, map, map_range, map_extractors):
        rot = Quaternion(ego_pose['rotation']).rotation_matrix
        rot = np.arctan2(rot[1, 0], rot[0, 0])
        center = np.array([ego_pose['translation'][0], ego_pose['translation'][1], np.cos(rot), np.sin(rot)])

        poly_names = self.get_ploy_names(map_extractors)
        line_names = self.get_line_names(map_extractors)
        lmap = get_local_map(map, center,
                             [map_range[0], map_range[1]], poly_names, line_names)
        return lmap

    def get_ploy_names(self, map_extractors):
        ploy_types = ['drivable_area', 'road_segment', 'road_block', 'lane', 'ped_crossing',
                      'walkway', 'stop_line', 'carpark_area', 'traffic_light']
        targets = []
        for ploy_type in ploy_types:
            if ploy_type in map_extractors:
                targets.append(ploy_type)
        return targets

    def get_line_names(self, map_extractors):
        line_types = ['road_divider', 'lane_divider']
        targets = []
        for line_type in line_types:
            if line_type in map_extractors:
                targets.append(line_type)
        return targets

    def get_lidar_data(self, sample_rec, nsweeps=1.0, min_distance=0.):
        """ 返回sample_rec帧下ego坐标系下的点云坐标
        Args:
            sample_rec: 指定时刻下的sample
            nsweeps: 融合过去指定数量帧下的lidar point到当前帧
            min_distance: 过滤离lidar top中心指定距离内的点云
        """
        points = np.zeros((5, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data']['LIDAR_TOP']
        ref_sd_rec = self.nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = self.nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        # ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transformation matrix from global to _current_ ego car frame. (global to cur ego)
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data']['LIDAR_TOP']
        current_sd_rec = self.nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = LidarPointCloud.from_file(os.path.join(self.nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose. (past ego to global)
            current_pose_rec = self.nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame. (sensor to past ego)
            current_cs_rec = self.nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
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
                current_sd_rec = self.nusc.get('sample_data', current_sd_rec['prev'])

        return points

    def cache_data(self, rank_indexes):
        raise NotImplementedError  # 弃用
        data_info = [None] * len(self.idx_2_sample_tokens)

        if self.mode == 'train':
            bar = tqdm(rank_indexes)
            for idx in bar:
                bar.set_description('Getting data info')
                sample = self.get_sample(idx)
                data_info[idx] = sample
            self.data_info = data_info

    def get_data_info(self, ann_path=None, source_ann_path=None, target_ann_path=None):
        # 弃用
        # data_info = []
        # if not hasattr(self, 'idx_2_sample_tokens'):
        #     # 基类里有一次调用，此时nuscensdataset还没初始化完成，故先返回
        #     return data_info
        #
        # # if self.mode == 'train':
        # #     bar = tqdm(range(len(self.idx_2_sample_tokens)))
        # #     for idx in bar:
        # #         bar.set_description('Getting data info')
        # #         sample = self.get_sample(idx)
        # #         data_info.append(sample)
        #
        pass

    def load_all_masks(self, data_info):
        for img_info in data_info:
            img_info['img_loaded'] = False
        return data_info

    def get_task_flags(self):
        task_flags = []
        if self.mode == 'train':
            for img_info in self.data_info:
                task_flags.append(img_info['task_id'])
        else:
            task_flags = [0 for i in range(len(self.data_info))]
        return np.array(task_flags)


    def combine_multi_frame_infos(self, key_info, other_infos):
        key_ego_pose = key_info['ego_pose']
        # ego_pose_standard = 'LIDAR_TOP'
        for other_info in other_infos:
            # 'camera_calibration' 'ego_pose'
            for camera_type, calibration in other_info['camera_calibration'].items():
                key_ego_2_global_trans = np.array(key_ego_pose[camera_type]['translation'])
                key_ego_2_global_rots = Quaternion(np.array(key_ego_pose[camera_type]['rotation']))
                global_2_key_ego = transform_matrix(key_ego_2_global_trans, key_ego_2_global_rots, inverse=True)

                sensor_2_sweep_ego_trans = np.array(calibration['translation'])
                sensor_2_sweep_ego_rots = Quaternion(np.array(calibration['rotation']))
                sensor_2_sweep_ego = transform_matrix(sensor_2_sweep_ego_trans, sensor_2_sweep_ego_rots)

                sweep_ego_2_global_trans = np.array(other_info['ego_pose'][camera_type]['translation'])
                sweep_ego_2_global_rots = Quaternion(np.array(other_info['ego_pose'][camera_type]['rotation']))
                sweep_ego_2_global = transform_matrix(sweep_ego_2_global_trans, sweep_ego_2_global_rots)

                sensor_2_key_ego = global_2_key_ego @ sweep_ego_2_global @ sensor_2_sweep_ego
                sensor_2_key_ego_trans = sensor_2_key_ego[:, 3][:3].tolist()
                sensor_2_key_ego_rots = R.from_matrix(sensor_2_key_ego[:3, :3])
                x, y, z, w = sensor_2_key_ego_rots.as_quat()
                other_info['camera_calibration'][camera_type]['translation'] = sensor_2_key_ego_trans
                other_info['camera_calibration'][camera_type]['rotation'] = [w, x, y, z]
                pass

        ann_infos = collate_function([key_info]+other_infos)
        ann_infos['anns'] = ann_infos['anns'][0]
        if 'drivable_area' in ann_infos:
            ann_infos['drivable_area'] = ann_infos['drivable_area'][0]
        if 'road_divider' in ann_infos:
            ann_infos['road_divider'] = ann_infos['road_divider'][0]
        if 'lane_divider' in ann_infos:
            ann_infos['lane_divider'] = ann_infos['lane_divider'][0]
        return ann_infos

    def get_train_data(self, idx):
        """
        Load image and annotation
        :param idx:
        :return: meta-data (a dict containing image, annotation and other information)
        """
        # idx = 10   # for debug
        multi_frame = self.use_multi_frame
        ann_info = self.get_sample(idx, key_idx_bia=0, is_key_frame=True)
        last_frame_info = {} if not multi_frame else self.get_sample(idx, key_idx_bia=-1, is_key_frame=False)

        # read img
        imgs = {}
        img_read_func = cv2.imread if self.img_read == 'cv2' else Image.open
        for camera, img_file in ann_info['camera_img_files'].items():
            imgs[camera] = [img_read_func(img_file)] if multi_frame else img_read_func(img_file)

        if last_frame_info:
            for camera, img_file in last_frame_info['camera_img_files'].items():
                imgs[camera].append(img_read_func(img_file))

        if len(imgs.keys()) == 0:
            print('image {} read failed at %s %s'.format(ann_info['scene_name'], ann_info['timestamp']))
            raise FileNotFoundError('Cant load image! Please check image path!')

        # get depth
        img_depths = {}
        if self.with_depth:
            # get lidar point in ego frame
            sample = self.nusc.get('sample', ann_info['sample_token'])
            pts = self.get_lidar_data(sample, 1, self.min_distance)
            res = self.get_img_depth(pts, (1600, 900), ann_info['ego_pose'], ann_info['camera_calibration'])
            img_depths.update(res)

        if multi_frame:
            ann_info = self.combine_multi_frame_infos(ann_info, [last_frame_info])
        if self.mode == 'train':
            meta = dict(imgs=imgs, ann_info=ann_info)
        else:
            # meta = dict(imgs=imgs, raw_imgs=deepcopy(imgs), ann_info=ann_info)
            meta = dict(imgs=imgs, ann_info=ann_info)
        meta.update(img_depths)
        meta = self.pipeline(meta, self.input_size)
        meta = load_boxes3d_groundtruth(meta)
        meta = load_map_groundtruth(meta,
                                    ploy_names=self.get_ploy_names(self.map_extractors),
                                    line_names=self.get_line_names(self.map_extractors))

        # for debug
        # self.vis_bev_imgs(meta, waitkey=False)
        # self.vis_boxes3d_in_imgs(meta, waitkey=True)

        meta = self.concat_multi_camera_images(meta)
        if multi_frame:
            meta['imgs'] = torch.from_numpy(meta['imgs'].transpose(1, 0, 4, 2, 3))
        else:
            meta['imgs'] = torch.from_numpy(meta['imgs'].transpose(0, 3, 1, 2))

        return meta

    def vis_boxes3d_in_imgs(self, meta, img_key, warp_matrix_key, waitkey, vis_boxes3d):
        from tools.nuscene.data_check import is_box_in_image, is_target_cls, corners_3d_box, draw_corners
        from pyquaternion import Quaternion
        vis_targets = ['vehicle', 'human']

        copy_meta = {img_key: {}}
        for camera_type, img in meta[img_key].items():
            camera_translation = np.array(meta['ann_info']['camera_calibration'][camera_type]['translation'])
            camera_quaternion = Quaternion(np.array(meta['ann_info']['camera_calibration'][camera_type]['rotation']))
            camera_rotation = camera_quaternion.rotation_matrix
            camera_intrinsic = meta['ann_info']['camera_calibration'][camera_type]['camera_intrinsic']
            warp_matrix = meta[warp_matrix_key][camera_type]
            img = meta[img_key][camera_type].astype(np.uint8).copy()
            if not vis_boxes3d:
                copy_meta[img_key][camera_type] = img.astype(np.float32)
                continue
            for box3d in meta['boxes3d']:
                if not is_target_cls(box3d['category_name'], vis_targets):
                    continue
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
                    draw_corners(img, pixel_points.astype(np.int32), (200, 123, 180), 1)

            copy_meta[img_key][camera_type] = img.astype(np.float32)
        show = self.vis_camera_imgs(copy_meta, img_key, waitkey)

        return show

    def vis_bev_imgs(self, meta, img_key, warp_matrix_key, waitkey):
        from tools.nuscene.data_check import concat_perspective_imgs
        def get_z_homograph_matrix(translation, orientation, camera_intrinsic, warp_matrix, z=0.):
            """获取图像坐标系到指定z平面世界坐标系的homograph矩阵
            后续：point_world (X, Y, Z=z) = H @ point_img
            """
            K = np.array(camera_intrinsic)
            R = Quaternion(np.array(orientation)).rotation_matrix
            T = np.array(translation)
            r = K @ R.transpose()
            t = -r @ T.transpose()
            r = warp_matrix @ r
            t = warp_matrix @ t
            r[..., -1] = r[..., -1] * z + t
            H = r
            H = np.linalg.inv(H)
            return H

        longitude = 80
        lateral = 20
        longitude /= 2
        lateral /= 2
        vis_img_width = 500
        vis_img_height = 1000
        src_points = np.array(
            [[longitude, lateral], [longitude, -lateral], [-longitude, -lateral], [-longitude, lateral]],
            dtype="float32")
        dst_points = np.array([[0., 0.], [vis_img_width, 0.], [vis_img_width, vis_img_height], [0., vis_img_height]],
                              dtype="float32")
        M = cv2.getPerspectiveTransform(src_points, dst_points)

        perspective_imgs = {}
        for camera_type, img in meta[img_key].items():
            img = img.astype(np.uint8)
            camera_calibration = meta['ann_info']['camera_calibration'][camera_type]
            warp_matrix = meta[warp_matrix_key][camera_type]
            homograph_matrix = get_z_homograph_matrix(camera_calibration['translation'],
                                                      camera_calibration['rotation'],
                                                      camera_calibration['camera_intrinsic'],
                                                      warp_matrix,
                                                      z=0.2)

            # 消逝点以上的数据会被投影到对称区域，先置0
            half_img = img.copy()
            h, w, _ = img.shape
            half_img[: 180 * h // 350, ...] = 0

            # 通过M和homograph_matrix投影到假想的图像平面
            perspective_img = cv2.warpPerspective(half_img, M @ homograph_matrix, (vis_img_width, vis_img_height))
            perspective_imgs[camera_type] = perspective_img

        p_img = concat_perspective_imgs(perspective_imgs)
        cv2.imshow('bev', p_img)
        if waitkey:
            cv2.waitKey(0)

    def vis_camera_imgs(self, meta, img_key, waitKey=False):
        def concat_camera_imgs(imgs):
            row1 = np.concatenate([imgs['CAM_FRONT_LEFT'], imgs['CAM_FRONT'], imgs['CAM_FRONT_RIGHT']], axis=1)
            row2 = np.concatenate([imgs['CAM_BACK_LEFT'], imgs['CAM_BACK'], imgs['CAM_BACK_RIGHT']], axis=1)
            all = np.concatenate([row1, row2], axis=0)
            return all

        def show_img(win_name, img, wait_key, width=1800):
            h, w, _ = img.shape
            ratio = width / w
            img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
            cv2.imshow(win_name, img)
            if wait_key:
                return cv2.waitKey(0)

        img = concat_camera_imgs(meta[img_key])
        show_img('boxes3d', img.astype(np.uint8), waitKey)
        return img.astype(np.uint8)

    def concat_multi_camera_images(self, meta):
        imgs = []
        camera_types = []
        for camera_type, img in meta['imgs'].items():
            if camera_type not in self.target_cameras:
                continue
            camera_types.append(camera_type)
            if isinstance(img, list):
                imgs.append(np.array(img))
            else:
                imgs.append(img)
        imgs = np.array(imgs, dtype=np.float32)
        meta['imgs'] = imgs
        meta['camera_types'] = camera_types
        return meta

    def get_val_data(self, idx):
        """
        Currently no difference from get_train_data.
        Not support TTA(testing time augmentation) yet.
        :param idx:
        :return:
        """
        # TODO: support TTA
        return self.get_train_data(idx)

    def __len__(self):
        return len(self.idx_2_sample_tokens) if not self.data_balance else len(self.sample_idxs)
