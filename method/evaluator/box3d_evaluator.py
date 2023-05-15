import logging
from nuscenes.nuscenes import NuScenes
import os
import json
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.evaluate import DetectionEval
import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

logger = logging.getLogger("ADMultiTaskPerceptionSDK")


class Box3dEvaluator:
    """
    适用于nuscenes 3d检测的评测
    """

    def __init__(self, head_kwargs, **kwargs):
        self.head_names = [head_name for head_name in list(head_kwargs.keys()) if 'head' in head_name]
        self.datasets = [head_kwargs[head].dataset for head in self.head_names]
        self.data_roots = [head_kwargs[head].data_root for head in self.head_names]
        self.splits = [head_kwargs[head].split for head in self.head_names]
        self.versions = [head_kwargs[head].version for head in self.head_names]
        self.meta = {
            'meta': {
                "use_camera": True,
                "use_lidar": False,
                'use_radar': False,
                "use_map": False,
                'use_external': False,
            }
        }

    def evaluate(self, results, save_dir, rank=-1):
        """
        Args:
            results: 多任务网络输出结果 (含有真值标签)
            save_dir:
            rank:
        Returns:
        """
        sensor_standard = 'LIDAR_TOP'
        metrics = {}

        # 1. combine prediction results from different heads
        outs = {'results': {}}
        for head_name, dataset, version, dataroot, split in zip(self.head_names, self.datasets, self.versions,
                                                                self.data_roots, self.splits):
            if dataset == 'NuScenes':
                if not hasattr(self, 'nusc'):
                    logger.info("init nusc in elevation first...")
                    self.nusc = NuScenes(version='v1.0-{}'.format(version),
                                         dataroot=os.path.join(dataroot, version),
                                         verbose=False)
                if not hasattr(self, 'det_cfg'):
                    self.det_cfg = config_factory("detection_cvpr_2019")

                out_name = head_name[:-5]
                for identify, task_res in results.items():
                    if identify in outs['results']:
                        outs['results'][identify].extend(task_res['pred'][out_name])
                    else:
                        outs['results'][identify] = task_res['pred'][out_name]

        # 2. convert prediction result to nuscenes assessment format
        outs_nuscene_format = {'results': {}}
        for sample_token, results in outs['results'].items():
            sample = self.nusc.get('sample', sample_token)

            sensor_token = sample['data'][sensor_standard]
            sample_data = self.nusc.get('sample_data', sensor_token)
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            ego_translation = np.array(ego_pose['translation'], dtype=np.float32)
            ego_quaternion = Quaternion(np.array(ego_pose['rotation'], dtype=np.float32))

            outs_nuscene_format['results'][sample_token] = []
            for box_pred in results:
                nusc_box = Box(box_pred['translation'].tolist(),
                               box_pred['size'].tolist(),
                               Quaternion(box_pred['rotation'].tolist()),
                               velocity=box_pred['velocity'].tolist()+[0.] if 'velocity' in box_pred \
                                   else (np.nan, np.nan, np.nan))
                nusc_box.rotate(ego_quaternion)
                nusc_box.translate(ego_translation)

                translation = nusc_box.center.tolist()
                size = nusc_box.wlh.tolist()
                rotation = nusc_box.orientation.elements.tolist()
                velocity = nusc_box.velocity[:2].tolist() if 'velocity' in box_pred else [0., 0.]

                sample_result = {
                    'sample_token': sample_token,
                    'translation': translation,
                    'rotation': rotation,
                    'size': size,
                    'velocity': velocity,
                    'detection_name': box_pred['class_name'],
                    'detection_score': box_pred['conf'],
                    'attribute_name': ''
                }
                outs_nuscene_format['results'][sample_token].append(sample_result)
        outs_nuscene_format.update(self.meta)

        if 'NuScenes' in self.datasets:
            save_file = "nuscenes_box3d_results.json"
            logger.info("save results to %s" % (save_file))
            with open(os.path.join(save_dir, save_file), 'w') as file:
                json.dump(outs_nuscene_format, file)

            logger.info("%s: start evaluate in %s" % (dataset, save_file))
            eval = DetectionEval(self.nusc,
                                 self.det_cfg,
                                 os.path.join(save_dir, save_file),
                                 split,
                                 output_dir=save_dir,
                                 verbose=False)
            res = eval.evaluate()
            NDS = res[0].nd_score
            tp_log_str = '%s  Score:  ' % (head_name)
            tp_scores = {
                'NDS': res[0].nd_score,
                'mAP': res[0].mean_ap,
                'mATE': res[0].tp_errors['trans_err'],
                'mASE': res[0].tp_errors['scale_err'],
                'mAOE': res[0].tp_errors['orient_err'],
                'mAVE': res[0].tp_errors['vel_err'],
                'mAAE': res[0].tp_errors['attr_err'],
            }
            for cls_name, score in tp_scores.items():
                log_str = "%s@%.4f | " % (cls_name, score)
                tp_log_str += log_str
            logger.info(tp_log_str)

            ap_each_class = res[0].mean_dist_aps
            ap_log_str = '%s  AP:  ' % (head_name)
            for cls_name, ap in ap_each_class.items():
                log_str = "%s@%.4f | " % (cls_name, ap)
                ap_log_str += log_str
            logger.info(ap_log_str)

            metrics_summary = res[0].serialize()
            logger.info('Per-class results:')
            logger.info('Object Class\tAP\tATE\tASE\tAOE\tAVE\tAAE')
            class_aps = metrics_summary['mean_dist_aps']
            class_tps = metrics_summary['label_tp_errors']
            for class_name in class_aps.keys():
                logger.info('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
                            % (class_name, class_aps[class_name],
                               class_tps[class_name]['trans_err'],
                               class_tps[class_name]['scale_err'],
                               class_tps[class_name]['orient_err'],
                               class_tps[class_name]['vel_err'],
                               class_tps[class_name]['attr_err']))

            metrics[head_name] = NDS
        else:
            raise NotImplementedError

        return metrics
