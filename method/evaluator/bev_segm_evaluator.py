import logging
import warnings
import json
import os
import numpy as np
from method.evaluator.utils.coco_eval import (
    coco,
    cocoeval,
    convert_utils
)

logger = logging.getLogger("ADMultiTaskPerceptionSDK")


class BEVSegmEvaluator:
    """
    适用于BEVSegmHead的评测
    """
    def __init__(self, head_kwargs, **kwargs):
        self.head_names = [head_name for head_name in list(head_kwargs.keys()) if 'head' in head_name]
        self.head_targets = [head_kwargs[head].target_names for head in self.head_names]
        pass

    def get_batch_iou(self, pred, gt):
        """Assumes preds has NOT been sigmoided yet
        """
        pred = pred > 0.5
        tgt = gt == 1
        intersect = np.sum(pred & tgt).astype(np.float32)
        union = np.sum(pred | tgt).astype(np.float32)
        return intersect, union, intersect / union if (union > 0) else 1.0

    def evaluate(self, results, save_dir, rank=-1):
        """
        Args:
            results: 多任务网络输出结果 (含有真值标签)
            save_dir:
            rank:
        Returns:
        """
        type = 'lss_based11'
        if type == 'lss_based':
            unions = {}
            inters = {}
            ious = {}
        else:
            ious = {}
        for head, cls_names in zip(self.head_names, self.head_targets):
            out_name = head[:-5]
            for identify, task_res in results.items():
                pred = task_res['pred'][out_name]
                gt = task_res['gt'][out_name]
                for i, cls_name in enumerate(cls_names):
                    inter, union, iou = self.get_batch_iou(pred[i], gt[i])
                    if type == 'lss_based':
                        if cls_name not in unions:
                            unions[cls_name] = union
                            inters[cls_name] = inter
                        else:
                            unions[cls_name] += union
                            inters[cls_name] += inter
                    else:
                        if cls_name not in ious:
                            ious[cls_name] = iou
                        else:
                            ious[cls_name] += iou
        if type == 'lss_based':
            for cls_name, union_sum in unions.items():
                inter_sum = inters[cls_name]
                ious[cls_name] = inter_sum/union_sum
                logger.info("Task: BEVSegm    IOU [%s]     [%f]" % (cls_name, ious[cls_name]))
        else:
            for cls_name, sum_score in ious.items():
                score = sum_score / len(results)
                ious[cls_name] = score
                logger.info("Task: BEVSegm    IOU [%s]     [%f]" % (cls_name, score))

        return ious
