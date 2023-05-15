import logging
import warnings
import json
import os
import numpy as np
from .bev_segm_evaluator import BEVSegmEvaluator
from .box3d_evaluator import Box3dEvaluator

logger = logging.getLogger("ADMultiTaskPerceptionSDK")


class MultiTaskEvaluator:
    """
    适用于多检测头的评测
    """
    def __init__(self, evaluator_names, evaluator_kwargs):
        self.evaluator_names = evaluator_names
        self.evaluator_kwargs = evaluator_kwargs
        for evaluator_name, evaluator_kwarg in zip(self.evaluator_names, self.evaluator_kwargs):
            name = evaluator_kwarg.name
            setattr(self, evaluator_name, eval(name)(head_kwargs=evaluator_kwarg))
        pass

    def evaluate(self, results, save_dir, rank=-1):
        """
        Args:
            results: 多任务网络输出结果 (含有真值标签)
            save_dir:
            rank:
        Returns:
        """
        metrics = {}
        for evaluator_name in self.evaluator_names:
            evaluator = getattr(self, evaluator_name)
            metric = evaluator.evaluate(results, save_dir, rank)
            metrics.update(metric)

        return metrics
