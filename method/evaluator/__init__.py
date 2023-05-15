# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from .bev_segm_evaluator import BEVSegmEvaluator
from .multitask_evaluator import MultiTaskEvaluator
from .box3d_evaluator import Box3dEvaluator

def build_evaluator(cfg):
    evaluator_cfg = copy.deepcopy(cfg)
    name = evaluator_cfg.pop("name")
    if name == "BEVSegmEvaluator":
        return BEVSegmEvaluator(**cfg)
    elif name == "Box3dEvaluator":
        return Box3dEvaluator(**cfg)
    else:
        raise NotImplementedError

def build_evaluators(cfg):
    evaluator_cfg = copy.deepcopy(cfg)
    evaluator_names = list(evaluator_cfg.keys())
    evaluator_kwargs = [evaluator_cfg[evaluator_name] for evaluator_name in evaluator_names]
    evaluator = MultiTaskEvaluator(evaluator_names, evaluator_kwargs)
    return evaluator

