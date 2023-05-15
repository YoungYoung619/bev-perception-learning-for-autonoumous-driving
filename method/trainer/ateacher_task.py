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
import json
import os
import warnings
import re
import collections
from typing import Any, Dict, List
from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np
import torch
import torch.distributed as dist
from torch._six import string_classes
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
import cv2

from method.data.batch_process import stack_batch_img
from method.util import convert_avg_params, gather_results, mkdir

from ..model.arch import build_model
from ..model.weight_averager import build_weight_averager
from method.data.transform.warp import warp_boxes

np_str_obj_array_pattern = re.compile(r'[SaUO]')
default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}")


def concat_all(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    elem = batch[0]
    elem_type = type(elem)

    if isinstance(elem, torch.Tensor):
        out = None
        # TODO: support pytorch < 1.3
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            # return collate_function([torch.as_tensor(b) for b in batch])
            return batch
        elif elem.shape == ():  # scalars
            # return torch.as_tensor(batch)
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        key_list = elem.keys()
        return {key: concat_all([d[key] for d in batch if key in d.keys()]) for key in key_list}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(concat_all(samples) for samples in zip(*batch)))
    elif isinstance(elem, list) or isinstance(elem, tuple):
        elem = list(elem) if isinstance(elem, tuple) else elem
        if len(elem) == 0:
            return elem
        if type(elem[0]) in [str, int, float, np.ndarray]:
            elems = []
            for b in batch:
                elems.extend(b)
            return elems
        else:
            transposed = zip(*batch)
            return [concat_all(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def get_pos_mask(batch, mask):
    if isinstance(batch, torch.Tensor):
        return batch[mask]
    elif isinstance(batch, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(batch, int):
        return torch.tensor(batch)[mask]
    elif isinstance(batch, string_classes):
        return batch
    elif isinstance(batch, collections.abc.Mapping):
        key_list = batch.keys()
        return {key: get_pos_mask(batch[key], mask) for key in key_list}
    elif isinstance(batch, list):
        if len(batch) == len(mask):
            new = []
            for i, val in enumerate(batch):
                if mask[i]:
                    new.append(val)
        else:
            return [get_pos_mask(val, mask) for val in batch]
        return new


class ATeacherTrainingTask(LightningModule):
    """
    Pytorch Lightning module of a general training task.
    Including training, evaluating and testing.
    Args:
        cfg: Training configurations
        evaluator: Evaluator for evaluating the model performance.
    """

    def __init__(self, cfg, evaluator=None):
        super(ATeacherTrainingTask, self).__init__()
        self.cfg = cfg
        self.model = build_model(cfg.model)
        self.teacher_model = build_model(cfg.model)
        self.teacher_model.eval()
        self.evaluator = evaluator
        self.save_flag = -10
        self.log_style = "ADMultiTaskPerceptionSDK"
        self.weight_averager = None
        if "weight_averager" in cfg.model:
            self.weight_averager = build_weight_averager(
                cfg.model.weight_averager, device=self.device
            )
            self.avg_model = copy.deepcopy(self.model)
        self.save_interval = cfg.save.interval
        self.burn_up_epoch = self.cfg.schedule.ateacher.burn_up_epoch
        self.burn_up = True

    def _preprocess_batch_input(self, batch):
        batch_imgs = batch["img"]
        if isinstance(batch_imgs, list):
            batch_imgs = [img.to(self.device) for img in batch_imgs]
            batch_img_tensor = stack_batch_img(batch_imgs, divisible=32)
            batch["img"] = batch_img_tensor
        return batch

    def forward(self, x):
        x = self.model(x)
        return x

    @torch.no_grad()
    def predict(self, batch, batch_idx=None, dataloader_idx=None):
        # batch = self._preprocess_batch_input(batch)
        preds = self.forward(batch["img"])
        results = self.model.head.post_process(preds, batch)
        return results

    def attach_label(self, meta, prediction):
        """只有高质量的标签才会attach"""
        conf_threshes = self.cfg.schedule.domain_cross.confidence_thresholds
        for task_name, contents in prediction.items():
            if task_name in ['face', 'moving_obstacle']:
                conf_thresh = conf_threshes[task_name]
                for batch_idx, batch_contents in contents.items():
                    if task_name == 'face':
                        meta['gt_face_bboxes'].append([])
                        meta['gt_face_labels'].append([])
                    elif task_name == 'moving_obstacle':
                        meta['gt_bboxes'].append([])
                        meta['gt_labels'].append([])
                    for cls_idx, cls_contents in batch_contents.items():
                        bboxes = np.array([box[:4] for box in cls_contents if box[-1] > conf_thresh[cls_idx]],
                                          dtype=np.float32)

                        if task_name == 'face':
                            meta['gt_face_bboxes'][batch_idx].extend(bboxes)
                            meta['gt_face_labels'][batch_idx].extend(np.array([cls_idx] * len(bboxes), dtype=np.int64))
                        elif task_name == 'moving_obstacle':
                            meta['gt_bboxes'][batch_idx].extend(bboxes)
                            meta['gt_labels'][batch_idx].extend(np.array([cls_idx] * len(bboxes), dtype=np.int64))

                    if task_name == 'face':
                        meta['gt_face_bboxes'][batch_idx] = np.array(meta['gt_face_bboxes'][batch_idx],
                                                                     dtype=np.float32)
                        meta['gt_face_labels'][batch_idx] = np.array(meta['gt_face_labels'][batch_idx], dtype=np.int64)
                    elif task_name == 'moving_obstacle':
                        meta['gt_bboxes'][batch_idx] = np.array(meta['gt_bboxes'][batch_idx], dtype=np.float32)
                        meta['gt_labels'][batch_idx] = np.array(meta['gt_labels'][batch_idx], dtype=np.int64)

    def change_post_process_confidence(self, task_name, high_confidence_cfg):
        # if task_name == '3d_kps':
        #     task_name = 'kps3d'
        #
        # if not hasattr(self, 'confidence_storage'):
        #     setattr(self, 'confidence_storage', {})
        # self.confidence_storage[task_name] = self.cfg['%s_cfg' % (task_name)].conf_thresh
        #
        # self.cfg.defrost()
        # self.cfg['%s_cfg' % (task_name)].conf_thresh = high_confidence_cfg[task_name]
        # self.cfg.freeze()
        pass

    def reset_post_process_confidence(self, task_name):
        # if task_name == '3d_kps':
        #     task_name = 'kps3d'
        #
        # self.cfg.defrost()
        # self.cfg['%s_cfg' % (task_name)].conf_thresh = self.confidence_storage[task_name]
        # self.cfg.freeze()
        pass

    def record_loss(self, save, loss_states, branch):
        assert branch in ['sup', 'unsup', 'domain']
        for name, val in loss_states.items():
            save['%s_%s' % (branch, name)] = val

    def _de_normalize(self, img, mean=[103.53, 116.28, 123.675], std=[57.375, 57.12, 58.395]):
        mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3) / 255
        std = np.array(std, dtype=np.float32).reshape(1, 1, 3) / 255
        img = img * std + mean
        img = (img * 255).astype(np.uint8)
        return img

    def training_step(self, batch, batch_idx):
        loss_states = {}
        src_weakly, src_strongly = batch['source']['weakly'], batch['source']['strongly']
        dst_weakly, dst_strongly = batch['target']['weakly'], batch['target']['strongly']

        # combine the source labels from strongly and weakly augmentation
        src_total = concat_all([src_strongly, src_weakly])

        if self.burn_up:
            preds, loss_sup, loss_states_sup = self.model.forward_train(src_total)
            loss_domain_, loss_states_domain_ = self.model.forward_domain_cross(
                imgs_src=src_total['img'])  # 这个只有burn up阶段才需要
            self.record_loss(loss_states, loss_states_sup, branch='sup')
            self.record_loss(loss_states, loss_states_domain_, branch='sup')

            loss = self.cfg.schedule.loss_weights.sup * loss_sup + 0.001 * loss_domain_
        else:
            # 1.
            # supervised loss from ground truth for student model
            preds, loss_sup, loss_states_sup = self.model.forward_train(src_total, branch='sup')
            # loss_domain_, loss_states_domain_ = self.model.forward_domain_cross(
            #     imgs_src=src_total['img'], imgs_dst=dst_total['img'])  # 这个只有burn up阶段才需要
            self.record_loss(loss_states, loss_states_sup, branch='sup')
            # self.record_loss(loss_states, loss_states_domain_, branch='sup')

            # 2.
            # generate the Pseudo label from target domain of weakly augmentation data
            self.change_post_process_confidence(src_total['img_info']['task_id'][0],
                                                self.cfg.schedule.domain_cross.confidence_thresholds)
            # pl seems to make all the torch modules to be trainable when in training step
            self.teacher_model.eval()
            dst_weakly_predictions = self.teacher_model.inference(dst_weakly, warp=False)
            self.reset_post_process_confidence(src_total['img_info']['task_id'][0])

            # debug
            # meta_test = dst_weakly
            # for batch_idx in range(len(meta_test['img'])):
            #     img = meta_test['img'][batch_idx].permute(1, 2, 0).cpu().numpy()
            #     img = self._de_normalize(img.copy())
            #     res = dst_weakly_predictions['moving_obstacle'][batch_idx]
            #     for cls_idx, boxes in res.items():
            #         for box in boxes:
            #             if box[-1] < 0.2:
            #                 continue
            #             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            #             cv2.rectangle(img, (x1, y1), (x2, y2), color=(100, 200, 100), thickness=1)
            #     cv2.imshow('dddd', img)

            # attach the high quality prediction to meta
            # self.attach_label(dst_weakly, dst_weakly_predictions)
            self.attach_label(dst_strongly, dst_weakly_predictions)

            # 3.
            # use pseudo label (high quality) to train student model in target domain (use strongly augmentation data)
            _, loss_unsup, loss_states_unsup = self.model.forward_train(dst_strongly, branch='unsup')
            self.record_loss(loss_states, loss_states_unsup, branch='unsup')

            # 4.
            # domain cross loss (use the weakly augmentation data)
            loss_domain, loss_states_domain = self.model.forward_domain_cross(imgs_src=src_weakly['img'],
                                                                              imgs_dst=dst_weakly['img'])
            self.record_loss(loss_states, loss_states_domain, branch='domain')

            loss = self.cfg.schedule.loss_weights.sup * loss_sup \
                   + self.cfg.schedule.loss_weights.unsup * loss_unsup \
                   + self.cfg.schedule.loss_weights.domain_cross * loss_domain

            # visualized pseudo labels
            # meta_test = dst_strongly
            # for batch_idx in range(len(meta_test['img'])):
            #     img = meta_test['img'][batch_idx].permute(1, 2, 0).cpu().numpy()
            #     img = self._de_normalize(img.copy())
            #     save_img = img.copy()
            #     cv2.imwrite('/Volumes/KINGSTON0/visualization/test/%d.jpg'%(batch_idx), save_img)
            #     show = img
            #     if 'gt_bboxes' in meta_test:
            #         bboxes = meta_test['gt_bboxes'][batch_idx]
            #         for box in bboxes:
            #             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            #             cv2.rectangle(show, (x1, y1), (x2, y2), thickness=2, color=(200, 100, 50))
            #
            #     if 'gt_face_bboxes' in meta_test:
            #         bboxes = meta_test['gt_face_bboxes'][batch_idx]
            #         for box in bboxes:
            #             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            #             cv2.rectangle(show, (x1, y1), (x2, y2), thickness=2, color=(50, 200, 100))
            #
            #     cv2.imshow('test', show)
            #     cv2.waitKey(0)

        # log train losses
        if self.global_step % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            log_msg = "Train|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                self.trainer.num_training_batches,
                memory,
                lr,
            )
            self.scalar_summary("Train_loss/lr", "Train", lr, self.global_step)
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
                self.scalar_summary(
                    "Train_loss/" + loss_name,
                    "Train",
                    loss_states[loss_name].mean().item(),
                    self.global_step,
                )
            self.logger.info(log_msg)

        return loss

    def training_step_end(self, step_output):
        if self.global_step % self.cfg.schedule.ateacher.teacher_update_step == 0:
            self._update_teacher_model(
                keep_rate=self.cfg.schedule.ateacher.ema_ratio)

    def training_epoch_end(self, outputs: List[Any]) -> None:
        self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_last.ckpt"))

        if self.current_epoch % self.save_interval == 0:
            self.trainer.save_checkpoint(os.path.join(self.cfg.save_dir, "model_epoch_%d.ckpt" % (self.current_epoch)))

    def validation_step(self, batch, batch_idx):
        # batch = self._preprocess_batch_input(batch)
        model = self.teacher_model if hasattr(self, 'teacher_model') else self.model
        model.eval()
        with torch.no_grad():
            preds, loss, loss_states = model.forward_train(batch)
        model.train()

        if batch_idx % self.cfg.log.interval == 0:
            memory = (
                torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else 0
            )
            lr = self.optimizers().param_groups[0]["lr"]
            log_msg = "Val|Epoch{}/{}|Iter{}({}/{})| mem:{:.3g}G| lr:{:.2e}| ".format(
                self.current_epoch + 1,
                self.cfg.schedule.total_epochs,
                self.global_step,
                batch_idx + 1,
                sum(self.trainer.num_val_batches),
                memory,
                lr,
            )
            for loss_name in loss_states:
                log_msg += "{}:{:.4f}| ".format(
                    loss_name, loss_states[loss_name].mean().item()
                )
            self.logger.info(log_msg)

        """
        generate a dict to store the gt and pred for each head
        {
            "file_name": {
                "pred": {
                    user-defined ...
                }
                "gt": {
                    user-defined ...
                }
            }
        }
        """
        dets = {}
        if 'face' in preds:
            valid_idx = torch.where(batch['img_info']['task_id'] == self.model.head_cfg['face_head'].task_tag)[0]
            if valid_idx.shape[0]:
                res = self.model.face_head.post_process(preds['face'][valid_idx], batch, valid_idx)
                dets['face'] = {}
                det_label = self.model.face_head.det_target
                for i, idx in enumerate(valid_idx):
                    dets['face'][batch['img_info']['file_name'][idx]] = {}
                    dets['face'][batch['img_info']['file_name'][idx]]['pred'] = res[i]
                    dets['face'][batch['img_info']['file_name'][idx]]['gt'] = {}
                    gt_bboxes = batch['gt_face_bboxes'][i]
                    width = batch['img_info']['width'][idx].cpu().numpy()
                    height = batch['img_info']['height'][idx].cpu().numpy()
                    warp_matrix = batch['warp_matrix'][idx]
                    gt_bboxes = warp_boxes(
                        gt_bboxes, np.linalg.inv(warp_matrix), width, height
                    )
                    for j, raw_label in enumerate(batch['gt_face_labels'][i]):
                        if raw_label not in det_label:
                            continue
                        map_label = self.model.face_head.map_label[det_label.index(raw_label)]
                        if map_label not in dets['face'][batch['img_info']['file_name'][idx]]['gt']:
                            dets['face'][batch['img_info']['file_name'][idx]]['gt'][map_label] = []
                        dets['face'][batch['img_info']['file_name'][idx]]['gt'][map_label].append(gt_bboxes[j].tolist())

        if 'moving_obstacle' in preds:
            valid_idx = \
                torch.where(batch['img_info']['task_id'] == self.model.head_cfg['moving_obstacle_head'].task_tag)[0]
            if valid_idx.shape[0]:
                res = self.model.moving_obstacle_head.post_process(preds['moving_obstacle'][valid_idx], batch,
                                                                   valid_idx)
                dets['moving_obstacle'] = {}
                det_label = self.model.moving_obstacle_head.det_target
                for i, idx in enumerate(valid_idx):
                    #
                    dets['moving_obstacle'][batch['img_info']['file_name'][idx]] = {}
                    dets['moving_obstacle'][batch['img_info']['file_name'][idx]]['pred'] = res[i]
                    dets['moving_obstacle'][batch['img_info']['file_name'][idx]]['gt'] = {}
                    gt_bboxes = batch['gt_bboxes'][i]
                    width = batch['img_info']['width'][idx].cpu().numpy()
                    height = batch['img_info']['height'][idx].cpu().numpy()
                    warp_matrix = batch['warp_matrix'][idx]
                    gt_bboxes = warp_boxes(
                        gt_bboxes, np.linalg.inv(warp_matrix), width, height
                    )
                    for j, raw_label in enumerate(batch['gt_labels'][i]):
                        if raw_label not in det_label:
                            continue
                        map_label = self.model.moving_obstacle_head.map_label[det_label.index(raw_label)]
                        if map_label not in dets['moving_obstacle'][batch['img_info']['file_name'][idx]]['gt']:
                            dets['moving_obstacle'][batch['img_info']['file_name'][idx]]['gt'][map_label] = []
                        dets['moving_obstacle'][batch['img_info']['file_name'][idx]]['gt'][map_label].append(
                            gt_bboxes[j].tolist())

        # 可视化调试
        # import cv2
        # for file_name, content in dets['face'].items():
        #     img = cv2.imread(file_name)
        #     img_p = img.copy()
        #     gt_bboxes = content['gt']
        #     for label, boxes in gt_bboxes.items():
        #         for box in boxes:
        #             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)
        #
        #     preds_bboxes = content['pred']
        #     for label, boxes in preds_bboxes.items():
        #         for box in boxes:
        #             if box[-1] < 0.2:
        #                 continue
        #             x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        #             cv2.rectangle(img_p, (x1, y1), (x2, y2), (0, 255, 0), thickness=2)
        #     cv2.imshow('gt', img)
        #     cv2.imshow('pred', img_p)
        #     cv2.waitKey(0)
        return dets

    def validation_epoch_end(self, validation_step_outputs):
        """
        Called at the end of the validation epoch with the
        outputs of all validation steps.Evaluating results
        and save best model.
        Args:
            validation_step_outputs: A list of val outputs
        """
        import json
        results = []

        for res in validation_step_outputs:
            results.append(res)
        all_results = (
            gather_results(results)
            if dist.is_available() and dist.is_initialized()
            else results
        )

        # print(self.local_rank, "all_results type: ", type(all_results))
        if all_results:
            flatten_results = {}
            for res in all_results:
                for key, val in res.items():
                    if key not in flatten_results:
                        flatten_results[key] = {}
                    flatten_results[key].update(val)

            eval_results = self.evaluator.evaluate(
                flatten_results, self.cfg.save_dir, rank=self.local_rank
            )
            sum_val = 0.
            print(eval_results)
            for task, metric in eval_results.items():
                sum_val += metric
            metric = sum_val / len(eval_results)
            # save best model
            if metric > self.save_flag:
                self.save_flag = metric
                best_save_path = os.path.join(self.cfg.save_dir, "model_best")
                mkdir(self.local_rank, best_save_path)
                self.trainer.save_checkpoint(
                    os.path.join(best_save_path, "model_best.ckpt")
                )
                self.save_model_state(
                    os.path.join(best_save_path, "nanodet_model_best.pth")
                )
                txt_path = os.path.join(best_save_path, "eval_results.txt")
                if self.local_rank < 1:
                    with open(txt_path, "a") as f:
                        f.write("Epoch:{}\n".format(self.current_epoch + 1))
                        for k, v in eval_results.items():
                            f.write("{}: {}\n".format(k, v))
            else:
                warnings.warn(
                    "Warning! Save_key is not in eval results! Only save model last!"
                )
            self.logger.log_metrics(eval_results, self.current_epoch + 1)
        else:
            self.logger.info("Skip val on rank {}".format(self.local_rank))

    def test_step(self, batch, batch_idx):
        # dets = self.predict(batch, batch_idx)
        # return dets
        pass

    def test_epoch_end(self, test_step_outputs):
        pass
        # results = {}
        # for res in test_step_outputs:
        #     results.update(res)
        # all_results = (
        #     gather_results(results)
        #     if dist.is_available() and dist.is_initialized()
        #     else results
        # )
        # if all_results:
        #     res_json = self.evaluator.results2json(all_results)
        #     json_path = os.path.join(self.cfg.save_dir, "results.json")
        #     json.dump(res_json, open(json_path, "w"))
        #
        #     if self.cfg.test_mode == "val":
        #         eval_results = self.evaluator.evaluate(
        #             all_results, self.cfg.save_dir, rank=self.local_rank
        #         )
        #         txt_path = os.path.join(self.cfg.save_dir, "eval_results.txt")
        #         with open(txt_path, "a") as f:
        #             for k, v in eval_results.items():
        #                 f.write("{}: {}\n".format(k, v))
        # else:
        #     self.logger.info("Skip test on rank {}".format(self.local_rank))

    def configure_optimizers(self):
        """
        Prepare optimizer and learning-rate scheduler
        to use in optimization.

        Returns:
            optimizer
        """
        optimizer_cfg = copy.deepcopy(self.cfg.schedule.optimizer)
        name = optimizer_cfg.pop("name")
        build_optimizer = getattr(torch.optim, name)
        optimizer = build_optimizer(params=self.parameters(), **optimizer_cfg)

        schedule_cfg = copy.deepcopy(self.cfg.schedule.lr_schedule)
        name = schedule_cfg.pop("name")
        build_scheduler = getattr(torch.optim.lr_scheduler, name)
        lr_scheduler = build_scheduler(optimizer=optimizer, **schedule_cfg)

        return dict(optimizer=optimizer, lr_scheduler=lr_scheduler)

    def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_idx=None,
            optimizer_closure=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
    ):
        """
        Performs a single optimization step (parameter update).
        Args:
            epoch: Current epoch
            batch_idx: Index of current batch
            optimizer: A PyTorch optimizer
            optimizer_idx: If you used multiple optimizers this indexes into that list.
            optimizer_closure: closure for all optimizers
            on_tpu: true if TPU backward is required
            using_native_amp: True if using native amp
            using_lbfgs: True if the matching optimizer is lbfgs
        """
        # warm up lr
        if self.trainer.global_step <= self.cfg.schedule.warmup.steps:
            if self.cfg.schedule.warmup.name == "constant":
                warmup_lr = (
                        self.cfg.schedule.optimizer.lr * self.cfg.schedule.warmup.ratio
                )
            elif self.cfg.schedule.warmup.name == "linear":
                k = (1 - self.trainer.global_step / self.cfg.schedule.warmup.steps) * (
                        1 - self.cfg.schedule.warmup.ratio
                )
                warmup_lr = self.cfg.schedule.optimizer.lr * (1 - k)
            elif self.cfg.schedule.warmup.name == "exp":
                k = self.cfg.schedule.warmup.ratio ** (
                        1 - self.trainer.global_step / self.cfg.schedule.warmup.steps
                )
                warmup_lr = self.cfg.schedule.optimizer.lr * k
            else:
                raise Exception("Unsupported warm up type!")
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def scalar_summary(self, tag, phase, value, step):
        """
        Write Tensorboard scalar summary log.
        Args:
            tag: Name for the tag
            phase: 'Train' or 'Val'
            value: Value to record
            step: Step value to record

        """
        if self.local_rank < 1:
            self.logger.experiment.add_scalars(tag, {phase: value}, step)

    def _update_teacher_model(self, keep_rate=0.9996):
        student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.teacher_model.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                        student_model_dict[key] *
                        (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.teacher_model.load_state_dict(new_teacher_dict)

    def info(self, string):
        self.logger.info(string)

    @rank_zero_only
    def save_model_state(self, path):
        self.logger.info("Saving model to {}".format(path))
        torch.save(self.model.state_dict(), path)

    # ------------Hooks-----------------
    def on_train_start(self) -> None:
        self.model.set_epoch(self.current_epoch)
        if self.current_epoch >= self.burn_up_epoch:
            self.burn_up = False
            self.logger.info("Start ateacher training")

    def on_fit_start(self) -> None:
        if "weight_averager" in self.cfg.model:
            self.logger.info("Weight Averaging is enabled")
            if self.weight_averager and self.weight_averager.has_inited():
                self.weight_averager.to(self.weight_averager.device)
                return
            self.weight_averager = build_weight_averager(
                self.cfg.model.weight_averager, device=self.device
            )
            self.weight_averager.load_from(self.model)

    def on_train_epoch_start(self):
        self.model.set_epoch(self.current_epoch)
        if self.current_epoch == self.burn_up_epoch:
            self.burn_up = False
            self._update_teacher_model(keep_rate=0.)
            self.logger.info("Start ateacher training")

    def on_train_epoch_end(self):
        # 本来应该在on_validation_epoch_start里的, 但如果没有 validation dataset的话，就需要在这里
        if self.weight_averager and self.global_step > 1:
            self.weight_averager.apply_to(self.avg_model)

    def on_train_batch_end(self, outputs, batch, batch_idx) -> None:
        if self.weight_averager:
            self.weight_averager.update(self.model, self.global_step)

    def on_validation_epoch_start(self):
        if self.weight_averager:
            self.weight_averager.apply_to(self.avg_model)
        pass

    def on_test_epoch_start(self) -> None:
        if self.weight_averager:
            self.on_load_checkpoint({"state_dict": self.state_dict()})
            self.weight_averager.apply_to(self.model)

    def on_load_checkpoint(self, checkpointed_state: Dict[str, Any]) -> None:
        if self.weight_averager:
            avg_params = convert_avg_params(checkpointed_state)
            if len(avg_params) != len(self.model.state_dict()):
                self.logger.info(
                    "Weight averaging is enabled but average state does not"
                    "match the model"
                )
            else:
                self.weight_averager = build_weight_averager(
                    self.cfg.model.weight_averager, device=self.device
                )
                self.weight_averager.load_state_dict(avg_params)
                self.logger.info("Loaded average state from checkpoint.")
