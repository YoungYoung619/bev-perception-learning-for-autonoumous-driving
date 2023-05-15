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

import argparse
import os
import warnings
import sys

sys.path.insert(0, os.path.abspath('.'))
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

from method.data.collate import (
    collate_function
)
from method.data.dataset import build_dataset
from method.evaluator import (
    build_evaluator,
    build_evaluators
)

from method.trainer import (
    TrainingTask,
)

from method.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    load_model_weight,
    mkdir,
)

from method.data.sampler.nuscenes_sampler import (
    NuScenesSampler,
    DistributedNuscenesSampler
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        help="train config file path",
        # default='/Users/lvanyang/ADAS/ADMultiTaskPerception/config/bevdepth/3ddet_centerhead.yml'
        default='/Users/lvanyang/ADAS/bev-perception-learning-for-autonoumous-driving/config/lift_splat_shoot/lss_segm.yml'
    )
    parser.add_argument(
        "--local_rank",
        default=-1,
        type=int,
        help="node rank for distributed training"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed")
    args = parser.parse_args()
    return args


def build_dataloader(cfg, train_dataset, val_dataset, world_size, local_rank):
    if isinstance(cfg.device.gpu_ids, list):
        if cfg.data.train.name in ['NuScenesDataset']:
            train_sampler = DistributedNuscenesSampler(train_dataset,
                                                       cfg.device.batchsize_per_gpu,
                                                       world_size,
                                                       local_rank)
            c_func = collate_function
        else:
            raise NotImplementedError

        if cfg.data.train.cache_data:
            # 不再支持cache, 大网络没必要
            raise NotImplementedError("No need to cache...")
            # train_dataset.cache_data(train_sampler.rank_index)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            num_workers=cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=c_func,
            sampler=train_sampler,
            drop_last=False,
            prefetch_factor=2,
        )

    else:
        if cfg.data.train.name in ['NuScenesDataset']:
            train_sampler = NuScenesSampler(train_dataset, cfg.device.batchsize_per_gpu)
            c_func = collate_function
        else:
            raise NotImplementedError
        if cfg.data.train.cache_data:
            raise NotImplementedError("No need to cache...")
            # train_dataset.cache_data(train_sampler.rank_index)

        train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                       batch_size=cfg.device.batchsize_per_gpu,
                                                       sampler=train_sampler,
                                                       # shuffle=False,
                                                       num_workers=cfg.device.workers_per_gpu,
                                                       pin_memory=False,
                                                       collate_fn=c_func,
                                                       drop_last=False)
    val_dataloader = None
    if val_dataset is not None:
        val_dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            shuffle=False,
            num_workers=cfg.device.workers_per_gpu,
            pin_memory=False,
            collate_fn=collate_function,
            drop_last=False)

    return train_dataloader, val_dataloader


def build_task(cfg, evaluator):
    if cfg.data.train.name in ['NuScenesDataset']:
        task = TrainingTask(cfg, evaluator)
    else:
        raise NotImplementedError
    return task


def build_task_evaluator(cfg):
    if hasattr(cfg, 'evaluator'):
        evaluator = build_evaluator(cfg.evaluator)
        return evaluator
    elif hasattr(cfg, 'evaluators'):
        evaluators = build_evaluators(cfg.evaluators)
        return evaluators
    else:
        return None


def main(args):
    load_config(cfg, args.config)
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)

    logger = NanoDetLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset = build_dataset(cfg.data.train, "train", logger)
    val_dataset = build_dataset(cfg.data.val, "val", logger) if hasattr(cfg.data, "val") else None
    evaluator = build_task_evaluator(cfg)

    logger.info("Creating model...")
    task = build_task(cfg, evaluator)

    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model, map_location=lambda storage, loc: storage)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger, name="model")
        if hasattr(task, 'teacher_model'):
            load_model_weight(task.teacher_model, ckpt, logger, name="teacher_model")
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy = "cpu", None, None
    else:
        accelerator, devices, strategy = "gpu", cfg.device.gpu_ids, "ddp"

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", None),
        strategy=strategy,
        precision=16 if 'enable_fp16' in cfg.model and cfg.model.enable_fp16 else 32,
        sync_batchnorm=True if 'enable_sync_bn' in cfg.model and cfg.model.enable_sync_bn else False,
    )

    train_dataloader, val_dataloader = build_dataloader(cfg,
                                                        train_dataset,
                                                        val_dataset,
                                                        trainer.world_size,
                                                        trainer.local_rank)

    trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
