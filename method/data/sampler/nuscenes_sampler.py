from __future__ import division
import copy

import numpy as np
import torch
from torch import distributed as dist
from torch.utils.data import Sampler
from torch.utils.data import DistributedSampler

class NuScenesSampler(Sampler):
    def __init__(self, dataset, samples_per_gpu=1):
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.flag = np.array([1] * len(dataset)) if not hasattr(dataset, 'flags') else np.array(dataset.flags)
        self.group_sizes = np.bincount(self.flag)
        self.max_group_size = self.group_sizes.max()
        self.single_task_samples_num = int(np.ceil(
            self.max_group_size / self.samples_per_gpu)) * self.samples_per_gpu
        self.num_tasks = len(np.where(self.group_sizes > 0)[0])
        self.num_samples = self.single_task_samples_num * self.num_tasks
        self.num_replicas = 1
        self.total_size = self.num_samples * self.num_replicas
        self.epoch = 0
        self.rank = 0
        self.rank_index = self._get_rank_index()

    def _get_rank_index(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.rank)

        all_indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            single_indices = []
            indice = np.where(self.flag == i)[0]
            indice = indice[list(torch.randperm(int(size),
                                                generator=g))]
            # calculate the oversample rate for small task dataset
            oversample_times = self.single_task_samples_num // size
            num_extra = self.single_task_samples_num - oversample_times * size
            for j in range(oversample_times):
                np.random.shuffle(indice)
                single_indices.append(copy.deepcopy(indice))
            np.random.shuffle(indice)
            single_indices.append(np.random.choice(indice, num_extra))
            single_indices = np.concatenate(single_indices)
            assert len(single_indices) == self.single_task_samples_num
            all_indices.append(single_indices)
        assert len(np.concatenate(all_indices)) == self.total_size

        rank_indices = []
        for i in range(self.num_replicas):
            rank_indices_per_gpu = []
            for j in range(self.num_tasks):
                num_per_slice = len(all_indices[j]) // self.num_replicas
                rank_indices_per_gpu.append(all_indices[j][i * num_per_slice:(i + 1) * num_per_slice])

            rank_indices_per_gpu = np.concatenate(rank_indices_per_gpu)
            rank_indices.append(rank_indices_per_gpu)

        rank_index = rank_indices[self.rank]

        return rank_index

    def __iter__(self):
        g = torch.Generator()
        # print('self.epoch', self.epoch)
        g.manual_seed((self.epoch + 1) * self.rank)
        rank_index = self.rank_index[torch.randperm(len(self.rank_index), generator=g)]

        return iter(rank_index)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


# class DistributedMultiTaskSampler(Sampler):
class DistributedNuscenesSampler(DistributedSampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None):
        _rank, _num_replicas = get_dist_info()
        # print("_rank, _num_replicas", _rank, _num_replicas)
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0

        self.flag = np.array([1] * len(dataset)) if not hasattr(dataset, 'flags') else np.array(dataset.flags)
        self.group_sizes = np.bincount(self.flag)
        self.max_group_size = self.group_sizes.max()

        self.single_task_samples_num = int(np.ceil(
            self.max_group_size / self.samples_per_gpu / self.num_replicas)) * self.samples_per_gpu * self.num_replicas
        self.num_tasks = len(np.where(self.group_sizes > 0)[0])

        self.num_samples = self.single_task_samples_num // self.num_replicas * self.num_tasks
        self.total_size = self.num_samples * self.num_replicas

    def _get_rank_indexes(self, generator):
        # deterministically shuffle based on epoch
        # g = torch.Generator()
        # g.manual_seed(self.rank)

        all_indices = []
        for i, size in enumerate(self.group_sizes):
            if size == 0:
                continue
            single_indices = []
            indice = np.where(self.flag == i)[0]
            indice = indice[list(torch.randperm(int(size),
                                                generator=generator))]
            # calculate the oversample rate for small task dataset
            oversample_times = self.single_task_samples_num // size
            num_extra = self.single_task_samples_num - oversample_times * size
            for j in range(oversample_times):
                np.random.shuffle(indice)
                single_indices.append(copy.deepcopy(indice))
            np.random.shuffle(indice)
            single_indices.append(np.random.choice(indice, num_extra))
            single_indices = np.concatenate(single_indices)
            assert len(single_indices) == self.single_task_samples_num
            all_indices.append(single_indices)
        assert len(np.concatenate(all_indices)) == self.total_size

        self.rank_indices = []
        for i in range(self.num_replicas):
            rank_indices_per_gpu = []
            for j in range(self.num_tasks):
                num_per_slice = len(all_indices[j]) // self.num_replicas
                rank_indices_per_gpu.append(all_indices[j][i * num_per_slice:(i + 1) * num_per_slice])

            rank_indices_per_gpu = np.concatenate(rank_indices_per_gpu)
            self.rank_indices.append(rank_indices_per_gpu)

        return self.rank_indices

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch + 1)

        self._get_rank_indexes(g)

        random_ranks = torch.randperm(self.num_replicas, generator=g)
        target_rank = random_ranks[self.rank]
        rank_index = self.rank_indices[target_rank]
        rank_index = rank_index[torch.randperm(len(rank_index), generator=g)]

        # debug
        # flags = []
        # for rank_index in self.rank_indices:
        #     rank_index = rank_index[torch.randperm(len(rank_index), generator=g)]
        #     flag = self.flag[rank_index]
        #     flags.append(flag)

        return iter(rank_index)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_dist_info():
    # if TORCH_VERSION < '1.0':
    #    initialized = dist._initialized
    # else:
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size