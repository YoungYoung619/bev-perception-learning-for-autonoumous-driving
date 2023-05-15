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

import collections
import re

import torch
from torch._six import string_classes

np_str_obj_array_pattern = re.compile(r"[SaUO]")

default_collate_err_msg_format = (
    "default_collate: batch must contain tensors, numpy arrays, numbers, "
    "dicts or lists; found {}"
)

KEY_LIST = list()
COUNT = 10


def collate_function(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""
    if len(batch) == 0:
        return batch

    elem = batch[0]
    elem_type = type(elem)

    global COUNT
    if COUNT > 0:
        if isinstance(elem, dict) and 'ann_info' in elem.keys():
            COUNT -= 1
            for instance in batch:
                for key in instance.keys():
                    if key not in KEY_LIST:
                        KEY_LIST.append(key)

    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return batch
        elif elem.shape == ():  # scalars
            return batch
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float32)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        if 'ann_info' in elem.keys():
            key_list = KEY_LIST
        else:
            key_list = elem.keys()
        return {key: collate_function([d[key] for d in batch if key in d.keys()]) for key in key_list}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(collate_function(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        transposed = zip(*batch)
        # for a in transposed:
        #     print(len(a))
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def collate_function_for_ateacher(batch):
    src_batch = [b for b in batch if 'source' in b]
    dst_batch = [b for b in batch if 'target' in b]
    src_data = collate_function(src_batch)
    dst_batch = collate_function(dst_batch)
    src_data.update(dst_batch)
    return src_data

def naive_collate(batch):
    """Only collate dict value in to a list. E.g. meta data dict and img_info
    dict will be collated."""

    elem = batch[0]
    if isinstance(elem, dict):
        return {key: naive_collate([d[key] for d in batch]) for key in elem}
    else:
        return batch
