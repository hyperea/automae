# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
import io
import numpy as np
from PIL import Image

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import torch
import h5py
import pickle
from typing import Dict, Optional, Callable, Tuple, Any


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class HDF5ImageNetDataset(datasets.VisionDataset):
    def __init__(self, 
                 path: str,
                 subset: str, 
                 transform: "Optional[Callable]" = None, 
                 target_transform: "Optional[Callable]" = None):
        super().__init__(root=path, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self._find_classes(path, subset)
        samples = self.make_dataset(path, subset, class_to_idx)
        
        self.classes = classes
        self.loader = pil_loader
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.imgs = self.samples

    def _find_classes(self, h5file_path: str, subset: str):
        cached_file = h5file_path + ".classes_cache.pkl"
        if os.path.exists(cached_file):
            with open(cached_file, "rb") as f:
                classes, class_to_idx = pickle.load(f)
                return classes, class_to_idx
                
        with h5py.File(h5file_path, "r") as f:
            classes = list(f[subset].keys())
            classes.sort()
            class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        
        with open(cached_file, "wb") as f:
            pickle.dump((classes, class_to_idx), f)

        return classes, class_to_idx

    @staticmethod
    def make_dataset(
        h5file_path: str, 
        subset: str, 
        class_to_idx: Dict[str, int]
    ):
        cached_file = h5file_path + f".instances_cache_{subset}.pkl"
        if os.path.exists(cached_file):
            with open(cached_file, "rb") as f:
                instances = pickle.load(f)
                return instances

        with h5py.File(h5file_path, "r") as f:
            instances = []
            for target_class in sorted(class_to_idx.keys()):
                class_index = class_to_idx[target_class]
                target_dir = os.path.join(subset, target_class)
                for fname in sorted(f[target_dir].keys()):
                    item = os.path.join(subset, target_class, fname), class_index
                    instances.append(item)
        
        with open(cached_file, "wb") as f:
            pickle.dump(instances, f)

        return instances

    def __getitem__(self, index: int) -> "Tuple[Any, Any]":
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(self.root, path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self) -> int:
        return len(self.samples)


def pil_loader(h5file_path: str, path: str) -> "Image.Image":
    with h5py.File(h5file_path, "r") as f:
        buf = f[path][()].tobytes()
        iof = io.BytesIO(buf)
        img = Image.open(iof)
        return img.convert('RGB')
    

class DataPrefetcher:
    def __init__(self, loader):
        self._len = len(loader)
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def __len__(self):
        return self._len

    def preload(self):
        self.next_input, self.next_target = next(self.loader)
        
        # if record_stream() doesn't work, another option is to make sure device inputs are created
        # on the main stream.
        # self.next_input_gpu = torch.empty_like(self.next_input, device='cuda')
        # self.next_target_gpu = torch.empty_like(self.next_target, device='cuda')
        # Need to make sure the memory allocated for next_* is not still in use by the main stream
        # at the time we start copying to next_*:
        # self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_target = self.next_target.cuda(non_blocking=True)
            # more code for the alternative if record_stream() doesn't work:
            # copy_ will record the use of the pinned source tensor in this side stream.
            # self.next_input_gpu.copy_(self.next_input, non_blocking=True)
            # self.next_target_gpu.copy_(self.next_target, non_blocking=True)
            # self.next_input = self.next_input_gpu
            # self.next_target = self.next_target_gpu

            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            self.next_input = self.next_input.float()
            self.next_input = self.next_input.sub_(self.mean).div_(self.std)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

def fast_collate(batch, memory_format):

    imgs = [img[0] for img in batch]
    targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)
    w = imgs[0].size[0]
    h = imgs[0].size[1]
    tensor = torch.zeros( (len(imgs), 3, h, w), dtype=torch.uint8).contiguous(memory_format=memory_format)
    for i, img in enumerate(imgs):
        nump_array = np.asarray(img, dtype=np.uint8)
        if(nump_array.ndim < 3):
            nump_array = np.expand_dims(nump_array, axis=-1)
        nump_array = np.rollaxis(nump_array, 2)
        tensor[i] += torch.from_numpy(nump_array)
    return tensor, targets
