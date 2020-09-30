# -*- coding: utf-8 -*-
"""lightning_mnist_benchmark.ipynb

Automatically generated by Colaboratory.

# Script to benchmark DP training using FakeData

The data augmentation applied with `torchvision` and `kornia`.

"""## Import needed libraries"""

#! Needed for Pytorch-Lightning profiling
import logging
logging.basicConfig(level=logging.INFO)

import os
from functools import partial
import numpy as np
import PIL.Image as Image

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import FakeData
from torchvision.models import resnet18
import torchvision as T


import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import kornia as K


"""## Define lightning model"""
class CoolSystem(pl.LightningModule):

    def __init__(self, batch_size: int = 32, augmentation_backend: str = 'kornia', image_size=(3, 224, 224), mode='default'):
        super(CoolSystem, self).__init__()
        self._batch_size: int = batch_size
        self.image_size = image_size
        self.mode = mode

        self.model = resnet18(False)

        if augmentation_backend == 'kornia':
            self.augmentation = torch.nn.Sequential(
                K.augmentation.RandomAffine([-45., 45.], [0., 0.5], [0.5, 1.5], [0., 0.5], p=1.),
                K.augmentation.ColorJitter(0.4, 0.4, 0.4, 0.1, p=1.),
                K.augmentation.Normalize(0.1307, 0.3081),
            )
            self.transform = lambda x: K.image_to_tensor(np.array(x)).float() / 255.

        elif augmentation_backend == 'torchvision':
            self.augmentation = None
            self.transform = T.transforms.Compose([
                T.transforms.RandomAffine([-45., 45.], [0., 0.5], [0.5, 1.5], [0., 0.5], resample=2),
                T.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
                T.transforms.ToTensor(),
                T.transforms.Normalize((0.1307,), (0.3081,)),
            ])

        elif augmentation_backend == 'albumentations':
            self.augmentation = None
            transform = A.Compose([
                A.ShiftScaleRotate(shift_limit=0.5, scale_limit=0.1, rotate_limit=45, always_apply=True, p=1.),
                A.HueSaturationValue(0.1, 0.4, 0.4, always_apply=True, p=1.),
                A.Normalize([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081], always_apply=True, p=1.),
                ToTensorV2(always_apply=True, p=1.)
            ])
            self.transform = partial(albu_transform, transform=transform)

        else:
            raise ValueError(f"Unsupported backend: {augmentation_backend}")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        if self.mode == 'default':
            if self.augmentation is not None:
                with torch.no_grad():
                    x = self.augmentation(x)  # => we perform GPU/Batched data augmentation
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        # (LBFGS it is automatically supported, no need for closure function)
        return torch.optim.Adam(self.parameters(), lr=0.0004)

    def train_dataloader(self):
        # REQUIRED
        # dataset = CIFAR10(os.getcwd(), train=True, download=False, transform=self.transform)
        dataset = FakeData(size=2560, transform=self.transform, image_size=self.image_size)
        loader = DataLoader(dataset, batch_size=self._batch_size, num_workers=os.cpu_count(),
                            collate_fn=partial(collate_fn_prefetch, aug=self.augmentation) if (self.mode == 'prefetch' and self.augmentation is not None) else None)
        return loader


def collate_fn_prefetch(batch, aug):
    imgs, labels = zip(*batch)
    # Using CUDA REQUIRE: torch.multiprocessing.set_start_method("spawn")
    imgs = torch.stack(imgs).to('cuda')
    return aug(imgs), torch.stack(labels)


def albu_transform(image, transform):
    return transform(image=np.array(image))['image']


if __name__ == '__main__':

    torch.multiprocessing.set_start_method("spawn")

    """## Run training"""

    from pytorch_lightning import Trainer

    mode = 'default'
    # Prefetch works bad with high GPU taken
    # mode = 'prefetch'

    dp_gpus = [1, 2, 3, 4]

    backends = [
        'albumentations',
        # 'kornia',
        # 'torchvision'
    ]

    image_sizes = [(3, 32, 32), (3, 224, 224), (3, 512, 512)]

    batch_sizes = [512]

    from collections import defaultdict
    results_dict = defaultdict(dict)

    for num_gpus in dp_gpus:
        results_dict[num_gpus] = {}
        for image_size in image_sizes:
            results_dict[num_gpus][image_size] = {}
            for backend in backends:
                results_dict[num_gpus][image_size][backend] = {}
                for batch_size in batch_sizes:

                    model = CoolSystem(batch_size=batch_size, augmentation_backend=backend, mode=mode)

                    # most basic trainer, uses good defaults
                    prof = pl.profiler.SimpleProfiler()
                    trainer = Trainer(profiler=prof, max_epochs=1, gpus=num_gpus, distributed_backend="dp")
                    trainer.fit(model)

                    # sum results
                    elapsed_time: float = 0.
                    for (key, val) in prof.recorded_durations.items():
                        elapsed_time += sum(val)

                    print(f"## Training gpus: {num_gpus} / Image Size: {image_size} / backend: {backend} / batch_size: {batch_size} took: {elapsed_time} (s)")
                    results_dict[num_gpus][image_size][backend][batch_size] = elapsed_time

    # print
    print(results_dict)
    for num_gpus, v1 in results_dict.items():
        for image_size, v2 in v1.items():
            for backend, v3 in v2.items():
                out_stream: str = f"{image_size}-{batch_size}-{backend}-{num_gpus}"
                for batch_size, elapsed_time in v3.items():
                    out_stream += f"\n{elapsed_time}"
                print(out_stream)
                print("########")
