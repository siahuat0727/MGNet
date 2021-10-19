import os
from pathlib import Path

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from utils import lazy_property
from utils.autoaugment import ImageNetPolicy
from .base import BaseDataModule


class ImageNet(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def mean(self):
        return (0.485, 0.456, 0.406)

    @property
    def std(self):
        return (0.229, 0.224, 0.225)

    @property
    def train_dataset(self):
        return ImageFolder(
            root=self.hparams.train_dir,
            transform=self.train_transform(),
        )

    @property
    def val_dataset(self):
        return ImageFolder(
            root=self.hparams.val_dir,
            transform=self.val_transform(),
        )

    @property
    def img_sz_(self):
        return 256

    @property
    def img_sz(self):
        return 224

    def train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.img_sz),
            transforms.RandomHorizontalFlip(),
            ImageNetPolicy(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

    def val_transform(self):
        return transforms.Compose([
            transforms.Resize(self.img_sz_),
            transforms.CenterCrop(self.img_sz),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])
