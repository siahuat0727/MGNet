import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as _MNIST

from .base import BaseDataModule
from utils import lazy_property


class MNIST_toy(BaseDataModule):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = (0.1307,)
        self.std = (0.3081,)

    # @lazy_property
    @property
    def train_dataset(self):
        return _MNIST(
            root=self.hparams.train_dir,
            train=True,
            transform=self.train_transform(),
            download=True
        )

    # @lazy_property
    @property
    def val_dataset(self):
        return _MNIST(
            root=self.hparams.val_dir,
            train=False,
            transform=self.val_transform(),
            download=True
        )

    def train_transform(self):
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

    def val_transform(self):
        return self.train_transform()
