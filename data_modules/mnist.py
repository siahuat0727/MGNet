import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST as _MNIST

from .base import BaseDataModule
from utils import lazy_property




class MNIST(BaseDataModule):

    def __init__(self, hparams):
        super().__init__(hparams)
        self.mean = (0.1307,)
        self.std = (0.3081,)
        self.img_sz = 112
        self.gaussian = hparams.gaussian

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

    def add_gaussian_noise(self, x):
        return x.add(self.gaussian*torch.randn_like(x)).clamp(x.min(), x.max())

    def train_transform(self):
        pad = int((self.img_sz - 28) / 2)
        max_trans = pad / self.img_sz
        max_trans = (max_trans, max_trans)

        return transforms.Compose([
            transforms.Pad(pad),
            transforms.RandomAffine(0, translate=max_trans),
            transforms.ToTensor(),
            transforms.Lambda(self.add_gaussian_noise),
            transforms.Normalize(self.mean, self.std)
        ])

    def val_transform(self):
        return self.train_transform()
