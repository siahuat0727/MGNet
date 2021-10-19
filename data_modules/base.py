import abc

from torch.utils.data import DataLoader
import pytorch_lightning as pl


class BaseDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    @abc.abstractproperty
    def train_dataset(self):
        pass

    @abc.abstractproperty
    def val_dataset(self):
        pass

    @property
    def train_size(self):
        return len(self.train_dataset)

    @property
    def val_size(self):
        return len(self.val_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers, shuffle=True, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        return self._val_dataloader()

    def shuffle_val_dataloader(self):
        return self._val_dataloader(shuffle=True)

    def _val_dataloader(self, shuffle=False):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.workers, shuffle=shuffle, pin_memory=True)

    def test_dataloader(self):
        return self.val_dataloader()
