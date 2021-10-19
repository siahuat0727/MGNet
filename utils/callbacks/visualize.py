from argparse import Namespace
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.callbacks.base import Callback

from utils.utils import denormalizes, draw_rect, check_is_better
from utils.color import Color


class ValVisualize(Callback):
    """
    Add model graph
    """

    def __init__(self, n_img=32, plot_per_val=1):
        self.count_val = 0
        self._train_batch = None
        self._test_batch = None
        self.device = None
        self.train_dataloader = None
        self.test_dataloader = None
        self.n_img = n_img
        self.plot_per_val = plot_per_val

    @property
    def train_batch(self):
        if self._train_batch is None:
            images, labels = next(iter(self.train_dataloader()))
            self._train_batch = images[:self.n_img].to(
                self.device), labels[:self.n_img].to(self.device)
        return self._train_batch

    @property
    def test_batch(self):
        if self._test_batch is None:
            images, labels = next(iter(self.test_dataloader()))
            self._test_batch = images[:self.n_img].to(
                self.device), labels[:self.n_img].to(self.device)
        return self._test_batch

    def plot_images(self, pl_module, batch, stage):
        # xx_images: high resolution
        # xx_imgs  : low resolution
        model = pl_module.model
        with torch.no_grad():
            images, labels = batch
            res = Namespace(**model(images))
            assert res.n_affparam.grad_fn is None

        ori_images = denormalizes(images, pl_module.dm.mean, pl_module.dm.std)

        # TODO directly use ori_images, no need to denormalize again?
        n_crop_imgs = torch.stack([
            model.glimpse_gen(ori_images, aff_param)
            for aff_param in res.n_affparam
        ], dim=0)

        n_rect_images = torch.stack([
            draw_rect(ori_images.clone(), aff_param)
            for aff_param in res.n_affparam
        ], dim=0)

        if pl_module.hparams.ssl:
            n_crop_imgs = torch.cat([
                n_crop_imgs, model.glimpse_gen(ori_images, res.ssl_affparam).unsqueeze(0)
            ], dim=0)

            is_better = check_is_better(
                res.n_logit[-1].detach(), res.ssl_logits.detach(), labels)

            colors = [
                Color.orange if better else Color.red
                for better in is_better
            ]

            n_rect_images = torch.cat([
                n_rect_images, draw_rect(
                    ori_images.clone(), res.ssl_affparam, colors=colors).unsqueeze(0)
            ], dim=0)

        images = torch.stack([  # Better name  # TODO use cat? then no need to change view later
            torch.stack([rect_images, self.upsampling(
                crop_imgs, pl_module.dm.img_sz)], dim=1)
            for rect_images, crop_imgs in zip(n_rect_images, n_crop_imgs)
        ], dim=2)

        nrow = pl_module.hparams.n_iter + int(pl_module.hparams.ssl)

        for i, rect_images in enumerate(n_rect_images):
            self.grid_plot(pl_module, f'{stage}_images_box{i+1}', rect_images)

        for i, images_ in enumerate(images):
            images_ = images_.view(-1, *images_.size()[2:])
            self.grid_plot(
                pl_module, f'{stage}_images_sample{i+1}', images_, nrow=nrow)

        self.add_histogram(
            pl_module, f'{stage}_scale', res.n_affparam[-1][:, 0])
        self.add_histogram(
            pl_module, f'{stage}_trans_x', res.n_affparam[-1][:, 1])
        self.add_histogram(
            pl_module, f'{stage}_trans_y', res.n_affparam[-1][:, 2])

    def grid_plot(self, pl_module, tag, imgs, **kwargs):
        grid = torchvision.utils.make_grid(
            imgs, pad_value=0.5, **kwargs)
        pl_module.logger.experiment.add_image(tag, grid, self.count_val)

    def add_histogram(self, pl_module, name, tensor):
        pl_module.logger.experiment.add_histogram(
            name, tensor, self.count_val)

    def upsampling(self, imgs, size):
        do_upsampling = nn.Upsample(size=size, mode='nearest')
        imgs = do_upsampling(imgs)
        return imgs.repeat(1, 3//imgs.size(1), 1, 1)

    def on_validation_end(self, trainer, pl_module):
        # TODO don't do at here
        self.device = pl_module.device
        self.train_dataloader = pl_module.dm.train_dataloader
        self.test_dataloader = pl_module.dm.shuffle_val_dataloader

        if self.count_val % self.plot_per_val == 0:
            self.plot_images(pl_module, self.train_batch, 'train')
            self.plot_images(pl_module, self.test_batch, 'test')
        self.count_val += 1
