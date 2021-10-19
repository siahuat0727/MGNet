from __future__ import print_function

from argparse import Namespace
from itertools import combinations

import torch
import torchvision
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, ReduceLROnPlateau
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.metrics.functional.classification import accuracy

from utils.utils import check_is_better, iou_loss, affparam2rect
from utils.layers import Identity
from utils.attacks import PGD

import models
from mgnet import MGNet


def get_model(hparams):
    if hparams.model == 'vanilla':
        model = models.__dict__['VanillaCNN']()
    else:
        model = torchvision.models.__dict__[hparams.model](
            pretrained=hparams.pretrained)

    # Replace fc to identity, treat it as backbone
    dim = model.fc.in_features
    model.fc = Identity()

    return MGNet(hparams, model, dim)


class LitModule(pl.LightningModule):
    def __init__(self, hparams, datamodule):
        super().__init__()
        self.hparams = hparams
        self.criterion_n = nn.CrossEntropyLoss(reduction='none')
        self.mseloss_n = nn.MSELoss(reduction='none')

        self.model = get_model(hparams)
        self.dm = datamodule
        self.train_size = self.dm.train_size
        self.val_size = self.dm.val_size
        self.grad_scale = torch.tensor(0.0)
        self.accs = torch.nn.Sequential(*[
            Accuracy(compute_on_step=False)
            for _ in range(self.hparams.n_iter)
        ])

        if hparams.adv:
            self.attacker = PGD(**vars(hparams))


    def forward(self, batch):
        images, labels = batch
        if self.hparams.adv:
            images = self.attacker.attack(self.model, images, labels)

        res = self.model(images)

        ce_loss, n_b_ce_loss, n_mask = self.get_ce_loss(res['n_logit'], labels)
        aux_loss = self.maybe_get_aux_loss(res, labels, n_mask)
        ss_loss = self.maybe_get_ss_loss(res, n_b_ce_loss)
        # We didn't apply the below losses in our BMVC2021 manuscript
        # left it here for further exploration
        ssl_loss, ssl_ratio = self.maybe_get_ssl_loss(res, labels)
        div_loss = self.maybe_get_div_loss(res)

        loss = (1 - self.hparams.alpha) * ce_loss
        loss += self.hparams.alpha * aux_loss
        loss += self.hparams.ss_coef * ss_loss
        loss += self.hparams.ssl_coef * ssl_loss  # TODO need average?
        loss += self.hparams.div_coef * div_loss

        return Namespace(**{
            'loss': loss,
            'n_ce_loss': n_b_ce_loss.detach().mean(dim=1),
            'n_logit': res['n_logit'],
            'labels': labels,
            'n_mask': n_mask,
            'ssl_ratio': ssl_ratio,
            'ssl_loss': ssl_loss,
            'ss_loss': ss_loss,
        })

    def on_after_backward(self):
        if self.hparams.ge is not None:
            self.model.grad_rescale()
        self.grad_scale = self.model.gem.scale()

    def training_step(self, batch, batch_nb):
        res = self(batch)

        n_acc = [
            accuracy(pred, res.labels)
            for pred in res.n_logit
        ]

        # Measure the weighted of each coefficient
        n_coef = res.n_mask.sum(dim=1) / res.n_mask.size(1)

        logs = {
            'loss/train': res.loss.item(),
            'loss/train_ssl': res.ssl_loss.item(),
            'loss/train_small_crop': res.ss_loss.item(),
            **{
                f'loss/train_ce_iter_{i+1}': loss_
                for i, loss_ in enumerate(res.n_ce_loss)
            },
            **{
                f'acc/train_iter_{i+1}': acc
                for i, acc in enumerate(n_acc)
            },
            **{
                f'acc_gain/train_acc{i+1}/acc{i}': acc_hi/acc_lo
                for i, (acc_lo, acc_hi) in enumerate(zip(n_acc, n_acc[1:]), 1)
            },
            **{
                f'iter/weight_iter{i+1}': coef
                for i, coef in enumerate(n_coef)
            },
            'ssl/ratio': res.ssl_ratio,
            'grad/apply_scale': self.grad_scale,
            'grad/equil_ratio': self.model.gem.ratio,
            **{
                f'grad/std_{i+1}': std
                for i, std in enumerate(self.model.gem.stds)
            },
            **{
                f'grad/scale_{i+1}': scale
                for i, scale in enumerate(self.model.gem.grad_scales)
            },
        }
        self.model.gem.update_ratio(
            self.current_epoch, self.hparams.epochs)
        self.log_dict(logs)
        return res.loss

    def validation_step(self, batch, batch_nb):
        res = self(batch)

        for pred, acc in zip(res.n_logit, self.accs):
            acc.update(pred, res.labels)

        logs = {
            'loss/val': res.loss,
            **{
                f'loss/val_ce_{i+1}': loss_
                for i, loss_ in enumerate(res.n_ce_loss)
            },
        }
        self.log_dict(logs)

    def validation_epoch_end(self, outs):
        accs = [acc.compute() for acc in self.accs]
        for acc in self.accs:
            acc.reset()

        val_accs = {
            f'acc/val_{i}': acc
            for i, acc in enumerate(accs, 1)
        }
        logs = {
            'acc/val': accs[-1],
            **{
                f'acc_gain/val_acc{i+1}/acc{i}': accs[i] / accs[i-1]
                for i in range(1, self.hparams.n_iter)
            },
        }
        self.log_dict(val_accs, prog_bar=True)
        self.log_dict(logs)
        self.log('checkpoint_on', logs['acc/val'])

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outs):
        return self.validation_epoch_end(outs)

    def configure_optimizers(self):

        # TODO clean code
        steps_per_epoch = (
            self.train_size//self.hparams.batch_size) // self.hparams.world_size

        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.hparams.learning_rate,
                                        weight_decay=self.hparams.weight_decay,
                                        momentum=0.9)
        else:
            assert self.hparams.optimizer == 'adam'
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.hparams.learning_rate,
                                         weight_decay=self.hparams.weight_decay)

        if self.hparams.scheduler == 'one-cycle':
            scheduler = OneCycleLR(optimizer,
                                   self.hparams.learning_rate,
                                   steps_per_epoch=steps_per_epoch,
                                   epochs=self.hparams.epochs,
                                   pct_start=0.2)
        elif self.hparams.scheduler == 'cosine':
            tmax = self.hparams.epochs * steps_per_epoch
            scheduler = CosineAnnealingLR(optimizer, T_max=tmax)
        elif self.hparams.scheduler == 'multi-step':
            milestones = [e * steps_per_epoch for e in self.hparams.schedule]
            scheduler = MultiStepLR(optimizer, milestones=milestones)
        else:
            assert self.hparams.scheduler == 'reduce'
            scheduler = ReduceLROnPlateau(optimizer)

        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'name': 'learning_rate'
        }
        return [optimizer], [scheduler]

    def get_n_mask(self, n_b_ce_loss):
        # TODO if delete sbs, then no need to pass n_b_ce_loss here
        n_mask = torch.ones_like(n_b_ce_loss).bool()

        if self.hparams.step_by_step:
            for i in range(1, self.hparams.n_iter):
                n_mask[i] = n_mask[i-1] & (n_b_ce_loss[i] <= n_b_ce_loss[i-1])

            mask = torch.ones_like(n_b_ce_loss[0]).bool()
            for i in reversed(range(self.hparams.n_iter - 1)):
                mask &= (~n_mask[i+1])
                n_mask[i] &= mask

            n_mask = n_mask.float()
            n_mask += torch.cat([torch.zeros_like(n_mask[-1:]),
                                 n_mask[:-1]], dim=0)
        else:
            assert self.hparams.n_iter_coef is not None
            n_mask = n_mask.float() * torch.tensor(self.hparams.n_iter_coef,
                                                   device=self.device).float().unsqueeze(1)

        assert n_mask.size() == (self.hparams.n_iter, n_b_ce_loss.size(1))
        n_mask = self._n_mask_normalize(n_mask)

        return n_mask

    def _n_mask_normalize(self, n_mask):
        # Normalize through iteration-dim, make each sample equally important
        return n_mask / n_mask.sum(dim=0)

    def get_ce_loss(self, n_pred, labels):

        n_b_ce_loss = torch.stack([
            self.criterion_n(pred, labels)
            for pred in n_pred
        ])

        n_mask = self.get_n_mask(n_b_ce_loss)
        ce_loss = (n_mask * n_b_ce_loss).sum(dim=0).mean()

        return ce_loss, n_b_ce_loss, n_mask

    def maybe_get_ssl_loss(self, res, labels):
        ssl_loss = torch.tensor(0.0, device=self.device)
        ratio = torch.tensor(0.0, device=self.device)

        if res.get('ssl_logits') is not None:
            logits = res['n_logit'][-1].detach()
            is_better = check_is_better(logits, res['ssl_logits'], labels)
            who_better = is_better.nonzero(as_tuple=False)
            ratio = is_better.float().mean()

            if who_better.nelement() > 0:
                ssl_loss += ratio * \
                    self.mseloss_n(res['n_affparam'][-1], res['ssl_affparam'])[
                        who_better].mean()
        return ssl_loss, ratio

    def maybe_get_ss_loss(self, res, n_b_ce_loss):
        ss_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.ss:

            is_better = n_b_ce_loss.detach()[-1] < self.hparams.ss_threshold
            who_better = is_better.nonzero(as_tuple=False)
            ratio = is_better.float().mean()

            if who_better.nelement() > 0:
                # a^s in our manuscript
                s, _, _ = res['n_affparam'][-1].t()
                ss_loss += ratio * \
                    (s - self.hparams.scale_min)[who_better].pow(2).mean()
        return ss_loss

    def maybe_get_aux_loss(self, res, labels, n_mask):
        aux_loss = torch.tensor(0.0, device=self.device)

        if res.get('n_aux_logit') is not None:
            assert res['n_aux_logit'].size(0) == res['n_logit'].size(0) - 1
            n_mask = n_mask[1:]
            n_mask = self._n_mask_normalize(n_mask)

            n_b_aux_loss = torch.stack([
                self.criterion_n(pred, labels)
                for pred in res['n_aux_logit']
            ])

            aux_loss += (n_mask * n_b_aux_loss).sum(dim=0).mean()

        return aux_loss

    def maybe_get_div_loss(self, res):
        div_loss = torch.tensor(0.0, device=self.device)

        if self.hparams.div:

            n_rect = torch.stack([
                affparam2rect(aff_param)
                for aff_param in res['n_affparam'][1:]
            ])

            div_loss = torch.stack([
                iou_loss(n_rect[r1], n_rect[r2])
                for r1, r2 in combinations(range(n_rect.size(0)), 2)
            ]).mean()

        return div_loss
