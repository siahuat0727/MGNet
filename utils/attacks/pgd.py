import torch
import torch.nn as nn


class PGD:
    """
    This is the multi-step version of FGSM attack.
    """

    def __init__(self, loss_fn=nn.CrossEntropyLoss(), eps=0.03, step_size=0.01, step_k=5, adv_all=False, **_):
        self.eps = eps
        self.step_k = step_k
        self.step_size = step_size
        self.loss_fn = loss_fn
        self.adv_all = adv_all

    def attack(self, model, img, label):
        img_ = img.clone()
        img_.requires_grad = True

        torch.set_grad_enabled(True)
        assert model.training == False
        for _ in range(self.step_k):

            # print(_, img_.grad)
            pred = model(img_)['n_logit'][-1]
            loss = self.loss_fn(pred, label)
            loss.backward()

            with torch.no_grad():
                img_ += self.step_size * img_.grad.data.sign()

                eta = torch.clamp(img_ - img, -self.eps, self.eps)
                img_.data = img + eta
                img_.grad.zero_()

                # img_ = torch.clamp(img_, clip_min, clip_max)
                # img_ = img_.detach()
                # img_.requires_grad_()
        torch.set_grad_enabled(False)
        img_.requires_grad = False
        img_ = img_.data

        return img_
