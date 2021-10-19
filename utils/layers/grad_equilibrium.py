import math
import torch


class GradientEquilibrium(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, stds, scales, i, is_fix, ratio):
        # TODO: no need to save all? some can save as constant
        ctx.save_for_backward(i, stds, scales, is_fix, ratio)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        i, stds, scales, is_fix, ratio = ctx.saved_tensors
        # assert scales[i] == 0.0
        if is_fix:
            # assert stds[i] == 0.0
            stds[i] = grad_output.std() + 1e-12
            grad_input = grad_output.clone()
        else:
            # assert stds[i] != 0.0
            scales[i] = (grad_output.std() + 1e-12) / stds[i]
            # assert scales[i] != 0.0
            if ratio:
                grad_input = grad_output / (ratio * scales[i])
            else:
                grad_input = grad_output.clone()

        return grad_input, None, None, None, None, None


class GradientEquilibriumModule:
    def __init__(self, n_iter, ratio, final_ratio):
        self.n_iter = n_iter
        if ratio is None:
            self.init_ratio = torch.tensor(0.0)
            self.final_ratio = torch.tensor(0.0)
        else:
            self.init_ratio = torch.tensor(ratio)
            self.final_ratio = torch.tensor(final_ratio)
        self.ratio = self.init_ratio.clone()
        self.ge = GradientEquilibrium.apply

        self.stds = None
        self.grad_scales = None
        self.reset()

    def reset(self):
        self.stds = torch.zeros(self.n_iter-1)
        self.grad_scales = torch.zeros(self.n_iter-1)

    def update_ratio(self, epoch, total_epoch):
        r = (math.cos(math.pi * epoch / total_epoch)+1)/2
        self.ratio = self.final_ratio * (1-r) + \
            (self.init_ratio - self.final_ratio) * r

    def scale(self):
        return self.ratio * (self.stds * self.grad_scales).sum() / self.stds.sum()

    def __call__(self, input, i_iter=0, fix=False):
        if i_iter < 0:
            return input
        fix = torch.tensor(fix)
        i_iter = torch.tensor(i_iter)
        return self.ge(input, self.stds, self.grad_scales, i_iter, fix, self.ratio)
