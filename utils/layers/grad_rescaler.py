import torch


class GradientRescaler(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, ratio):
        ctx.save_for_backward(ratio)
        return input

    @staticmethod
    def backward(ctx, grad_output):
        ratio, = ctx.saved_tensors
        grad_input = grad_output.clone() * ratio
        return grad_input, None
