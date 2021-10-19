''' Visual attention model '''


import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.layers import GradientRescaler, GradientEquilibriumModule
from utils.utils import affparam2st, st2param, affparam2mat


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, mode='fan_out')
    elif isinstance(module, nn.BatchNorm2d):
        module.weight.data.fill_(1)
        module.bias.data.zero_()
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        module.bias.data.zero_()


class GlimpseGenerator(nn.Module):
    def __init__(self, down_factor):
        super(GlimpseGenerator, self).__init__()
        # TODO save as buffer
        self.down_factor = down_factor

    def forward(self, img, aff_param):
        assert img.dim() == 4
        down_sz = int(img.size(2) // self.down_factor)

        aff_mat = affparam2mat(aff_param)

        glimpse_region = F.affine_grid(
            aff_mat,
            (*img.size()[:2], down_sz, down_sz),
            align_corners=True
        ).type_as(img)
        return F.grid_sample(img, glimpse_region, align_corners=True)


class FeatureFusionModule(nn.Module):
    # TODO rewrite a more efficient module
    # since we can reuse some hidden states during iterations
    def __init__(self, dim, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.mha = nn.MultiheadAttention(dim, 1, **kwargs)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.LayerNorm(dim)

        for name, weight in self.mha.named_parameters():
            if weight.dim() == 2:
                assert 'proj' in name, name
                nn.init.kaiming_normal_(weight.data)
            elif weight.dim() == 1:
                assert 'bias' in name, name
                weight.data.zero_()

    def forward(self, *args, **kwargs):
        x = self.mha(*args, **kwargs)[0]
        x = self.norm(x)[-1]
        x = self.relu(x)
        return x


class LocalizationNetwork(nn.Module):
    def __init__(self, dim_in, dim_out, hparams):
        super(LocalizationNetwork, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        self.sigmoid = nn.Sigmoid()

        # Static gradient re-scaling
        self.grad_rescaler = GradientRescaler.apply
        self.grad_factor = torch.tensor(hparams.s)  # TODO save as buffer

        self.hparams = hparams
        initialize_weights(self.fc)

    def init_affine_fc(self, m):
        ''' Good when using grad re-scaling method
            very bad when using graidient equilibrium
        '''
        m.weight.data.zero_()
        m.bias.data.copy_(self.identity_affparam())

    def forward(self, x):
        aff_param = self.fc(x)
        aff_param = self.sigmoid(aff_param)
        aff_param = self.grad_rescaler(aff_param, self.grad_factor)

        scale, trans = affparam2st(aff_param)
        scale = scale * self.hparams.scale_range + self.hparams.scale_min

        if self.hparams.trans_method == 'tight-crop':
            trans = 2 * (trans - 0.5) * (1 - scale)
        elif self.hparams.trans_method == 'center-invariant':
            trans = 2 * (trans - 0.5) * \
                ((1 / self.hparams.scale_min) - 1) * scale
        else:
            assert self.hparams.trans_method == 'naive'
            trans = trans * self.hparams.trans_range + self.hparams.trans_min
        return st2param(scale, trans)


class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(dim_in, dim_out)
        initialize_weights(self.fc)

    def forward(self, x):
        return self.fc(x)


class MGNet(nn.Module):
    def __init__(self, hparams, backbone, dim):
        super(MGNet, self).__init__()

        self.backbone = backbone

        self.glimpse_gen = GlimpseGenerator(hparams.scale)
        self.feat_fusion = FeatureFusionModule(dim+3, dropout=hparams.dropout)
        self.loc_network = LocalizationNetwork(dim+3, 3, hparams)

        self.global_fc = Classifier(dim+3, hparams.num_class)
        if hparams.aux:
            self.glimpse_fc = Classifier(dim, hparams.num_class)

        # Adaptive gradient re-scaling
        # If ge is None, then this module only visualize the gradient
        self.gem = GradientEquilibriumModule(
            hparams.n_iter, hparams.ge, hparams.ge_final)

        self.hparams = hparams
        # self.apply(initialize_weights)

    def identity_affparam(self):
        return torch.tensor([1.0, 0.0, 0.0])

    def glimpse_feature(self, img, aff_param):
        glimpse = self.glimpse_gen(img, aff_param)
        return self.backbone(glimpse)

    def feature_fusion(self, x):
        return self.feat_fusion(x, x, x, need_weights=False)

    def decode(self, n_affparam, n_feature):
        if not self.hparams.no_detach:
            n_affparam = n_affparam.detach()
        if self.hparams.no_spatial_clue:
            n_affparam = torch.zeros_like(n_affparam)
        # H_t in manuscript
        n_feature = torch.cat([n_feature, n_affparam], dim=2)

        out = self.feature_fusion(n_feature)

        # If hparams.ge is False, gem do nothing but just visualize the gradient
        i_iter = n_feature.size(0) - 1
        out_fix = self.gem(out, i_iter-1, fix=True)
        out_nofix = self.gem(out, i_iter, fix=False)

        logits = self.global_fc(out_fix)
        aff_param = self.loc_network(out_nofix)

        return logits, aff_param

    def grad_rescale(self):
        assert self.gem.stds.norm() != 0.0
        scale = self.gem.scale()
        for param in self.affparam_fc.parameters():
            param.grad /= scale

    def gen_ssl_affparam(self, aff_param):

        assert aff_param.dim() == 2
        assert aff_param.size(1) == 3

        s, x, y = aff_param.t()

        trans_bound = 1-s

        dx, dy = 2*(torch.rand((2, *s.size()), device=aff_param.device)-0.5) * \
            trans_bound.unsqueeze(0) * self.hparams.ssl_explore
        x, y = x+dx, y+dy

        x = torch.min(torch.max(x, -trans_bound), trans_bound)
        y = torch.min(torch.max(y, -trans_bound), trans_bound)

        ssl_affparam = torch.stack([s, x, y], dim=1)
        assert ssl_affparam.size() == aff_param.size()
        return ssl_affparam

    def _forward(self, img):

        # A zero-dimensional tensor that is concatenatable with any tensor
        n_feature, n_logit, n_affparam = [torch.Tensor().type_as(img)] * 3

        for i in range(self.hparams.n_iter):
            if i == 0:
                aff_param = self.identity_affparam().repeat(img.size(0), 1).type_as(img)
            else:
                aff_param = next_aff_param

            # h_t in our manuscript
            feature = self.glimpse_feature(img, aff_param)

            n_affparam = torch.cat([n_affparam, aff_param.unsqueeze(0)])
            n_feature = torch.cat([n_feature, feature.unsqueeze(0)])

            logits, next_aff_param = self.decode(n_affparam, n_feature)

            n_logit = torch.cat([n_logit, logits.unsqueeze(0)])

        return {
            'n_logit': n_logit,
            'n_affparam': n_affparam,
            'n_feature': n_feature,
        }

    def _forward_ssl(self, img, n_affparam, n_feature, **_kwargs):
        with torch.no_grad():
            ssl_affparam = self.gen_ssl_affparam(n_affparam[-1])

            feature = self.glimpse_feature(img, ssl_affparam)

            ssl_logits, _ = self.decode(
                torch.cat([n_affparam[:-1], ssl_affparam.unsqueeze(0)]),
                torch.cat([n_feature[:-1], feature.unsqueeze(0)]).detach()
            )

        return {
            'ssl_logits': ssl_logits,
            'ssl_affparam': ssl_affparam,
        }

    def _forward_aux(self, n_feature, **_kwargs):
        return {
            'n_aux_logit': torch.stack([
                self.glimpse_fc(feature)
                for feature in n_feature[1:]
            ])
        }

    def forward(self, x):
        self.gem.reset()
        res = self._forward(x)
        if self.training:
            if self.hparams.aux:
                res_ = self._forward_aux(**res)
                res.update(res_)
            if self.hparams.ssl:
                res_ = self._forward_ssl(x, **res)
                res.update(res_)
        return res
