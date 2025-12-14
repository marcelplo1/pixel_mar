import os
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from models import DenoisingMLP
from utils import save_img_as_fig, unpatchify

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        self.denoising_net = DenoisingMLP(
            input_dim=args.channels * args.patch_size**2,
            output_dim=args.channels * args.patch_size**2,
            number_classes=args.class_num,
            seq_len=(args.img_size // args.patch_size) ** 2,
            batch_size=args.gen_batch_size,
            hidden_dim=256
        )

        self.img_size = args.img_size
        self.num_classes = args.class_num

        # self.label_drop_prob = args.label_drop_prob
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        #self.ema_decay1 = args.ema_decay1
        #self.ema_decay2 = args.ema_decay2
        #self.ema_params1 = None
        #self.ema_params2 = None

        # generation hyper params
        self.method = args.sampling_method
        self.steps = args.num_timesteps
        #self.cfg_scale = args.cfg
        #self.cfg_interval = (args.interval_min, args.interval_max)

        self.channels = args.channels
        self.patch_size = args.patch_size
        self.is_debug = args.is_debug
        self.output_dir = args.output_dir


    def drop_labels(self, labels):
        drop = torch.rand(labels.shape[0], device=labels.device) < self.label_drop_prob
        out = torch.where(drop, torch.full_like(labels, self.num_classes), labels)
        return out

    def sample_t(self, n: int, device=None):
        t = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(t)

    def forward(self, x, z, labels):
        #labels = self.drop_labels(labels) if self.training else labels

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        xt = t * x + (1 - t) * e
        v = (x - xt) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.denoising_net(xt, z, t.flatten(), labels)

        v_pred = (x_pred - xt) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2
        loss = loss.mean()

        if self.is_debug:
            save_img_as_fig(unpatchify(x.reshape(x.shape[0], x.shape[1], -1), self.patch_size , x.shape[1], self.channels), filename="ground_truth.png", path=self.output_dir)
            save_img_as_fig(unpatchify(x_pred.reshape(x.shape[0], x.shape[1], -1), self.patch_size, x.shape[1], self.channels), filename="prediction.png", path=self.output_dir)

        return loss

    @torch.no_grad()
    def generate(self, xt, z, labels):
        device = labels.device
        bsz = labels.size(0)
        #xt = self.noise_scale * torch.randn(bsz, self.channels, self.img_size, self.img_size, device=device)
        timesteps = torch.linspace(self.t_eps, 1.0 - self.t_eps, self.steps+1, device=device).view(-1, *([1] * xt.ndim)).expand(-1, bsz, -1, -1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps - 1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            xt = stepper(xt, z, t, t_next, labels)
        # last step euler
        xt = self._euler_step(xt, z, timesteps[-2], timesteps[-1], labels)
        return xt

    @torch.no_grad()
    def _forward_sample(self, xt, z, t, labels):
        # conditional
        x_cond = self.denoising_net(xt, z, t.flatten(), labels)
        v_cond = (x_cond - xt) / (1.0 - t).clamp_min(self.t_eps)

        # unconditional
        #x_uncond = self.denoising_net(xt, z, t.flatten(), torch.full_like(labels, self.num_classes))
        #v_uncond = (x_uncond - xt) / (1.0 - t).clamp_min(self.t_eps)

        # cfg interval
        # low, high = self.cfg_interval
        # interval_mask = (t < high) & ((low == 0) | (t > low))
        # cfg_scale_interval = torch.where(interval_mask, self.cfg_scale, 1.0)

        #return v_uncond + cfg_scale_interval * (v_cond - v_uncond)
        return v_cond

    @torch.no_grad()
    def _euler_step(self, xt, z, t, t_next, labels):
        v_pred = self._forward_sample(xt, z, t, labels)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def _heun_step(self, xt, z, t, t_next, labels):
        v_pred_t = self._forward_sample(xt, z, t, labels)

        xt_next_euler = xt + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(xt_next_euler, z, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params1, source_params):
            targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
        for targ, src in zip(self.ema_params2, source_params):
            targ.detach().mul_(self.ema_decay2).add_(src, alpha=1 - self.ema_decay2)