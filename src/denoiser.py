import os
import numpy as np
import torch
import torch.nn as nn

from matplotlib import pyplot as plt

from model.denoising_model import denoising_models
from utils.utils import save_img_as_fig, unpatchify

class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        self.denoising_net = denoising_models[args.denoising_model](
            input_size=args.img_size,
            in_channels=args.channels,
            num_classes=args.class_num,
            mae_hidden_dim=args.mae_hidden_dim,
            bottleneck_dim=args.bottleneck_dim_final
        )

        self.img_size = args.img_size
        self.num_classes = args.class_num

        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        self.ema_params = None

        self.method = args.sampling_method
        self.steps = args.num_timesteps

        self.channels = args.channels
        self.patch_size = args.patch_size
        self.is_debug = args.is_debug
        self.output_dir = args.output_dir

    def sample_t(self, n: int, device=None):  # lognormal distribution
        t = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(t)

    def forward(self, x, z, mask):
        B, N, D = x.shape
        x = x.view(B*N, -1)
        z = z.reshape(B*N, -1)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        xt = t * x + (1 - t) * e
        v = (x - xt) / (1 - t).clamp_min(self.t_eps)

        x_pred = self.denoising_net(xt, z, t.flatten())

        v_pred = (x_pred - xt) / (1 - t).clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2

        if mask is not None:
            mask = mask.view(-1, 1)
            loss = (loss * mask).sum() / mask.sum()

        if self.is_debug:
            x = x.view(B, N, D)
            x_pred = x_pred.view(B, N, D)
            save_img_as_fig(unpatchify(x.reshape(x.shape[0], x.shape[1], -1), self.patch_size , x.shape[1], self.channels), filename="ground_truth.png", path=self.output_dir)
            save_img_as_fig(unpatchify(x_pred.reshape(x.shape[0], x.shape[1], -1), self.patch_size, x.shape[1], self.channels), filename="prediction.png", path=self.output_dir)

        return loss

    @torch.no_grad()
    def generate(self, xt, z):
        device = z.device
        bsz = xt.size(0)
        timesteps = torch.linspace(self.t_eps, 1.0 - self.t_eps, self.steps+1, device=device).view(-1, *([1] * xt.ndim)).expand(-1, bsz, -1)

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
            xt = stepper(xt, z, t, t_next)
        # last step euler
        xt = self._euler_step(xt, z, timesteps[-2], timesteps[-1])
        return xt

    @torch.no_grad()
    def _forward_sample(self, xt, z, t):
        x_pred = self.denoising_net(xt, z, t.flatten())
        v_pred = (x_pred - xt) / (1.0 - t).clamp_min(self.t_eps)
        return v_pred

    @torch.no_grad()
    def _euler_step(self, xt, z, t, t_next):
        v_pred = self._forward_sample(xt, z, t)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def _heun_step(self, xt, z, t, t_next):
        v_pred_t = self._forward_sample(xt, z, t)

        xt_next_euler = xt + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(xt_next_euler, z, t_next)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def update_ema(self):
        ema_decay = 0.9996
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)