import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.utils import save_img_as_fig, unpatchify

class Denoiser(nn.Module):
    def __init__(
        self,
        denoising_model,
        output_dir,
        sampling_method = 'euler',
        pred_type = 'v',
        diffusion_batch_mul=4,
        num_timesteps = 100,
        sample_t_mean = 0.0,
        sample_t_std = 1.0,
        t_eps = 1e-2,
        t_eps_sample = 1e-5,
        noise_scale = 1.0,
        ema_decay = 0.9999,
        use_logging=False,
        train_shift=None,
        sample_shift=None,
        t_sampling_method='logit_normal'
    ):
        super().__init__()

        self.denoising_net = denoising_model

        self.img_size = denoising_model.img_size
        self.channels = denoising_model.in_channels
        self.patch_size = denoising_model.patch_size

        self.t_sampling_method = t_sampling_method
        self.P_mean = sample_t_mean
        self.P_std = sample_t_std
        self.t_eps = t_eps
        self.t_eps_sample = t_eps_sample
        self.noise_scale = noise_scale
        self.method = sampling_method
        self.steps = num_timesteps

        self.use_logging = use_logging
        self.output_dir = output_dir

        self.pred_type = pred_type
        self.ema_decays = ema_decay if isinstance(ema_decay, list) else [ema_decay]
        self.diffusion_batch_mul = diffusion_batch_mul

        self.train_shift = train_shift
        self.sample_shift = sample_shift
        self.ema_params_list = None


    def _shift_t(self, t, shift):
        """Apply time shift (t=0 clean, t=1 noise)."""
        t = shift * t / (1.0 + (shift - 1.0) * t)
        return t

    def sample_t_plateau_logit_normal(self, n: int, device=None):
        """Plateau-logit-normal, uniform plateau [mode, 1-eps]."""
        noise = torch.randn(n, device=device)
        t_gaussian = noise * self.P_std + self.P_mean
        eps = 1e-3

        t_logit = torch.sigmoid(t_gaussian)
        t_star = torch.sigmoid(torch.tensor(self.P_mean, device=device))

        t_uniform = torch.rand(n, device=device) * (1.0 - eps - t_star) + t_star
        t = torch.where(t_logit > t_star, t_uniform, t_logit)

        if self.train_shift is not None:
            return self._shift_t(t, self.train_shift)
        return t

    def sample_t(self, n: int, device=None):
        if self.t_sampling_method == 'uniform':
            return self.sample_t_uniform(n, device)
        elif self.t_sampling_method == 'logit_normal':
            return self.sample_t_logit_normal(n, device)
        elif self.t_sampling_method == 'plateau_logit_normal':
            return self.sample_t_plateau_logit_normal(n, device)
        else:
            raise ValueError(f"Unknown t_sampling_method: '{self.t_sampling_method}'.")

    def sample_t_uniform(self, n: int, device=None):
        """Uniform timestep sampling over [0, 1]."""
        t = torch.rand(n, device=device)
        if self.train_shift is not None:
            return self._shift_t(t, self.train_shift)
        return t

    def sample_t_logit_normal(self, n: int, device=None):
        """Logit-normal"""
        t = torch.randn(n, device=device) * self.P_std + self.P_mean
        t = torch.sigmoid(t)
        if self.train_shift is not None:
            return self._shift_t(t, self.train_shift)
        return t

    def forward(self, x, z, mask, labels):
        B, N, D = x.shape
        x = x.reshape(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        labels = labels.repeat(self.diffusion_batch_mul*N)
        mask = mask.reshape(B*N).repeat(self.diffusion_batch_mul)

        z_input = z.reshape(B*N, -1).repeat(self.diffusion_batch_mul, 1)

        if self.training:
            t_eps = self.t_eps
        else:
            t_eps = self.t_eps_sample

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        xt = (1 - t) * x + t * e
        if self.pred_type == 'x':
            v = (xt - x) / t.clamp_min(t_eps)
        elif self.pred_type == 'v':
            v = e - x

        pred = self.denoising_net(xt, z_input, t.flatten())

        if self.pred_type == 'x':
            v_pred = (xt - pred) / t.clamp_min(t_eps)
        elif self.pred_type == 'v':
            v_pred = pred

        loss = (v - v_pred) ** 2

        if mask is not None:
            loss = loss.view(-1, N, D)
            mask = mask.view(-1, N, 1)
            loss = ((loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * D)).mean()
        return loss

    @torch.no_grad()
    def generate(self, xt, z, labels, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        device = z.device
        bsz = xt.size(0)
        timesteps = torch.linspace(1.0 - self.t_eps_sample, self.t_eps_sample, self.steps+1, device=device)
        if self.sample_shift is not None:
            timesteps = self._shift_t(timesteps, self.sample_shift)

        timesteps = timesteps.view(-1, 1, 1).expand(-1, bsz, 1)

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps-1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            xt = stepper(xt, z, t, t_next, z_uncond, cfg_scale, cfg_interval)
        xt = self._euler_step(xt, z, timesteps[-2], timesteps[-1], z_uncond, cfg_scale, cfg_interval)
        return xt

    def _pred_to_velocity(self, pred, xt, t):
        if self.pred_type == 'v':
            return pred
        elif self.pred_type == 'x':
            return (xt - pred) / t.clamp_min(self.t_eps)

    @torch.no_grad()
    def _forward_sample(self, xt, z, t, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        pred_cond = self.denoising_net(xt, z, t.view(-1))
        v_cond = self._pred_to_velocity(pred_cond, xt, t)

        if cfg_scale == 1.0 or z_uncond is None:
            return v_cond

        pred_uncond = self.denoising_net(xt, z_uncond, t.view(-1))
        v_uncond = self._pred_to_velocity(pred_uncond, xt, t)

        low, high = cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(interval_mask, cfg_scale, 1.0)

        return v_uncond + scale * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, xt, z, t, t_next, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        v_pred = self._forward_sample(xt, z, t, z_uncond, cfg_scale, cfg_interval)
        return xt + (t_next - t) * v_pred

    @torch.no_grad()
    def _heun_step(self, xt, z, t, t_next, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        v_pred_t = self._forward_sample(xt, z, t, z_uncond, cfg_scale, cfg_interval)
        xt_next_euler = xt + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(xt_next_euler, z, t_next, z_uncond, cfg_scale, cfg_interval)
        return xt + (t_next - t) * 0.5 * (v_pred_t + v_pred_t_next)

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for ema_params, decay in zip(self.ema_params_list, self.ema_decays):
            for targ, src in zip(ema_params, source_params):
                targ.detach().mul_(decay).add_(src, alpha=1 - decay)