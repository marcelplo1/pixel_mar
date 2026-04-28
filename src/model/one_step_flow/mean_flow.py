import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.func import jvp

from utils.utils import save_img_as_fig, unpatchify


class MeanFlow(nn.Module):
    """
    Per-token MeanFlow objective based on the idea of pixel-MeanFlow. 
    A fraction of tokens are pure flow-matching (r == t), the rest use the
    mean-flow identity v_t = u + (t - r) * du/dt computed via jvp.
    """
    def __init__(
        self,
        denoising_model,
        output_dir,
        sampling_method='euler',
        diffusion_batch_mul=1,
        num_timesteps=1,
        sample_t_mean=0.0,
        sample_t_std=1.0,
        t_eps=5e-2,
        t_eps_sample=1e-5,
        noise_scale=1.0,
        ema_decay=0.9999,
        use_logging=False,
        train_shift=None,
        sample_shift=None,
        data_proportion=0.5,
        norm_p=1.0,
        norm_eps=1e-2,
        t_sampling_method='logit_normal',
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
        self.diffusion_batch_mul = diffusion_batch_mul

        self.data_proportion = data_proportion
        self.norm_p = norm_p
        self.norm_eps = norm_eps

        self.use_logging = use_logging
        self.output_dir = output_dir

        self.train_shift = train_shift
        self.sample_shift = sample_shift
        self.ema_decays = ema_decay if isinstance(ema_decay, list) else [ema_decay]
        self.ema_params_list = None
        self.log_counter = 0
        self.log_batch_pred = 100

    def _shift_t(self, t, shift):
        return shift * t / (1.0 + (shift - 1.0) * t)

    def sample_t(self, n: int, device=None):
        if self.t_sampling_method == 'uniform':
            t = torch.rand(n, device=device)
        elif self.t_sampling_method == 'logit_normal':
            t = torch.randn(n, device=device) * self.P_std + self.P_mean
            t = torch.sigmoid(t)
        else:
            raise ValueError(f"Sampling method not supported.")
        if self.train_shift is not None:
            t = self._shift_t(t, self.train_shift)
        return t

    def sample_t_r(self, n: int, device=None):
        """Sample (t, r) per token. A data_proportion fraction is
        r == t (flow-matching mode), the remainder is sorted so that t >= r."""
        t = self.sample_t(n, device=device)
        r = self.sample_t(n, device=device)
        fm_mask = torch.rand(n, device=device) < self.data_proportion
        r = torch.where(fm_mask, t, r)
        t, r = torch.maximum(t, r), torch.minimum(t, r)
        return t, r, fm_mask


    def _adp_wt(self, loss):
        adp = (loss + self.norm_eps) ** self.norm_p
        return loss / adp.detach()

    def forward(self, x, z, mask, labels):
        B, N, D = x.shape
        x = x.reshape(B * N, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(B * N, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(B * N).repeat(self.diffusion_batch_mul)

        if self.training:
            t_eps = self.t_eps
        else:
            t_eps = self.t_eps_sample

        num_tokens = x.size(0)
        device = x.device
        t, r, fm_mask = self.sample_t_r(num_tokens, device=device)
        t_view = t.view(-1, 1)
        r_view = r.view(-1, 1)

        e = torch.randn_like(x) * self.noise_scale
        xt = (1 - t_view) * x + t_view * e
        vt = (xt - x) / t_view.clamp_min(t_eps)
        #vt = e - x

        # Network wrapper
        def u_fn(z_t_in, t_in, r_in):
            return self.denoising_net(z_t_in, z, t_in, t_in - r_in)

        dtdt = torch.ones_like(t)
        dtdr = torch.zeros_like(r)

        (u, v), (du_dt, _) = jvp(u_fn, (xt, t, r), (vt, dtdt, dtdr))

        # Compound MeanFlow target (du/dt detached because second order)
        V = u + (t_view - r_view) * du_dt.detach()
        v_target = vt.detach()

        # Per-element squared error.
        err_u = (V - v_target) ** 2
        err_v = (v - v_target) ** 2

        # Per-token squared error summed over channels and loss weightening (like pMF)
        loss_u = self._adp_wt(err_u.sum(dim=-1))
        loss_v = self._adp_wt(err_v.sum(dim=-1))

        loss_u = loss_u.view(-1, N)
        loss_v = loss_v.view(-1, N)
        mask_view = mask.view(-1, N)

        denom = mask_view.sum(dim=1).clamp_min(1.0)
        loss_u_final = ((loss_u * mask_view).sum(dim=1) / denom).mean()
        loss_v_final = ((loss_v * mask_view).sum(dim=1) / denom).mean()

        # Un-weighted per-element MSE on masked tokens for monitoring
        with torch.no_grad():
            mask_full = mask_view.unsqueeze(-1)
            n_elt = mask_full.sum().clamp_min(1.0) * D
            err_u_view = err_u.view(-1, N, D)
            err_v_view = err_v.view(-1, N, D)
            self.last_loss_u_raw = (err_u_view * mask_full).sum() / n_elt
            self.last_loss_v_raw = (err_v_view * mask_full).sum() / n_elt

        loss = loss_u_final + loss_v_final
        return loss

    @torch.no_grad()
    def generate(self, xt, z, labels, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0), token_positions=None):
        device = z.device
        B = xt.size(0)

        timesteps = torch.linspace(
            1.0 - self.t_eps_sample, self.t_eps_sample, self.steps + 1, device=device
        )
        if self.sample_shift is not None:
            timesteps = self._shift_t(timesteps, self.sample_shift)

        for i in range(self.steps):
            t_now = timesteps[i].expand(B)
            t_next = timesteps[i + 1].expand(B)
            h = t_now - t_next  # > 0

            u_cond, _ = self.denoising_net(xt, z, t_now, h)

            if cfg_scale != 1.0 and z_uncond is not None:
                u_uncond, _ = self.denoising_net(xt, z_uncond, t_now, h)
                low, high = cfg_interval
                interval_mask = (t_now < high) & ((low == 0) | (t_now > low))
                scale = torch.where(interval_mask, cfg_scale, 1.0).view(-1, 1)
                u = u_uncond + scale * (u_cond - u_uncond)
            else:
                u = u_cond

            xt = xt - h.view(-1, 1) * u

        return xt

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for ema_params, decay in zip(self.ema_params_list, self.ema_decays):
            for targ, src in zip(ema_params, source_params):
                targ.detach().mul_(decay).add_(src, alpha=1 - decay)
