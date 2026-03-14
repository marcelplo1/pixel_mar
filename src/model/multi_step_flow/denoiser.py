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
        use_latent_space=False,
        time_shift=None,
        atol=1e-5,
        rtol=1e-5
    ):
        super().__init__()

        self.denoising_net = denoising_model

        self.img_size = denoising_model.img_size
        self.channels = denoising_model.in_channels
        self.patch_size = denoising_model.patch_size

        self.P_mean = sample_t_mean
        self.P_std = sample_t_std
        self.t_eps = t_eps
        self.t_eps_sample = t_eps_sample
        self.noise_scale = noise_scale
        self.method = sampling_method
        self.steps = num_timesteps

        self.use_logging = use_logging
        self.output_dir = output_dir
        self.use_latent_space = use_latent_space

        self.pred_type = pred_type
        self.ema_decays = ema_decay if isinstance(ema_decay, list) else [ema_decay]
        self.diffusion_batch_mul = diffusion_batch_mul

        self.time_shift = time_shift
        self.atol = atol
        self.rtol = rtol
        self.ema_params_list = None
        self.log_counter = 0
        self.log_batch_pred = 100

    def _shift_t(self, t):
        """Apply time shift."""
        s = 1.0 - t
        s = self.time_shift * s / (1.0 + (self.time_shift - 1.0) * s)
        return 1.0 - s

    def sample_t(self, n: int, device=None):  # lognormal distribution
        """Log normal time sampling"""
        t = torch.randn(n, device=device) * self.P_std + self.P_mean
        t = torch.sigmoid(t)
        if self.time_shift is not None:
            return self._shift_t(t)
        else:
            return t

    def forward(self, x, z, mask, labels):
        B, N, D = x.shape
        x = x.view(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        labels = labels.repeat(self.diffusion_batch_mul*N)
        mask = mask.reshape(B*N).repeat(self.diffusion_batch_mul)

        if self.training:
            t_eps = self.t_eps
        else:
            t_eps = self.t_eps_sample

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        xt = t * x + (1 - t) * e
        v = (x - xt) / (1 - t).clamp_min(t_eps)

        pred = self.denoising_net(xt, z, t.flatten())

        if self.pred_type == 'x':
            v_pred = (pred - xt) / (1 - t).clamp_min(t_eps)
            x_pred = pred
        elif self.pred_type == 'v':
            v_pred = pred
            x_pred = xt + (1-t).clamp_min(t_eps) * pred
        elif self.pred_type == 'e':
            v_pred = (xt-pred)/(t).clamp_min(t_eps)
            x_pred = (xt-(1-t) * pred) / t.clamp_min(t_eps)

        loss = (v - v_pred) ** 2

        if mask is not None:
            loss = loss.view(-1, N, D)
            mask = mask.view(-1, N, 1)
            loss = ((loss * mask).sum(dim=(1, 2)) / (mask.sum(dim=(1, 2)) * D)).mean()

        self.log_counter += 1
        if self.use_logging and dist.get_rank() == 0 and self.log_counter % self.log_batch_pred == 0:
            self.log_counter = 0
            time_step = round(t[0].item(), 1)

            # Skip pixel-space visualization when operating in latent space
            if self.use_latent_space == False:
                x_vis = x.view(self.diffusion_batch_mul, B, N, D)[0]
                x_pred_vis = x_pred.view(self.diffusion_batch_mul, B, N, D)[0].clamp(-1, 1)
                v_pred_vis = v_pred.view(self.diffusion_batch_mul, B, N, D)[0]
                mask_vis = mask.view(self.diffusion_batch_mul, B, N, 1)[0]

                x_pred_vis[(mask_vis==0).expand_as(x_pred_vis)] = -1.0

                folder = os.path.join(self.output_dir, "last_training_predictions")
                os.makedirs(folder, exist_ok=True)

                x_path =  os.path.join(folder, "ground_truth_t={}.png".format(time_step))
                save_img_as_fig(unpatchify(x_vis, self.patch_size, self.channels),
                                file_path=x_path, size=self.img_size)

                x_pred_path = os.path.join(folder, "prediction_t={}.png".format(time_step))
                save_img_as_fig(unpatchify(x_pred_vis, self.patch_size, self.channels),
                                file_path=x_pred_path.format(time_step), size=self.img_size)

        return loss

    @torch.no_grad()
    def generate(self, xt, z, labels, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        device = z.device
        bsz = xt.size(0)
        timesteps = torch.linspace(self.t_eps_sample, 1.0 - self.t_eps_sample, self.steps+1, device=device)
        if self.time_shift is not None:
            timesteps = self._shift_t(timesteps)

        # TODO: Remove the test method and itegrate it with fixed ODEs
        if self.method == 'dopri5':
            return self._generate_adaptive(xt, z, timesteps, z_uncond, cfg_scale, cfg_interval)

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

    @torch.no_grad()
    def _generate_adaptive(self, xt, z, t_span, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        from torchdiffeq import odeint

        def drift_fn(t, x):
            t_batch = t.expand(x.size(0)).unsqueeze(-1)
            return self._forward_sample(x, z, t_batch, z_uncond, cfg_scale, cfg_interval)

        trajectory = odeint(
            drift_fn,
            xt,
            t_span,
            method=self.method,
            atol=self.atol,
            rtol=self.rtol,
        )

        return trajectory[-1]

    def _pred_to_velocity(self, pred, xt, t):
        if self.pred_type == 'v':
            return pred
        elif self.pred_type == 'x':
            return (pred - xt) / (1.0 - t).clamp_min(self.t_eps)
        elif self.pred_type == 'e':
            return (xt - pred) / (t).clamp_min(self.t_eps)

    @torch.no_grad()
    def _forward_sample(self, xt, z, t, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        # Conditional prediction
        pred_cond = self.denoising_net(xt, z, t.view(-1))
        v_cond = self._pred_to_velocity(pred_cond, xt, t)

        if cfg_scale == 1.0 or z_uncond is None:
            return v_cond

        # Unconditional prediction
        pred_uncond = self.denoising_net(xt, z_uncond, t.view(-1))
        v_uncond = self._pred_to_velocity(pred_uncond, xt, t)

        # CFG interval masking
        low, high = cfg_interval
        interval_mask = (t < high) & ((low == 0) | (t > low))
        scale = torch.where(interval_mask, cfg_scale, 1.0)

        return v_uncond + scale * (v_cond - v_uncond)

    @torch.no_grad()
    def _euler_step(self, xt, z, t, t_next, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        v_pred = self._forward_sample(xt, z, t, z_uncond, cfg_scale, cfg_interval)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def _heun_step(self, xt, z, t, t_next, z_uncond=None, cfg_scale=1.0, cfg_interval=(0.0, 1.0)):
        v_pred_t = self._forward_sample(xt, z, t, z_uncond, cfg_scale, cfg_interval)

        xt_next_euler = xt + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(xt_next_euler, z, t_next, z_uncond, cfg_scale, cfg_interval)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def update_ema(self):
        source_params = list(self.parameters())
        for ema_params, decay in zip(self.ema_params_list, self.ema_decays):
            for targ, src in zip(ema_params, source_params):
                targ.detach().mul_(decay).add_(src, alpha=1 - decay)