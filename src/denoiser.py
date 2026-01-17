import os
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist

from utils.utils import save_img_as_fig, unpatchify

class Denoiser(nn.Module):
    def __init__(
        self,
        args,
        denoisingMLP
    ):
        super().__init__()

        self.denoising_net = denoisingMLP

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
        self.is_debug = args.is_debug
        self.output_dir = args.output_dir

        self.pred_type = args.pred_type
        self.ema_decay = args.ema_decay

        self.diffusion_batch_mul = 4
        self.log_counter = 0

    def sample_t(self, n: int, device=None):  # lognormal distribution
        #t = torch.randn(n, device=device) * self.P_std + self.P_mean
        t = torch.randn(n, device=device)
        return torch.sigmoid(t)

    def forward(self, x, z, mask):

        B, N, D = x.shape
        x = x.view(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        mask = mask.reshape(B*N).repeat(self.diffusion_batch_mul)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        xt = t * x + (1 - t) * e
        v = (x - xt) / (1 - t).clamp_min(self.t_eps)

        pred = self.denoising_net(xt, z, t.flatten())

        if self.pred_type == 'x':
            v_pred = (pred - xt) / (1 - t).clamp_min(self.t_eps)
            x_pred = pred
        elif self.pred_type == 'v':
            v_pred = pred
            x_pred = xt + (1-t).clamp_min(self.t_eps) * pred
        elif self.pred_type == 'e':
            v_pred = (xt-pred)/(t).clamp_min(self.t_eps)
            x_pred = (xt-(1-t) * pred) / t.clamp_min(self.t_eps)

        # l2 loss
        loss = (v - v_pred) ** 2

        if mask is not None:
            mask = mask.view(-1, 1)
            loss = (loss * mask).sum() / (mask.sum() * D)

        self.log_counter += 1
        if self.is_debug and dist.get_rank() == 0 and self.log_counter % 100 == 0:
            self.log_counter = 0
            time_step = round(t[0].item(), 1)

            x_vis = x.view(self.diffusion_batch_mul, B, N, D)[0]
            x_pred_vis = x_pred.view(self.diffusion_batch_mul, B, N, D)[0].clamp(-1, 1)
            v_pred_vis = v_pred.view(self.diffusion_batch_mul, B, N, D)[0]
            mask_vis = mask.view(self.diffusion_batch_mul, B, N, 1)[0]
            
            error_vis = torch.abs(x_vis - x_pred_vis)

            #x_pred_vis *= mask_vis

            save_img_as_fig(unpatchify(x_vis, self.denoising_net.patch_size, N, self.channels), 
                            filename="ground_truth_t={}.png".format(time_step), path=self.output_dir, size=self.img_size)
            
            save_img_as_fig(unpatchify(x_pred_vis, self.denoising_net.patch_size, N, self.channels), 
                            filename="prediction_t={}.png".format(time_step), path=self.output_dir, size=self.img_size)

            save_img_as_fig(unpatchify(v_pred_vis, self.denoising_net.patch_size, N, self.channels), 
                            filename="velocity_field_t={}.png".format(time_step), path=self.output_dir, size=self.img_size)

            save_img_as_fig(unpatchify(error_vis, self.denoising_net.patch_size, N, self.channels), 
                            filename="error_map_t={}.png".format(time_step), path=self.output_dir, size=self.img_size)

        return loss

    @torch.no_grad()
    def generate(self, xt, z):
        device = z.device
        bsz = xt.size(0)
        timesteps = torch.linspace(self.t_eps, 1.0 - self.t_eps, self.steps+1, device=device)
        timesteps = timesteps.view(-1, 1, 1).expand(-1, bsz, 1)  

        if self.method == "euler":
            stepper = self._euler_step
        elif self.method == "heun":
            stepper = self._heun_step
        else:
            raise NotImplementedError

        # ode
        for i in range(self.steps):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            xt = stepper(xt, z, t, t_next)
            # TODO Log each noisy step
        return xt

    @torch.no_grad()
    def _forward_sample(self, xt, z, t):
        pred = self.denoising_net(xt, z, t.view(-1))
        if self.pred_type == 'v':
            v_pred = pred
        elif self.pred_type == 'x':
            pred = torch.clamp(pred, -1.0, 1.0) #TODO
            v_pred = (pred - xt) / (1.0 - t).clamp_min(self.t_eps)
        elif self.pred_type == 'e':
            v_pred = (xt - pred) / (t).clamp_min(self.t_eps)
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
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)