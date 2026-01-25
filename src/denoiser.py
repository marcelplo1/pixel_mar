import os
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
        diffusion_batch_multi=4,
        num_timesteps = 100,
        sample_t_mean = 0.0,
        sample_t_std = 1.0,
        t_eps = 1e-2,
        noise_scale = 1.0,
        ema_decay = 0.9999,
        use_logging=False
    ):
        super().__init__()

        self.denoising_net = denoising_model

        self.img_size = denoising_model.img_size
        self.channels = denoising_model.in_channels
        self.patch_size = denoising_model.patch_size

        self.P_mean = sample_t_mean
        self.P_std = sample_t_std
        self.t_eps = t_eps
        self.noise_scale = noise_scale
        self.method = sampling_method
        self.steps = num_timesteps

        self.use_logging = use_logging
        self.output_dir = output_dir

        self.pred_type = pred_type
        self.ema_decay = ema_decay
        self.diffusion_batch_mul = diffusion_batch_multi

        self.ema_params = None
        self.log_counter = 0
        self.log_batch_pred = 100

    def sample_t(self, n: int, device=None):  # lognormal distribution
        t = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(t)

    def forward(self, x, z, mask, labels):
        B, N, D = x.shape
        x = x.view(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        z = z.reshape(B*N, -1).repeat(self.diffusion_batch_mul, 1)
        labels = labels.repeat(self.diffusion_batch_mul*N)
        mask = mask.reshape(B*N).repeat(self.diffusion_batch_mul)

        t = self.sample_t(x.size(0), device=x.device).view(-1, *([1] * (x.ndim - 1)))
        e = torch.randn_like(x) * self.noise_scale

        xt = t * x + (1 - t) * e
        v = (x - xt) / (1 - t).clamp_min(self.t_eps)

        pred = self.denoising_net(xt, z, t.flatten(), labels)

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
        if self.use_logging and dist.get_rank() == 0 and self.log_counter % self.log_batch_pred == 0:
            self.log_counter = 0
            time_step = round(t[0].item(), 1)

            x_vis = x.view(self.diffusion_batch_mul, B, N, D)[0]
            x_pred_vis = x_pred.view(self.diffusion_batch_mul, B, N, D)[0].clamp(-1, 1)
            v_pred_vis = v_pred.view(self.diffusion_batch_mul, B, N, D)[0]
            mask_vis = mask.view(self.diffusion_batch_mul, B, N, 1)[0]
            
            error_vis = torch.abs(x_vis - x_pred_vis)

            x_pred_vis[(mask_vis==0).expand_as(x_pred_vis)] = -1.0


            folder = os.path.join(self.output_dir, "last_training_predictions")
            os.makedirs(folder, exist_ok=True)

            x_path =  os.path.join(folder, "ground_truth_t={}.png".format(time_step))
            save_img_as_fig(unpatchify(x_vis, self.patch_size, self.channels), 
                            file_path=x_path, size=self.img_size)
            
            x_pred_path = os.path.join(folder, "prediction_t={}.png".format(time_step))
            save_img_as_fig(unpatchify(x_pred_vis, self.patch_size, self.channels), 
                            file_path=x_pred_path.format(time_step), size=self.img_size)

            # save_img_as_fig(unpatchify(v_pred_vis, self.denoising_net.patch_size, N, self.channels), 
            #                 filename="velocity_field_t={}.png".format(time_step), path=self.output_dir, size=self.img_size)

            # save_img_as_fig(unpatchify(error_vis, self.denoising_net.patch_size, N, self.channels), 
            #                 filename="error_map_t={}.png".format(time_step), path=self.output_dir, size=self.img_size)

        return loss

    @torch.no_grad()
    def generate(self, xt, z, labels):
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
        for i in range(self.steps-1):
            t = timesteps[i]
            t_next = timesteps[i + 1]
            xt = stepper(xt, z, t, t_next, labels)
            # TODO Log each noisy step
        xt = self._euler_step(xt, z, timesteps[-2], timesteps[-1], labels)
        return xt

    @torch.no_grad()
    def _forward_sample(self, xt, z, t, labels):
        pred = self.denoising_net(xt, z, t.view(-1), labels)
        if self.pred_type == 'v':
            v_pred = pred
        elif self.pred_type == 'x':
            v_pred = (pred - xt) / (1.0 - t).clamp_min(self.t_eps)
        elif self.pred_type == 'e':
            v_pred = (xt - pred) / (t).clamp_min(self.t_eps)
        return v_pred

    @torch.no_grad()
    def _euler_step(self, xt, z, t, t_next, labels):
        v_pred = self._forward_sample(xt, z, t, labels)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def _heun_step(self, xt, z, t, t_next, labels):
        v_pred_t = self._forward_sample(xt, z, t)

        xt_next_euler = xt + (t_next - t) * v_pred_t
        v_pred_t_next = self._forward_sample(xt_next_euler, z, t_next, labels)

        v_pred = 0.5 * (v_pred_t + v_pred_t_next)
        xt_next = xt + (t_next - t) * v_pred
        return xt_next

    @torch.no_grad()
    def update_ema(self):
        ema_decay = self.ema_decay
        source_params = list(self.parameters())
        for targ, src in zip(self.ema_params, source_params):
            targ.detach().mul_(ema_decay).add_(src, alpha=1 - ema_decay)