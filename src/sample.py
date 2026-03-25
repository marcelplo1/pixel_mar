import os
import torch
import numpy as np
import math

from utils.utils import sample_order, save_img_as_fig, unpatchify

def mask_by_order(mask_len, order, bsz, seq_len, device ):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=device))
    return masking

@torch.no_grad()
def sample(args, ar_model, denoiser, labels, device, model_params, sampler_params, rae_tokenizer=None):
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    bsz = args.gen_batch_size
    patch_size = args.patch_size
    img_size = args.img_size
    channels = args.channels
    noise_scale = model_params.get('noise_scale', 1.0)
    num_ar_steps = sampler_params.get('num_ar_steps', 64)

    cfg_scale = getattr(args, 'cfg_scale', 1.0)
    cfg_interval = (getattr(args, 'cfg_interval_min', 0.0), getattr(args, 'cfg_interval_max', 1.0))
    use_latent = getattr(args, 'use_latent_space', False) and rae_tokenizer is not None

    seq_len = (img_size// patch_size) ** 2

    if use_latent:
        embed_dim = rae_tokenizer.latent_dim
    else:
        embed_dim = (patch_size ** 2) * channels

    cur_tokens = torch.zeros(bsz, seq_len, embed_dim, device=device)
    orders = sample_order(bsz, seq_len, device)

    num_generation_passes = sampler_params.get('num_generation_passes', 1)

    # --- Pass 0: autoregressive generation ---
    xt_global = noise_scale * torch.randn(bsz, seq_len, embed_dim, device=device)
    num_visible = 0
    for i in range(num_ar_steps):
        if use_latent:
            ar_model_input = cur_tokens
        else:
            ar_model_input = unpatchify(cur_tokens, patch_size, channels=channels)

        if cfg_scale != 1.0:
            z_cond, mask, _ = ar_model(ar_model_input, orders, labels, num_visible)
            z_uncond, _, _ = ar_model(ar_model_input, orders, labels, num_visible, force_unconditional=True)
        else:
            z_cond, mask, _ = ar_model(ar_model_input, orders, labels, num_visible)
            z_uncond = None

        mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_ar_steps)
        mask_len = int(np.floor(seq_len * mask_ratio))
        next_num_visible = seq_len - mask_len

        if next_num_visible <= num_visible and i < num_ar_steps - 1:
            next_num_visible = num_visible + 1

        ids_to_predict = orders[:, num_visible:next_num_visible]
        mask_to_pred = torch.zeros(bsz, seq_len, device=device, dtype=torch.bool)
        mask_to_pred.scatter_(1, ids_to_predict.long(), True)
        num_visible = next_num_visible

        pred_indices = mask_to_pred.nonzero(as_tuple=True)
        z_c = z_cond[pred_indices]
        z_u = z_uncond[pred_indices] if z_uncond is not None else None
        xt_mask = xt_global[pred_indices]
        y = labels.repeat(z_c.shape[0] // bsz)

        # Linear CFG schedule (following MAR): ramp from 1.0 to cfg_scale as more tokens become visible
        cfg_scale_iter = 1.0 + (cfg_scale - 1.0) * num_visible / seq_len
        sampled_x = denoiser.generate(xt_mask, z_c, y, z_uncond=z_u, cfg_scale=cfg_scale_iter, cfg_interval=cfg_interval)
        cur_tokens[pred_indices] = sampled_x

        if args.use_logging and local_rank == 0:
            folder = os.path.join(args.output_dir, "ar_generation_steps")
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, "pass0_step_{}.png".format(i))

            if use_latent:
                save_img_as_fig(rae_tokenizer.decode(cur_tokens), file_path=file_path, size=img_size)
            else:
                save_img_as_fig(unpatchify(cur_tokens, patch_size, channels=channels), file_path=file_path, size=img_size)

    if use_latent:
        img = rae_tokenizer.decode(cur_tokens)
    else:
        img = unpatchify(cur_tokens, patch_size, channels=channels)

    return img