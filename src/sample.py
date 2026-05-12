import os
import torch
import numpy as np
import math

from utils.utils import sample_order, save_img_as_fig, save_pca_viz, unpatchify

def mask_by_order(mask_len, order, bsz, seq_len, device ):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=device))
    return masking

def compute_mask_schedule(schedule_type, num_ar_steps, seq_len):
    """Compute the list of cumulative visible token counts for each AR step.
    """
    if schedule_type == 'exp':
        schedule = []
        prev = 0
        for i in range(num_ar_steps):
            t = 1.0 - (i + 1) / num_ar_steps
            mask_ratio = 1.0 - np.exp(-5.0 * t)
            mask_len = int(np.floor(seq_len * mask_ratio))
            num_visible = seq_len - mask_len
            if num_visible <= prev and i < num_ar_steps - 1:
                num_visible = prev + 1
            schedule.append(num_visible)
            prev = num_visible
        return schedule
    else:  # cosine (default)
        schedule = []
        prev = 0
        for i in range(num_ar_steps):
            mask_ratio = np.cos(math.pi / 2.0 * (i + 1) / num_ar_steps)
            mask_len = int(np.floor(seq_len * mask_ratio))
            num_visible = seq_len - mask_len
            if num_visible <= prev and i < num_ar_steps - 1:
                num_visible = prev + 1
            schedule.append(num_visible)
            prev = num_visible
        return schedule

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
    use_latent = rae_tokenizer is not None

    seq_len = (img_size// patch_size) ** 2

    if use_latent:
        embed_dim = rae_tokenizer.latent_dim
    else:
        embed_dim = (patch_size ** 2) * channels

    cur_tokens = torch.zeros(bsz, seq_len, embed_dim, device=device)
    orders = sample_order(bsz, seq_len, device)

    num_generation_passes = sampler_params.get('num_generation_passes', 1)

    mask_schedule_type = sampler_params.get('mask_schedule', 'cosine')
    schedule = compute_mask_schedule(mask_schedule_type, num_ar_steps, seq_len)

    xt_global = noise_scale * torch.randn(bsz, seq_len, embed_dim, device=device)
    num_visible = 0
    for i, next_num_visible in enumerate(schedule):
        recon_weight = getattr(args, 'recon_weight', 0.0)
        if cfg_scale != 1.0:
            z_cond, mask, x_recon = ar_model(cur_tokens, orders, labels, num_visible)
            z_uncond, _, _ = ar_model(cur_tokens, orders, labels, num_visible, force_unconditional=True)
        else:
            z_cond, mask, x_recon = ar_model(cur_tokens, orders, labels, num_visible)
            z_uncond = None

        ids_to_predict = orders[:, num_visible:next_num_visible]
        mask_to_pred = torch.zeros(bsz, seq_len, device=device, dtype=torch.bool)
        mask_to_pred.scatter_(1, ids_to_predict.long(), True)
        num_visible = next_num_visible

        pred_indices = mask_to_pred.nonzero(as_tuple=True)
        xt_mask = xt_global[pred_indices]
        if denoiser.denoising_net.denoiser_type == 'cross_attn':
            # Each token needs the full z sequence of its image
            z_c = z_cond[pred_indices[0]]   # (num_tokens, N, z_dim)
            z_u = z_uncond[pred_indices[0]] if z_uncond is not None else None
            token_positions = pred_indices[1]
        else:
            z_c = z_cond[pred_indices]      # (num_tokens, z_dim)
            z_u = z_uncond[pred_indices] if z_uncond is not None else None
            token_positions = None
        y = labels.repeat(z_c.shape[0] // bsz)

        # Linear CFG schedule (following MAR): ramp from 1.0 to cfg_scale as more tokens become visible
        cfg_scale_iter = 1.0 + (cfg_scale - 1.0) * num_visible / seq_len
        #cfg_scale_iter = cfg_scale
        sampled_x = denoiser.generate(xt_mask, z_c, y, z_uncond=z_u, cfg_scale=cfg_scale_iter, cfg_interval=cfg_interval, token_positions=token_positions)
        cur_tokens[pred_indices] = sampled_x

        if args.use_logging and local_rank == 0:
            base = os.path.join(args.output_dir, "images")
            folder_steps  = os.path.join(base, "ar_generation_steps")
            folder_z_pca  = os.path.join(base, "ar_condition_z_pca")
            folder_recon  = os.path.join(base, "ar_condition_z_decoded")
            os.makedirs(folder_steps, exist_ok=True)

            if use_latent:
                save_img_as_fig(rae_tokenizer.decode(cur_tokens),
                                file_path=os.path.join(folder_steps, "step_{}.png".format(i)), size=img_size)
                os.makedirs(folder_z_pca, exist_ok=True)
                save_pca_viz(z_cond, os.path.join(folder_z_pca, "step_{}.png".format(i)), img_size, idx=0)
                if recon_weight > 0:
                    os.makedirs(folder_recon, exist_ok=True)
                    save_img_as_fig(rae_tokenizer.decode(x_recon).clamp(-1, 1),
                                    file_path=os.path.join(folder_recon, "step_{}.png".format(i)), size=img_size)
            else:
                save_img_as_fig(unpatchify(cur_tokens, patch_size, channels=channels),
                                file_path=os.path.join(folder_steps, "step_{}.png".format(i)), size=img_size)
                
                if recon_weight > 0:
                    os.makedirs(folder_recon, exist_ok=True)
                    save_img_as_fig(unpatchify(x_recon, patch_size, channels=channels).clamp(-1, 1),
                                    file_path=os.path.join(folder_recon, "step_{}.png".format(i)), size=img_size)

    if use_latent:
        img = rae_tokenizer.decode(cur_tokens)
    else:
        img = unpatchify(cur_tokens, patch_size, channels=channels)

    return img