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
def sample(args, mae, denoiser, labels, device, model_params, sampler_params):
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    bsz = args.gen_batch_size
    patch_size = args.patch_size
    img_size = args.img_size
    channels = args.channels
    noise_scale = model_params.get('noise_scale', 1.0)
    num_ar_steps = sampler_params.get('num_ar_steps', 64)

    cfg_scale = getattr(args, 'cfg_scale', 1.0)
    cfg_schedule = getattr(args, 'cfg_schedule', 'linear')

    seq_len = (img_size// patch_size) ** 2
    embed_dim = (patch_size ** 2) * channels

    cur_tokens = torch.zeros(bsz, seq_len, embed_dim, device=device)
    orders = sample_order(bsz, seq_len, device)
    
    num_generation_passes = sampler_params.get('num_generation_passes', 2)
    xt_global = noise_scale * torch.randn(bsz, seq_len, embed_dim, device=device)
    for pass_idx in range(num_generation_passes):
        num_visible = 0

        for i in range(num_ar_steps):
            tokens = unpatchify(cur_tokens, patch_size, channels=channels)

            # In refinement passes, MAE sees the full image for better context
            mae_num_visible = seq_len if pass_idx > 0 else num_visible

            if cfg_scale != 1.0:
                # CFG: run MAE conditionally and unconditionally, combine z
                z_cond, mask = mae(tokens, orders, labels, mae_num_visible)
                z_uncond, _ = mae(tokens, orders, labels, mae_num_visible, force_unconditional=True)

                if cfg_schedule == "linear":
                    cfg_iter = 1.0 + (cfg_scale - 1.0) * num_visible / seq_len
                else:  # constant
                    cfg_iter = cfg_scale

                z = z_uncond + cfg_iter * (z_cond - z_uncond)
            else:
                z, mask = mae(tokens, orders, labels, mae_num_visible)

            mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_ar_steps)
            mask_len = int(np.floor(seq_len * mask_ratio))

            next_num_visible = seq_len - mask_len

            if next_num_visible <= num_visible and i < num_ar_steps - 1:
                next_num_visible = num_visible + 1

            ids_to_predict = orders[:, num_visible:next_num_visible]
            mask_to_pred = torch.zeros(bsz, seq_len, device=device, dtype=torch.bool)
            mask_to_pred.scatter_(1, ids_to_predict.long(), True)

            num_visible = next_num_visible

            z = z[mask_to_pred.nonzero(as_tuple=True)]
            xt_mask = xt_global[mask_to_pred.nonzero(as_tuple=True)]
            y = labels.repeat(z.shape[0] // bsz)

            sampled_x = denoiser.generate(xt_mask, z, y)

            cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_x

            if args.use_logging and local_rank == 0:
                folder = os.path.join(args.output_dir, "ar_generation_steps")
                os.makedirs(folder, exist_ok=True)
                file_path = os.path.join(folder, "pass{}_step_{}.png".format(pass_idx, i))
                save_img_as_fig(unpatchify(cur_tokens, patch_size, channels=channels), file_path=file_path, size=img_size)

    img = unpatchify(cur_tokens, patch_size, channels=channels)

    return img


