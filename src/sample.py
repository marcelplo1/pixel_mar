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
def sample(args, mae, denoiser, labels, device, model_params):
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    bsz = args.gen_batch_size
    patch_size = mae.patch_size
    img_size = mae.img_size
    channels = mae.channels
    noise_scale = model_params.get('noise_scale', 1.0)
    num_ar_steps = model_params.get('num_ar_steps', 64)

    seq_len = (img_size// patch_size) ** 2
    embed_dim = (patch_size ** 2) * channels

    tokens = torch.zeros(bsz, seq_len, embed_dim, device=device)
    mask = torch.ones(bsz, seq_len, device=device)
    orders = sample_order(bsz, seq_len, device)

    for i in range(num_ar_steps):
        cur_tokens = tokens.clone()
        z = mae(tokens, mask, labels)

        mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_ar_steps)
        mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(torch.Tensor([1]).to(device),
                            torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
        mask_next = mask_by_order(mask_len[0], orders, bsz, seq_len, device)
        if i >= num_ar_steps - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next[:bsz].bool())
        mask = mask_next

        z = z[mask_to_pred.nonzero(as_tuple=True)]
        xt_mask = noise_scale * torch.randn(z.shape[0], embed_dim).cuda()
        y = labels.repeat(z.shape[0] // bsz)

        sampled_x = denoiser.generate(xt_mask, z, y)

        cur_tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_x
        tokens = cur_tokens.clone()

        if args.use_logging and local_rank == 0:
            folder = os.path.join(args.output_dir, "ar_generation_steps")
            os.makedirs(folder, exist_ok=True)
            file_path = os.path.join(folder, "sampling_step_{}.png".format(i))
            save_img_as_fig(unpatchify(tokens.reshape(tokens.shape[0], tokens.shape[1], -1), patch_size, channels), file_path=file_path, size=img_size)

    img = unpatchify(tokens, patch_size, channels=channels)
    
    return img


