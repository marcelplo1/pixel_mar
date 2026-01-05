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
def sample(args, mae, denoiser, labels, device):
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    bsz = args.gen_batch_size
    seq_len = (args.img_size // args.patch_size) ** 2
    embed_dim = (args.patch_size ** 2) * args.channels

    tokens = torch.zeros(bsz, seq_len, embed_dim, device=device)
    mask = torch.ones(bsz, seq_len, device=device)
    orders = sample_order(bsz, seq_len, device)

    ar_steps = args.num_ar_steps
    for i in range(ar_steps):
        z = mae(tokens, mask, labels)

        mask_ratio = np.cos(math.pi / 2. * (i + 1) / ar_steps)
        mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(torch.Tensor([1]).to(device),
                            torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
        mask_next = mask_by_order(mask_len[0], orders, bsz, seq_len, device)
        if i >= ar_steps - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next[:bsz].bool())
        mask = mask_next

        z = z[mask_to_pred.nonzero(as_tuple=True)]

        new_bsz = z.shape[0]
        xt = args.noise_scale * torch.randn(new_bsz, embed_dim, device=device)

        sampled_x = denoiser.generate(xt, z)

        tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_x
        if args.is_debug and local_rank == 0:
            folder = os.path.join(args.output_dir, "ar_generation_steps")
            save_img_as_fig(unpatchify(tokens.reshape(tokens.shape[0], tokens.shape[1], -1), args.patch_size, seq_len, args.channels), filename="sampling_step_{}.png".format(i), path=folder)

    img = unpatchify(tokens, args.patch_size, seq_len, channels=args.channels)
    
    return img


