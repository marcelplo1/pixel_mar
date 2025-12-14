import torch
import numpy as np
import math

from utils import patchify, sample_order, unpatchify

def mask_by_order(mask_len, order, bsz, seq_len, device ):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=device))
    return masking

@torch.no_grad()
def sample(args, mae, denoiser, device):
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    bsz = args.gen_batch_size
    seq_len = (args.img_size // args.patch_size) ** 2

    xt = args.noise_scale * torch.randn(bsz, args.channels, args.img_size, args.img_size, device=device)
    xt = patchify(xt, args.patch_size)
    tokens = torch.zeros_like(xt)
    mask = torch.ones(bsz, seq_len, device=device)
    orders = sample_order(bsz, seq_len, device)
 
    assert args.num_images % args.class_num == 0, "Number of images per class must be the same"
    class_label_gen_world = np.arange(0, args.class_num).repeat(args.num_images // args.class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    ar_steps = args.num_ar_steps

    for i in range(ar_steps):
        z = mae(xt, mask)

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

        start_idx = world_size * bsz * i + local_rank * bsz
        end_idx = start_idx + bsz
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()

        sampled_x = denoiser.generate(xt, z, labels_gen)

        xt[mask_to_pred.nonzero(as_tuple=True)] = sampled_x[mask_to_pred.nonzero(as_tuple=True)]
        tokens[mask_to_pred.nonzero(as_tuple=True)] = sampled_x[mask_to_pred.nonzero(as_tuple=True)]

    img = unpatchify(tokens, args.patch_size, seq_len, channels=args.channels)
    
    return img


