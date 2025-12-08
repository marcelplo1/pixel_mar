import torch
import numpy as np
import math

from utils import sample_order, save_token_as_fig, unpatchify

def mask_by_order(mask_len, order, bsz, seq_len, device ):
    masking = torch.zeros(bsz, seq_len).cuda()
    masking = torch.scatter(masking, dim=-1, index=order[:, :mask_len.long()], src=torch.ones(bsz, seq_len, device=device))
    return masking

def get_v_prediction(pred, xt, t, pred_type="x"):
    if pred_type == "x":
        v_pred = (pred - xt) / (1-t).clamp(min=0.05)
    elif pred_type == "eps":
        v_pred = (xt - pred) / t.clamp(min=0.05)
    elif pred_type == "v":
        v_pred = pred
    return v_pred

@torch.no_grad()
def sample(denoising_mlp, mae, bsz, seq_len, embed_dim, patch_size, device, pred_type = "x", num_iter=64, num_timesteps=32, debug=False):
    xt = torch.randn((bsz, seq_len, embed_dim), device=device)
    tokens = torch.zeros_like(xt)
    mask = torch.ones(bsz, seq_len, device=device)
    orders = sample_order(bsz, seq_len, device)

    t_eps = 0.0
    time_steps = torch.linspace(t_eps, 1 - t_eps, num_timesteps + 1, device=device)

    for i in range(num_iter):
        z = mae(xt, mask)

        mask_ratio = np.cos(math.pi / 2. * (i + 1) / num_iter)
        mask_len = torch.Tensor([np.floor(seq_len * mask_ratio)]).to(device)
        mask_len = torch.maximum(torch.Tensor([1]).to(device),
                            torch.minimum(torch.sum(mask, dim=-1, keepdims=True) - 1, mask_len))
        mask_next = mask_by_order(mask_len[0], orders, bsz, seq_len, device)
        if i >= num_iter - 1:
            mask_to_pred = mask[:bsz].bool()
        else:
            mask_to_pred = torch.logical_xor(mask[:bsz].bool(), mask_next[:bsz].bool())
        mask = mask_next

        x_i = xt[mask_to_pred.nonzero(as_tuple=True)]
        z_i = z[mask_to_pred.nonzero(as_tuple=True)]

        for t_idx in range(num_timesteps):
            t = time_steps[t_idx]
            t_next = time_steps[t_idx + 1]

            t_batch = torch.full((x_i.shape[0], 1), t, device=device)
            t_next_batch = torch.full((x_i.shape[0], 1), t_next, device=device)

            pred = denoising_mlp(torch.cat([x_i, z_i], dim=-1), t_batch)

            v_pred = get_v_prediction(pred, x_i, t_batch, pred_type=pred_type)
            x_i = x_i + (t_next_batch - t_batch) * v_pred

        xt[mask_to_pred.nonzero(as_tuple=True)] = x_i
        tokens[mask_to_pred.nonzero(as_tuple=True)] = x_i
    
    if debug:
        save_token_as_fig(tokens, patch_size=patch_size, filename=f"last_sample.png", path="./output")

    img = unpatchify(tokens, patch_size, seq_len, channels=1)

    return img


