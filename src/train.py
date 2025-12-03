import torch
import torch.optim as optim
from src.utils import patchify, unpatchify, sample_order, random_masking, compute_loss


def random_masking(x, orders, min_mask_rate=0.7):
    bsz, seq_len, embed_dim = x.shape
    #min_mask_rate = stats.truncnorm((MIN_MASK_RATE - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
    num_masked_tokens = int(np.ceil(seq_len * min_mask_rate))
    mask = torch.zeros(bsz, seq_len, device=x.device)
    mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
    return (1-mask)

def train_one_epoch(dataloader, mae, denoising_mlp, optimizer, patch_size, min_mask_rate, device, debug=False):
    t_eps = 1e-2

    for samples, labels in dataloader:
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x = patchify(samples, patch_size)
        orders = sample_order(x.shape[0], x.shape[1], device    )
        mask = random_masking(x, orders, min_mask_rate)

        z = mae(x, mask)

        eps = torch.randn_like(x)
        t = torch.rand(x.size(0), x.size(1), 1, device=device).clamp(t_eps, 1 - t_eps)
        xt = t * x + (1 - t) * eps

        t = t.reshape(t.shape[0]*t.shape[1], -1)
        xt = xt.reshape(xt.shape[0]*xt.shape[1], -1)
        z = z.reshape(z.shape[0]*z.shape[1], -1)
        gt = x.reshape(x.shape[0]*x.shape[1], -1).clone().detach()
        eps = eps.reshape(eps.shape[0]*eps.shape[1], -1)

        patch_prediction = denoising_mlp(torch.cat([xt, z], dim=1), t)

        loss = compute_loss(gt, eps, xt, t, patch_prediction, pred_type="eps", loss_type="eps", debug=debug)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()