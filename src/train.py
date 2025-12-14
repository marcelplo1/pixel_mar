import os
import torch
import torch.optim as optim
import numpy as np
from sample import sample
from utils import patchify, sample_order, save_img_as_fig, save_multiple_imgs_as_fig, save_plot, unpatchify

def random_masking(x, orders, min_mask_rate=0.7):
    bsz, seq_len, embed_dim = x.shape
    #min_mask_rate = stats.truncnorm((MIN_MASK_RATE - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
    num_masked_tokens = int(np.ceil(seq_len * min_mask_rate))
    mask = torch.zeros(bsz, seq_len, device=x.device)
    mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
    return (1-mask)

def train_one_epoch(args, dataloader, mae, denoiser, optimizer, device, global_rank=0):
    mae.train()
    denoiser.train()    
    
    losses = []
    for samples, labels in dataloader:
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x = patchify(samples, args.patch_size)
        orders = sample_order(x.shape[0], x.shape[1], device)
        mask = random_masking(x, orders, args.min_mask_rate)

        z = mae(x, mask)
        
        loss = denoiser(x, z, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if args.is_debug and global_rank == 0:
            save_img_as_fig(unpatchify(x*mask.unsqueeze(-1), args.patch_size, x.shape[1], args.channels), filename="mask.png", path="./output")

    return losses

def evaluate(args, mae, denoiser, device, epoch=None, global_rank=0):
    mae.eval()
    denoiser.eval()

    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    num_steps = args.num_images // (args.batch_size * world_size) + 1

    for step in range(num_steps):
        sampled_imgs = sample(args, mae, denoiser, device)

        torch.distributed.barrier() if torch.distributed.is_initialized() else None

        if global_rank == 0:
            if epoch is not None:
                folder = os.path.join(args.output_dir, "generation_{}".format(args.dataset))
                path = os.path.join(folder, 'epoch{}'.format(epoch))

                for i in range(sampled_imgs.shape[0]):
                    save_img_as_fig(sampled_imgs, filename=f"sample_{step}.png", path=path)
            else:
                save_multiple_imgs_as_fig(sampled_imgs, args.patch_size, filename=f"sampled_batch_{step}.png", path=args.output_dir)