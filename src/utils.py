import os
import torch
import math
import numpy as np
import matplotlib.pyplot as plt


def patchify(x, patch_size):
    bsz, c, h, w = x.shape
    p = patch_size
    h_, w_ = h // p, w // p

    x = x.reshape(bsz, c, h_, p, w_, p)
    x = torch.einsum('nchpwq->nhwcpq', x)
    x = x.reshape(bsz, h_ * w_, c * p ** 2)
    return x  # [n, l, d]

def unpatchify(x, patch_size, seq_len, channels=3):
    bsz = x.shape[0]
    p = patch_size
    c = channels
    h_, w_ = int(math.sqrt(seq_len)), int(math.sqrt(seq_len))

    x = x.reshape(bsz, h_, w_, c, p, p)
    x = torch.einsum('nhwcpq->nchpwq', x)
    x = x.reshape(bsz, c, h_ * p, w_ * p)
    return x  # [n, c, h, w]

def sample_order(bsz, seq_len, device):
    orders = []
    for _ in range(bsz):
        order = np.arange(seq_len)
        np.random.shuffle(order)
        orders.append(order)
    orders = torch.tensor(np.array(orders), device=device)
    return orders
    
def save_checkpoint(path, mae, denoiser, optimizer):
    torch.save({
        "mae_state_dict": mae.state_dict(),
        "denoising_state_dict": denoiser.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

def load_checkpoint(path, mae, denoiser, optimizer):
    checkpoint = torch.load(path)
    mae.load_state_dict(checkpoint["mae_state_dict"])
    denoiser.load_state_dict(checkpoint["denoising_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

def save_img_as_fig(x, filename, path="./output"):
    with torch.no_grad():
        gen_img = np.round(np.clip(x[0].float().cpu().numpy().transpose([1, 2, 0]) * 255, 0, 255))
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
    plt.figure()
    plt.imshow(gen_img)
    plt.axis("off")
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/" + filename)
    plt.close()

def save_multiple_imgs_as_fig(imgs, patch_size, filename, path="./output"):
    bsz = imgs.shape[0]
    n_row = int(bsz ** 0.5)
    n_col = int(bsz / n_row)

    plt.figure(figsize=(n_col, n_row))
    for i in range(bsz):
        if imgs.shape[1] == 1:
            plot = imgs[i, 0].cpu().numpy()
        else:
            plot = imgs[i].permute(1, 2, 0)
            plot = plot.cpu().numpy()
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(plot, cmap="gray")
        plt.axis("off")

    plt.suptitle("Generated Batch Samples")
    os.makedirs(path, exist_ok=True)
    plt.savefig(os.path.join(path, "batch_plot.png"))
    plt.show()

def save_plot(data, filename, path="./output"):
    plt.figure()
    plt.plot(data)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{filename.split('.')[0]} over Epochs")
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/" + filename)
    plt.close()