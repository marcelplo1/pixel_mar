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

def unpatchify(x, patch_size, seq_len):
    bsz = x.shape[0]
    p = patch_size
    c = 1
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

def compute_loss(x1, eps, xt, t, pred_raw, pred_type, loss_type, debug=False):
    if loss_type == "v":
        if pred_type == "x":
            x1_hat = pred_raw
            v_hat = (x1_hat - xt) / (1-t).clamp(min=0.05)
        elif pred_type == "eps":
            eps_hat = pred_raw
            v_hat = (xt - eps_hat) / t
        elif pred_type == "v":
            v_hat = pred_raw
        # v_target = x1 - eps
        v_target = (x1 - xt) / (1 - t).clamp(min=0.05)
        if debug:
            save_patch_as_figure(x1[0], filename="x.png")
            save_patch_as_figure(x1_hat[0], filename="x_prediction.png")
        return ((v_hat - v_target)**2).mean()
    
    elif loss_type == "x":
        if pred_type == "x":
            x1_hat = pred_raw
        elif pred_type == "eps":
            eps_hat = pred_raw
            x1_hat = (xt - eps_hat*(1-t)) / t
        elif pred_type == "v":
            v_hat = pred_raw
            x1_hat = xt + (1 - t) * v_hat
        if debug:
            save_patch_as_figure(x1[0], filename="x.png")
            save_patch_as_figure(x1_hat[0], filename="x_prediction.png")
        return ((x1_hat - x1)**2).mean()

    elif loss_type == "eps":
        if pred_type == "x":
            x1_hat = pred_raw
            eps_hat = (xt - t*x1_hat) / (1 - t)
        elif pred_type == "eps":
            eps_hat = pred_raw
        elif pred_type == "v":
            v_hat = pred_raw
            eps_hat = xt - t * v_hat
        if debug:
            save_patch_as_figure(x1[0], filename="x.png")
            save_patch_as_figure(x1_hat[0], filename="x_prediction.png")
        return ((eps_hat - eps)**2).mean()
    
def save_checkpoint(path, mae, denoising_mlp, optimizer):
    torch.save({
        "mae_state_dict": mae.state_dict(),
        "denoising_state_dict": denoising_mlp.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }, path)

def load_checkpoint(path, mae, denoising_mlp, optimizer):
    checkpoint = torch.load(path)
    mae.load_state_dict(checkpoint["mae_state_dict"])
    denoising_mlp.load_state_dict(checkpoint["denoising_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
def save_patch_as_figure(x, filename, path="./output"):
    x = unpatchify(x.unsqueeze(0), patch_size=1, seq_len=1)
    with torch.no_grad():
        patch_img = x.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(patch_img, cmap="gray")
    plt.axis("off")
    plt.savefig(f"{path}/" + filename)
    plt.close()

def save_img_as_fig(x, patch_size, filename, path="./output"):
    x = unpatchify(x, patch_size=patch_size, seq_len=x.shape[1])
    with torch.no_grad():
        img = x.squeeze().cpu().numpy()
    plt.figure()
    plt.imshow(img[0], cmap="gray")
    plt.axis("off")
    plt.savefig(f"{path}/" + filename)
    plt.close()

def save_multiple_imgs_as_fig(imgs, patch_size, filename, path="./output"):
    bsz = imgs.shape[0]
    n_row = int(bsz ** 0.5)
    n_col = int(bsz / n_row)

    plt.figure(figsize=(n_col, n_row))
    for i in range(bsz):
        if imgs.shape[1] == 1:
            plot = imgs[i, 0]
        else:
            plot = imgs[i]
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(plot)
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
    plt.savefig(f"{path}/" + filename)
    plt.close()