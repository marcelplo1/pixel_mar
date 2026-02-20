import csv
import os
import cv2
import torch
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image 


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    min_lr = 0.0
    lr_schedule = "constant"
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
    else:
        if lr_schedule == "constant":
            lr = args.lr
        elif lr_schedule == "cosine":
            lr = min_lr + (args.lr - min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
        else:
            raise NotImplementedError
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr

def patchify(x, p):
    """
    x: (B, C, H, W)
    x: (B, N, patch_size**2 * C)
    """
    bsz, c, h, w = x.shape
    h_, w_ = h // p, w // p

    x = x.reshape(bsz, c, h_, p, w_, p)
    x = torch.einsum('nchpwq->nhwcpq', x)
    x = x.reshape(bsz, h_ * w_, c * p ** 2)
    return x  # [B, N, D]

def unpatchify(x, p, channels=3):
    """
    x: (B, N, patch_size**2 * C)
    imgs: (B, C, H, W)
    """
    bsz, n, d = x.shape
    c = channels
    h = w = int(n ** 0.5)
    assert h * w == n

    x = x.reshape(shape=(bsz, h, w, c, p, p))
    x = torch.einsum('nhwcpq->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
    return imgs

def sample_order(bsz, seq_len, device):
    orders = []
    for _ in range(bsz):
        order = np.arange(seq_len)
        np.random.shuffle(order)
        orders.append(order)
    orders = torch.tensor(np.array(orders), device=device)
    return orders

def save_img_as_fig(x, file_path, size=32):
    with torch.no_grad():
        x = (x + 1) / 2
        gen_img = np.clip(x[0].float().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    gen_img = cv2.resize(gen_img, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(file_path, gen_img[:, :, ::-1])

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

def save_plot(data, filename, path="./output", y_label="y-axis"):
    plt.figure()
    plt.plot(data)
    plt.xlabel("x-epochs")
    plt.ylabel(y_label)
    plt.title(f"{filename.split('.')[0]} over Epochs")
    os.makedirs(path, exist_ok=True)
    plt.savefig(f"{path}/" + filename)
    plt.close()

def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

def write_csv(name, path, list):
    csv_file = os.path.join(path, name)
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        for item in list:
            writer.writerow([item])