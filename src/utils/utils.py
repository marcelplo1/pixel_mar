import csv
import os
import cv2
import torch
import math
import numpy as np
import io
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image 


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
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

def sample_order(bsz, seq_len, device, strategy='random'):
    """Generate token ordering for AR generation.
    """
    if strategy == 'random' or strategy is None:
        orders = []
        for _ in range(bsz):
            order = np.arange(seq_len)
            np.random.shuffle(order)
            orders.append(order)
        return torch.tensor(np.array(orders), device=device)
    elif strategy == 'raster':
        order = list(range(seq_len))
        order = torch.tensor(order, device=device).unsqueeze(0).expand(bsz, -1)
        return order

    raise ValueError(f"Unknown order strategy: {strategy}")

def save_pca_viz(z, file_path, img_size, idx=None):
    """Project token features to 3 PCA components → RGB and save one image of the batch.
    """
    if idx is None:
        idx = int(np.random.randint(z.shape[0]))
    z_np = z[idx].float().cpu().numpy()  # (N, D)
    N = z_np.shape[0]
    H = W = int(N ** 0.5)

    mean = z_np.mean(axis=0, keepdims=True)
    x = z_np - mean
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    comps = vt[:3].T  # (D, 3)

    proj = (z_np - mean) @ comps  # (N, 3)
    rgb = np.empty((N, 3), dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(proj[:, c].min()), float(proj[:, c].max())
        if hi <= lo:
            lo, hi = 0.0, 1.0
        rgb[:, c] = np.clip((proj[:, c] - lo) / (hi - lo), 0.0, 1.0)

    img = (rgb.reshape(H, W, 3) * 255).astype(np.uint8)
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(file_path, img[:, :, ::-1])


def save_img_as_fig(x, file_path, size=32):
    with torch.no_grad():
        x = (x + 1) / 2
        gen_img = np.clip(x[0].float().cpu().numpy().transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
    gen_img = cv2.resize(gen_img, (size, size), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(file_path, gen_img[:, :, ::-1])


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