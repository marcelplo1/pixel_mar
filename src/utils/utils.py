import csv
import os
import cv2
import torch
import math
import numpy as np
import io
import lmdb
import pickle
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

class SingleImageDataset(Dataset):
    """Dataset that returns the same image repeatedly, for overfitting tests."""
    def __init__(self, image_path, img_size, label=0, length=64):
        from torchvision import transforms
        self.length = length
        self.label = label
        transform = transforms.Compose([
            transforms.Lambda(lambda img: center_crop_arr(img, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        img = Image.open(image_path).convert('RGB')
        self.image = transform(img)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.image, self.label


class ImageNetLMDB(Dataset):
    def __init__(self, db_path, transform=None):
        self.db_path = db_path
        self.transform = transform
        
        # Open the environment once to get the number of entries
        env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with env.begin(write=False) as txn:
            # txn.stat()['entries'] counts all keys including metadata.
            # Use the 'length' metadata key if available, otherwise find
            # the actual number of sequential integer keys.
            length_val = txn.get(b'length') or txn.get(b'num_samples')
            if length_val is not None:
                self.length = int(length_val)
            else:
                # Binary search for the last valid integer key
                lo, hi = 0, txn.stat()['entries']
                while lo < hi:
                    mid = (lo + hi) // 2
                    if txn.get(str(mid).encode('ascii')) is not None:
                        lo = mid + 1
                    else:
                        hi = mid
                self.length = lo
        env.close()
        
        self.env = None

    def _init_db(self):
        """Initializes the LMDB environment for each worker process."""
        self.env = lmdb.open(self.db_path, readonly=True, lock=False, readahead=False, meminit=False)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if self.env is None:
            self._init_db()

        with self.env.begin(write=False) as txn:
            byte_key = str(index).encode('ascii')
            byteflow = txn.get(byte_key)

        if byteflow is None:
            raise KeyError(f"Key {index} not found in LMDB")

        data = pickle.loads(byteflow)
        img = Image.open(io.BytesIO(data['image'])).convert('RGB')
        label = data['label']

        if self.transform is not None:
            img = self.transform(img)
        return img, label