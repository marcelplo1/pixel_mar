"""
t-SNE visualization of 4 representation spaces:
  1. DINOv2 (768-d semantic features)
  2. MAE    (768-d reconstruction-oriented features)
  3. VAE    (16-d KL-regularized latents)
  4. Pixel  (768-d raw patch pixels)

For each space we mean-pool across spatial tokens to get one vector per image,
then run t-SNE and plot a 2x2 comparison figure coloured by ImageNet class.
"""

import sys
import os
import random
import argparse

import json
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from utils.utils import center_crop_arr, patchify


def get_imagenet_label_map():
    """Load human-readable ImageNet class names from torchvision's bundled json."""
    try:
        import torchvision
        json_path = os.path.join(os.path.dirname(torchvision.__file__),
                                 '_meta', 'imagenet_class_index.json')
        with open(json_path) as f:
            idx_to_info = json.load(f)
        # {wnid: human_name}
        return {v[0]: v[1] for v in idx_to_info.values()}
    except Exception:
        return None

def wnid_to_human(wnid, label_map):
    """Convert a WordNet ID like 'n01440764' to a human-readable name."""
    if label_map and wnid in label_map:
        return label_map[wnid]
    return wnid  # fallback to wnid

def select_random_classes(dataset, num_classes=10, imgs_per_class=500, seed=42):
    """Randomly pick `num_classes` classes and sample `imgs_per_class` images each."""
    rng = random.Random(seed)
    all_classes = list(range(len(dataset.classes)))
    chosen = sorted(rng.sample(all_classes, num_classes))

    targets = np.array(dataset.targets)
    indices = []
    for cls in chosen:
        cls_indices = np.where(targets == cls)[0]
        if len(cls_indices) > imgs_per_class:
            cls_indices = rng.sample(cls_indices.tolist(), imgs_per_class)
        indices.extend(cls_indices)

    # Remap labels to 0..num_classes-1 for clean plotting
    label_map = {cls: i for i, cls in enumerate(chosen)}

    # Resolve human-readable names
    imagenet_names = get_imagenet_label_map()
    class_names = [wnid_to_human(dataset.classes[c], imagenet_names) for c in chosen]

    return indices, label_map, chosen, class_names

@torch.no_grad()
def encode_all(dataloader, encoders, device, per_patch=False, seed=42):
    """Run all encoders on the dataset and collect embeddings + labels.
    """
    embeddings = {name: [] for name in encoders}
    labels = []
    rng = random.Random(seed)

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        for name, enc_fn in encoders.items():
            z = enc_fn(images)                    # [B, N, D]
            if per_patch:
                idx = rng.randint(0, z.shape[1] - 1)
                z_out = z[:, idx, :].float()      # [B, D]  one random patch
            else:
                z_out = z.float().mean(dim=1)     # [B, D]  mean-pool over patches
            embeddings[name].append(z_out.cpu())
        labels.append(targets)

    for name in embeddings:
        embeddings[name] = torch.cat(embeddings[name], dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return embeddings, labels

def pixel_patchify(images, patch_size=16):
    """Patchify images into [B, N, D] raw pixel tokens."""
    return patchify(images, patch_size)

def run_tsne(embeddings, perplexity=30, seed=42):
    """Run t-SNE on each representation space."""
    results = {}
    for name, X in embeddings.items():
        print(f"  Running t-SNE on {name} (shape {X.shape}) ...")
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed,
                     init='pca', learning_rate='auto')
        results[name] = tsne.fit_transform(X)
    return results

def rescale_tsne(coords, target_range=100):
    """Rescale t-SNE coordinates to [-target_range, target_range]."""
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs * target_range
    return coords

def plot_tsne(tsne_results, labels, class_names, save_dir):
    """Create 4 separate t-SNE plots (one per representation space)."""
    order = ['DINOv2', 'MAE', 'VAE', 'Pixel']
    num_classes = len(class_names)
    cmap = plt.cm.get_cmap('tab10' if num_classes <= 10 else 'tab20', num_classes)

    os.makedirs(save_dir, exist_ok=True)

    for name in order:
        coords = rescale_tsne(tsne_results[name].copy(), target_range=100)

        fig, ax = plt.subplots(figsize=(10, 8))
        for c in range(num_classes):
            mask = labels == c
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=8, alpha=0.9, color=cmap(c), label=class_names[c], edgecolors='none')

        ax.set_xlim(-100, 100)
        ax.set_ylim(-100, 100)
        ax.set_xlabel('Dim 1', fontsize=13)
        ax.set_ylabel('Dim 2', fontsize=13)
        #ax.set_title(name, fontsize=16, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal')

        ax.legend(loc='best', fontsize=8, markerscale=3, framealpha=0.8,
                  ncol=2 if num_classes > 10 else 1)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'tsne_{name.lower()}.png')
        fig.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  Saved {save_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/mnt/lustre/datasets/ImageNet2012')
    parser.add_argument('--num_classes', type=int, default=20)
    parser.add_argument('--imgs_per_class', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--perplexity', type=float, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save the 4 t-SNE plots')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--per_patch', action='store_true',
                        help='Use one random patch per image instead of mean-pooling')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'helpers', 'tsne_plots')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # --- Dataset ---
    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    print("Loading ImageNet dataset ...")
    full_dataset = datasets.ImageFolder(
        os.path.join(args.data_path, 'train'), transform=transform
    )

    indices, label_map, chosen_classes, class_names = select_random_classes(
        full_dataset, args.num_classes, args.imgs_per_class, args.seed
    )
    subset = Subset(full_dataset, indices)

    # Wrap to remap labels
    class RemappedDataset(torch.utils.data.Dataset):
        def __init__(self, subset, label_map, orig_dataset):
            self.subset = subset
            self.label_map = label_map
            self.targets = orig_dataset.targets

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, idx):
            img, label = self.subset[idx]
            return img, self.label_map[label]

    dataset = RemappedDataset(subset, label_map, full_dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=8, pin_memory=True)

    total = len(dataset)
    print(f"Selected {args.num_classes} classes, {total} images total")
    print(f"Classes: {class_names}")

    # --- Load tokenizers ---
    print("Loading DINOv2 tokenizer ...")
    from rae.dinov2_tokenizer import Dinov2Tokenizer
    dinov2 = Dinov2Tokenizer(
        dinov2_path='facebook/dinov2-with-registers-base',
        rae_decoder_config_path=None,  # no decoder needed
        rae_decoder_ckp=None,
        rae_norm_stats=os.path.join(PROJECT_ROOT, 'rae_models', 'dinov2', 'stat.pt'),
    ).to(device).eval()

    print("Loading MAE tokenizer ...")
    from rae.mae_tokenizer import MaeTokenizer
    mae = MaeTokenizer(
        mae_path='facebook/vit-mae-base',
        rae_decoder_config_path=None,  # no decoder needed
        rae_decoder_ckp=None,
        rae_norm_stats=os.path.join(PROJECT_ROOT, 'rae_models', 'mae', 'stat.pt'),
    ).to(device).eval()

    print("Loading VAE tokenizer ...")
    from rae.vae_tokenizer import VaeTokenizer
    vae = VaeTokenizer(
        vae_path=os.path.join(PROJECT_ROOT, 'rae_models', 'vae', 'kl16.ckpt'),
        embed_dim=16,
        latent_scale=0.2325,
        img_size=args.img_size,
    ).to(device).eval()

    patch_size = args.patch_size

    encoders = {
        'DINOv2': lambda imgs: dinov2.encode(imgs),
        'MAE':    lambda imgs: mae.encode(imgs),
        'VAE':    lambda imgs: vae.encode(imgs),
        'Pixel':  lambda imgs: pixel_patchify(imgs, patch_size),
    }

    # --- Encode ---
    print("Encoding images ...")
    embeddings, labels = encode_all(dataloader, encoders, device,
                                     per_patch=args.per_patch, seed=args.seed)
    for name, emb in embeddings.items():
        print(f"  {name}: {emb.shape}")

    # --- t-SNE ---
    print("Running t-SNE ...")
    tsne_results = run_tsne(embeddings, perplexity=args.perplexity, seed=args.seed)

    # --- Plot ---
    print("Plotting ...")
    plot_tsne(tsne_results, labels, class_names, args.output_dir)
    print("Done!")


if __name__ == '__main__':
    main()