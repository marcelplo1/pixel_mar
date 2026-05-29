
"""
Analysis of 5 representation spaces:
  1. DINOv2-B       (768-d  semantic features, base)
  2. DINOv2-L       (1024-d semantic features, large)
  3. MAE-B          (768-d  reconstruction-oriented features)
  4. SD-VAE (KL-16) (16-d   KL-regularized latents)
  5. Pixel          (768-d  raw patch pixels)
"""

import sys
import os
import random
import argparse

import json
import numpy as np
import torch
from PIL import Image as PILImage
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
        return {v[0]: v[1] for v in idx_to_info.values()}
    except Exception:
        return None

def wnid_to_human(wnid, label_map):
    if label_map and wnid in label_map:
        return label_map[wnid]
    return wnid

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

    label_map = {cls: i for i, cls in enumerate(chosen)}
    imagenet_names = get_imagenet_label_map()
    class_names = [wnid_to_human(dataset.classes[c], imagenet_names) for c in chosen]

    return indices, label_map, chosen, class_names

@torch.no_grad()
def encode_all(dataloader, encoders, device, per_patch=False, seed=42):
    """Run all encoders on the dataset and collect embeddings + labels."""
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
    max_abs = np.max(np.abs(coords))
    if max_abs > 0:
        coords = coords / max_abs * target_range
    return coords

@torch.no_grad()
def measure_lpips(tokenizers, dataloader, device, num_images=500):
    """Compute mean LPIPS between original images and their AE reconstructions.
    tokenizers: dict {name: nn.Module with .encode() and .decode()}
    Returns:    dict {name: float mean LPIPS}
    """
    try:
        import lpips as lpips_lib
    except ImportError:
        raise ImportError("pip install lpips")

    loss_fn = lpips_lib.LPIPS(net='vgg').to(device).eval()

    scores = {name: [] for name in tokenizers}
    collected = 0

    for images, _ in dataloader:
        if collected >= num_images:
            break
        remaining = num_images - collected
        if images.shape[0] > remaining:
            images = images[:remaining]
        images = images.to(device)

        for name, tok in tokenizers.items():
            z = tok.encode(images)
            x_rec = tok.decode(z).clamp(-1, 1)
            if x_rec.shape[-2:] != images.shape[-2:]:
                x_rec = torch.nn.functional.interpolate(
                    x_rec, size=images.shape[-2:], mode='bicubic', align_corners=False
                ).clamp(-1, 1)
            d = loss_fn(images, x_rec)       # [B, 1, 1, 1]
            scores[name].extend(d.view(-1).cpu().tolist())

        collected += images.shape[0]
        print(f"  LPIPS: {collected}/{num_images} images", end='\r')

    print()
    return {name: float(np.mean(vals)) for name, vals in scores.items()}


def plot_tsne(tsne_results, labels, class_names, save_dir):
    """Create one t-SNE plot per representation space."""
    order = ['DINOv2-B', 'DINOv2-L', 'MAE-B', 'SD-VAE (KL-16)', 'Pixel']
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
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_aspect('equal')

        ax.legend(loc='best', fontsize=8, markerscale=3, framealpha=0.8,
                  ncol=2 if num_classes > 10 else 1)

        plt.tight_layout()
        fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        save_path = os.path.join(save_dir, f'tsne_{fname}.pdf')
        fig.savefig(save_path, bbox_inches='tight')
        print(f"  Saved {save_path}")
        plt.close(fig)


def save_pca_viz(z_np, file_path, img_size):
    """Project to 3 PCA components for RGB."""
    N = z_np.shape[0]
    H = W = int(N ** 0.5)
    mean = z_np.mean(axis=0, keepdims=True)
    x = z_np - mean
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    k = min(3, vt.shape[0])
    comps = vt[:k].T
    if k < 3:
        comps = np.concatenate([comps, np.zeros((comps.shape[0], 3 - k))], axis=1)
    proj = (z_np - mean) @ comps
    rgb = np.empty((N, 3), dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(proj[:, c], [1.0, 99.0])
        if hi <= lo:
            lo, hi = float(proj[:, c].min()), float(proj[:, c].max())
        if hi <= lo:
            lo, hi = 0.0, 1.0
        rgb[:, c] = np.clip((proj[:, c] - lo) / (hi - lo), 0.0, 1.0)
    img = PILImage.fromarray((rgb.reshape(H, W, 3) * 255).astype(np.uint8))
    img.resize((img_size, img_size), PILImage.NEAREST).save(file_path)


def save_tensor_img(x, file_path, img_size):
    """x: (1, 3, H, W) tensor in [-1, 1] → save as PNG."""
    arr = ((x[0].float().cpu().numpy().transpose(1, 2, 0) + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    PILImage.fromarray(arr).resize((img_size, img_size), PILImage.BILINEAR).save(file_path)


def _try_decode(tok, z):
    """Decode z if the tokenizer has a trained decoder, else return None."""
    if hasattr(tok, 'decoder') and tok.decoder is None:
        return None
    try:
        return tok.decode(z).clamp(-1, 1)
    except Exception:
        return None


def build_tokenizers(device, args, with_decoder=False):
    from tokenizers.dinov2_tokenizer import Dinov2Tokenizer
    from tokenizers.mae_tokenizer import MaeTokenizer
    from tokenizers.vae_tokenizer import VaeTokenizer
    decoder_config = args.rae_decoder_config if with_decoder else None
    return {
        'SD-VAE (KL-16)': VaeTokenizer(
            vae_path=os.path.join(PROJECT_ROOT, 'tokenizer_models', 'vae', 'kl16.ckpt'),
            embed_dim=16, latent_scale=0.2325, img_size=args.img_size,
        ).to(device).eval(),
        'DINOv2-B': Dinov2Tokenizer(
            dinov2_path='facebook/dinov2-with-registers-base',
            rae_decoder_config_path=decoder_config,
            rae_decoder_ckp=args.dinov2_decoder_ckp if with_decoder else None,
            rae_norm_stats=os.path.join(PROJECT_ROOT, 'tokenizer_models', 'dinov2', 'stat.pt'),
        ).to(device).eval(),
        'DINOv2-L': Dinov2Tokenizer(
            dinov2_path='facebook/dinov2-with-registers-large',
            rae_decoder_config_path=decoder_config,
            rae_decoder_ckp=args.dinov2_large_decoder_ckp if with_decoder else None,
            rae_norm_stats=os.path.join(PROJECT_ROOT, 'tokenizer_models', 'dinov2_large', 'stat.pt'),
        ).to(device).eval(),
        'MAE-B': MaeTokenizer(
            mae_path='facebook/vit-mae-base',
            rae_decoder_config_path=decoder_config,
            rae_decoder_ckp=args.mae_decoder_ckp if with_decoder else None,
            rae_norm_stats=os.path.join(PROJECT_ROOT, 'tokenizer_models', 'mae', 'stat.pt'),
        ).to(device).eval(),
    }


def main():
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument('--data_path',  type=str, default='/weka/datasets/ImageNet2012')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--device',     type=str, default='cuda')
    parser.add_argument('--seed',       type=int, default=42)
    parser.add_argument('--img_size',   type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)

    # Decoder checkpoints (shared by LPIPS and token_viz)
    parser.add_argument('--rae_decoder_config',       type=str, default=None)
    parser.add_argument('--dinov2_decoder_ckp',       type=str, default=None)
    parser.add_argument('--dinov2_large_decoder_ckp', type=str, default=None)
    parser.add_argument('--mae_decoder_ckp',          type=str, default=None)

    # t-SNE
    tsne = parser.add_argument_group('t-SNE')
    tsne.add_argument('--tsne',           action='store_true')
    tsne.add_argument('--num_classes',    type=int,   default=20)
    tsne.add_argument('--imgs_per_class', type=int,   default=500)
    tsne.add_argument('--perplexity',     type=float, default=30)
    tsne.add_argument('--patch_size',     type=int,   default=16)
    tsne.add_argument('--per_patch',      action='store_true',
                      help='Use one random patch per image instead of mean-pooling')

    # LPIPS
    lpips = parser.add_argument_group('LPIPS')
    lpips.add_argument('--lpips',                  action='store_true')
    lpips.add_argument('--lpips_num_images',        type=int, default=500)
    lpips.add_argument('--save_reconstructions',    action='store_true')
    lpips.add_argument('--reconstruction_classes',  type=int, nargs='*',
                       default=[207, 817, 921, 281, 130, 975],
                       help='ImageNet class indices for the reconstruction grid')
    lpips.add_argument('--reconstruction_image_idx', type=int, default=10)

    # Token viz
    tviz = parser.add_argument_group('Token viz')
    tviz.add_argument('--token_viz',            action='store_true')
    tviz.add_argument('--token_viz_image',       type=str,   default=None)
    tviz.add_argument('--token_viz_mask_ratio',  type=float, default=0.8)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(PROJECT_ROOT, 'analysis', 'plots')
    if args.rae_decoder_config is None:
        args.rae_decoder_config = os.path.join(
            PROJECT_ROOT, 'src', 'tokenizers', 'rae_decoder_configs', 'ViTXL'
        )
    if args.dinov2_decoder_ckp is None:
        args.dinov2_decoder_ckp = os.path.join(PROJECT_ROOT, 'tokenizer_models', 'dinov2', 'model.pt')
    if args.dinov2_large_decoder_ckp is None:
        args.dinov2_large_decoder_ckp = os.path.join(PROJECT_ROOT, 'tokenizer_models', 'dinov2_large', 'model.pt')
    if args.mae_decoder_ckp is None:
        args.mae_decoder_ckp = os.path.join(PROJECT_ROOT, 'tokenizer_models', 'mae', 'model.pt')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Load tokenizers
    with_decoder = args.lpips or args.save_reconstructions or args.token_viz
    print("Loading tokenizers ...")
    tokenizers = build_tokenizers(device, args, with_decoder=with_decoder)

    if args.tsne:
        print("Loading ImageNet dataset ...")
        full_dataset = datasets.ImageFolder(
            os.path.join(args.data_path, 'train'), transform=transform
        )

        indices, label_map, chosen_classes, class_names = select_random_classes(
            full_dataset, args.num_classes, args.imgs_per_class, args.seed
        )
        dataloader = DataLoader(Subset(full_dataset, indices), batch_size=args.batch_size,
                                shuffle=False, num_workers=8, pin_memory=True)

        print(f"Selected {args.num_classes} classes, {len(indices)} images total")
        print(f"Classes: {class_names}")

        encoders = {name: (lambda t: lambda imgs: t.encode(imgs))(tok)
                    for name, tok in tokenizers.items()}
        encoders['Pixel'] = lambda imgs: pixel_patchify(imgs, args.patch_size)

        print("Encoding images ...")
        embeddings, raw_labels = encode_all(dataloader, encoders, device,
                                            per_patch=args.per_patch, seed=args.seed)
        labels = np.array([label_map[l] for l in raw_labels])
        for name, emb in embeddings.items():
            print(f"  {name}: {emb.shape}")

        print("Running t-SNE ...")
        tsne_results = run_tsne(embeddings, perplexity=args.perplexity, seed=args.seed)

        print("Plotting ...")
        tsne_dir = 'tsne_token' if args.per_patch else 'tsne'
        plot_tsne(tsne_results, labels, class_names, os.path.join(args.output_dir, tsne_dir))

    if args.lpips or args.save_reconstructions:
        print("Building val loader ...")
        full_val = datasets.ImageFolder(
            os.path.join(args.data_path, 'val'), transform=transform
        )
        rng = np.random.RandomState(args.seed)
        eval_indices = rng.permutation(len(full_val))[:args.lpips_num_images].tolist()
        eval_loader = DataLoader(Subset(full_val, eval_indices), batch_size=args.batch_size,
                                 shuffle=False, num_workers=8, pin_memory=True)

        if args.lpips:
            print(f"Measuring LPIPS on {args.lpips_num_images} val images ...")
            lpips_scores = measure_lpips(tokenizers, eval_loader, device, args.lpips_num_images)
            print("\nLPIPS reconstruction quality (lower is better):")
            for name, score in lpips_scores.items():
                print(f"  {name:14s}: {score:.4f}")

        if args.save_reconstructions:
            print("Saving reconstructions ...")
            rec_dir = os.path.join(args.output_dir, 'reconstructions')
            os.makedirs(rec_dir, exist_ok=True)

            if args.reconstruction_classes:
                val_targets = np.array(full_val.targets)
                sample_images, file_labels = [], []
                for cls in args.reconstruction_classes:
                    cls_idx = np.where(val_targets == cls)[0]
                    if len(cls_idx) == 0:
                        raise ValueError(f"Class {cls} has no images in {args.data_path}/val")
                    if args.reconstruction_image_idx >= len(cls_idx):
                        raise ValueError(
                            f"--reconstruction_image_idx={args.reconstruction_image_idx} out of "
                            f"range for class {cls} (only {len(cls_idx)} images)"
                        )
                    sample_images.append(full_val[int(cls_idx[args.reconstruction_image_idx])][0])
                    file_labels.append(f"class_{cls}")
                sample_images = torch.stack(sample_images)
            else:
                sample_images, _ = next(iter(eval_loader))
                sample_images = sample_images[:8]
                file_labels = [str(i) for i in range(len(sample_images))]

            with torch.no_grad():
                for img, label in zip(sample_images, file_labels):
                    x = img.unsqueeze(0).to(device)
                    save_tensor_img(x, os.path.join(rec_dir, f'{label}_original.png'), args.img_size)
                    for name, tok in tokenizers.items():
                        z = tok.encode(x)
                        x_rec = tok.decode(z).clamp(-1, 1)
                        if x_rec.shape[-2:] != x.shape[-2:]:
                            x_rec = torch.nn.functional.interpolate(
                                x_rec, size=x.shape[-2:], mode='bicubic', align_corners=False
                            ).clamp(-1, 1)
                        fname = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
                        save_tensor_img(x_rec, os.path.join(rec_dir, f'{label}_{fname}.png'), args.img_size)
            print(f"  Saved reconstructions to {rec_dir}")

    if args.token_viz:
        if args.token_viz_image is None:
            raise ValueError("--token_viz requires --token_viz_image <path>")
        print("Running token visualization ...")
        x = transform(PILImage.open(args.token_viz_image).convert('RGB')).unsqueeze(0).to(device)

        out_dir = os.path.join(args.output_dir, 'token_viz')
        os.makedirs(out_dir, exist_ok=True)
        save_tensor_img(x, os.path.join(out_dir, 'original.png'), args.img_size)

        rng = torch.Generator()
        for name, tok in tokenizers.items():
            print(f"  [{name}] encoding ...")
            z = tok.encode(x)
            N = z.shape[1]

            rng.manual_seed(args.seed)
            num_masked = int(N * args.token_viz_mask_ratio)
            mask = torch.zeros(N, dtype=torch.bool, device=device)
            mask[torch.randperm(N, generator=rng)[:num_masked].to(device)] = True

            z_masked = z.clone()
            z_masked[0, mask] = 0.0

            H = W = int(N ** 0.5)
            mask_pixel = (mask.view(H, W)
                          .repeat_interleave(args.img_size // H, dim=0)
                          .repeat_interleave(args.img_size // W, dim=1))
            x_masked = x.clone()
            x_masked[0, :, mask_pixel] = 0.0
            save_tensor_img(x_masked, os.path.join(out_dir, f'{name}_original_masked.png'), args.img_size)

            save_pca_viz(z[0].float().cpu().numpy(),
                         os.path.join(out_dir, f'{name}_pca_full.png'), args.img_size)
            save_pca_viz(z_masked[0].float().cpu().numpy(),
                         os.path.join(out_dir, f'{name}_pca_masked.png'), args.img_size)

            x_rec = _try_decode(tok, z)
            x_rec_masked = _try_decode(tok, z_masked)
            if x_rec is not None:
                save_tensor_img(x_rec,        os.path.join(out_dir, f'{name}_decoded_full.png'),   args.img_size)
                save_tensor_img(x_rec_masked, os.path.join(out_dir, f'{name}_decoded_masked.png'), args.img_size)

        print(f"  Saved token viz to {out_dir}")

    print("Done!")


if __name__ == '__main__':
    main()
