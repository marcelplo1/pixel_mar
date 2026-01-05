import argparse
import csv
import os
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import Image

import copy
from denoiser import Denoiser
from model.mae import MAE
from utils.utils import load_checkpoint, save_checkpoint, save_plot
from train_eval import evaluate, train_one_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--is_debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--img_size", type=int, default=64, help="Image size (assumed square)")
    parser.add_argument("--output_dir", type=str, default="./output/debug", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pred_type", type=str, default="x", help="Prediction type for the diffusion", choices=["x", "eps", "v"])
    parser.add_argument("--load_check", action="store_true", help="Load model from checkpoint before training")
    parser.add_argument("--patch_size", type=int, default=4, help="Patch size for masking")
    parser.add_argument("--t_eps", type=float, default=5e-2, help="Epsilon value for time sampling in diffusion")
    parser.add_argument("--bottleneck_dim", type=float, default=256, help="Dimension of the bottleneck layer of the tokenizer")
    parser.add_argument("--bottleneck_dim_final", type=float, default=256, help="Dimension of the bottleneck layer of the final layer")
    parser.add_argument("--activate_ema", action="store_true", help="Use exponential moving average for the model parameters")
    parser.add_argument("--denoising_model", type=str, default="denoisingMLP-S/04", help="Model to denoise the per patch distribution")
    parser.add_argument("--mae_hidden_dim", type=float, default=256, help="Dimension MAE model hidden dimention")

    # Training
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_mask_rate", type=float, default=0.7, help="Minimum mask rate")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Noise scale for diffusion")
    parser.add_argument('--P_mean', default=-0.8, type=float, help='mean for the normal distribution for time sampling')
    parser.add_argument('--P_std', default=0.8, type=float, help='std for the normal distribution for time sampling')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay for the optimizer')

    # Sampling 
    parser.add_argument("--gen_batch_size", type=int, default=512, help="Batch size for sampling")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--num_images", type=int, default=10000, help="Number of images to generate during evaluation")
    parser.add_argument("--num_ar_steps", type=int, default=16, help="Number of sampling steps during evaluation")
    parser.add_argument("--num_timesteps", type=int, default=64, help="Number of timesteps for diffusion process during sampling")
    parser.add_argument("--online_eval_freq", type=int, default=20, help="Frequency of online evaluation during training")
    parser.add_argument("--sampling_method", type=str, default="heun", help="Sampling method to use", choices=["euler", "heun"])
    parser.add_argument("--sampled_img_size", type=int, default=28, help="Size of generated images during evaluation")

    # Dataset
    parser.add_argument("--dataset", type=str, default="mnist", help="Dataset to use", choices=["mnist", "cifar10", "imagenet", "single"])
    parser.add_argument("--class_num", type=int, default=10, help="Number of classes in the dataset")
    parser.add_argument("--has_fixed_target_class", action="store_true", help="Whether to use a fixed target class for training")
    parser.add_argument("--fixed_target_class", type=int, default=0, help="Fix the target class to train on")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--imagnet_path", type=str, default="/mnt/lustre/work/geiger/gwb012/marcel/data/imagenet_subset/train", help="Path to ImageNet dataset")

    return parser

def main(): 
    args = create_parser().parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    global_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    if args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(), 
            transforms.Lambda(lambda x: x.repeat(args.channels, 1, 1)), 
            transforms.Normalize([0.5] * args.channels, [0.5] * args.channels)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(), 
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    if args.dataset == "mnist":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif args.dataset == "cifar10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    elif args.dataset == "single":
        single_image = Image.open("./data/single_image.png").convert("RGB")
        single_image = transform(single_image).unsqueeze(0)  # (1, C, H, W)

        images = single_image.repeat(1000, 1, 1, 1)
        labels = torch.zeros(1000, dtype=torch.long)  # always label 0

        dataset = torch.utils.data.TensorDataset(images, labels)

    if args.has_fixed_target_class:
        indices = [i for i in range(len(dataset)) if dataset[i][1] == args.fixed_target_class]
        dataset = Subset(dataset, indices)
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_train, num_workers=10, pin_memory=True)

    mae = MAE(
        args.img_size, 
        patch_size=args.patch_size,
        channels=args.channels, 
        hidden_dim=args.mae_hidden_dim, 
        depth=6, 
        num_classes=args.class_num, 
        bottleneck_dim=args.bottleneck_dim
    ).to(DEVICE)
    denoiser = Denoiser(args).to(DEVICE)
    
    optimizer = optim.AdamW(
        list(mae.parameters()) + list(denoiser.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.95)
    )

    if args.load_check:
        print("Loading from checkpoint...")
        load_checkpoint(
            path=f"{args.output_dir}/checkpoint.pt",
            mae=mae,
            denoiser=denoiser,
            optimizer=optimizer
        )
        mae.to(DEVICE)
        denoiser.to(DEVICE)
    else:
        mae.ema_params = copy.deepcopy(list(mae.parameters()))
        denoiser.ema_params = copy.deepcopy(list(denoiser.parameters()))

    if args.evaluate:
        print("Starting sampling...")
        evaluate(args, mae, denoiser, DEVICE, epoch=None, global_rank=global_rank, dataset=dataset)
        return

    print("Starting training...")
    losses = []
    fids = []
    for epoch in range(args.epochs):
        epoch_losses = train_one_epoch(args, epoch, dataloader, mae, denoiser, optimizer, DEVICE, global_rank=global_rank)
        losses.append(sum(epoch_losses) / len(epoch_losses))
        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {np.mean(epoch_losses):.4f}")
            save_plot(losses, filename="loss.png", path=args.output_dir, y_label="Loss")

        if int(epoch) % args.online_eval_freq == 0:
            print("Starting online evaluation...")
            evaluate(args, mae, denoiser, DEVICE, epoch=epoch, global_rank=global_rank, fids=fids, dataset=dataset)

        if global_rank == 0:
            save_plot(fids, "fid.png", args.output_dir, y_label="FID")
            csv_file = os.path.join(args.output_dir, "fids.csv")
            with open(csv_file, "w", newline="") as f:
                writer = csv.writer(f)
                for item in fids:
                    writer.writerow([item])
            file_np = os.path.join(args.output_dir, "fids.npz")
            np.savez(file_np, fids=fids)

    if global_rank == 0:
        save_checkpoint(
            path=f"{args.output_dir}/checkpoint.pt",
            mae=mae,
            denoiser=denoiser,
            optimizer=optimizer
        )
        
if __name__ == "__main__":
    main()
    









