import argparse
import numpy as np
import torch
import torch.distributed as dist
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models import MAE, DenoisingMLP
from src.utils import load_checkpoint, save_checkpoint, save_img_as_fig, save_plot, save_multiple_imgs_as_fig
from src.sample import sample
from src.train import train_one_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--is_debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--img_size", type=int, default=28, help="Image size (assumed square)")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--min_mask_rate", type=float, default=0.7, help="Minimum mask rate")
    parser.add_argument("--patch_size", type=int, default=7, help="Patch size for masking")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset to use", choices=["MNIST", "CIFAR10", "IMAGENET"])
    parser.add_argument("--fixed_target_class", type=int, default=8, help="Fix the target class to train on")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--load_checkpoint", action="store_true", help="Load model from checkpoint before training")
    parser.add_argument("--pred_type", type=str, default="x", help="Prediction type for the diffusion", choices=["x", "eps", "v"])
    return parser

def main(): 
    args = create_parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    global_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
    ])

    if args.dataset == "MNIST":
        dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    elif args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    indices = [i for i, t in enumerate(dataset.targets) if t == args.fixed_target_class]
    subset = Subset(dataset, indices)
    dataloader = DataLoader(subset, batch_size=args.batch_size, shuffle=True)

    mae = MAE(args.img_size, patch_size=args.patch_size, hidden_dim=768)
    denoising_mlp = DenoisingMLP(2*args.patch_size**2, output_dim=args.patch_size**2, hidden_dim=512)
    optimizer = optim.Adam(list(mae.parameters()) + list(denoising_mlp.parameters()), lr=1e-4)

    losses = []
    if global_rank == 0:
        epochs = tqdm(range(args.epochs), desc="Training Epochs")
    else:
        epochs = range(args.epochs)

    if args.load_checkpoint:
        load_checkpoint(
            path=f"{args.output_dir}/checkpoint.pt",
            mae=mae,
            denoising_mlp=denoising_mlp,
            optimizer=optimizer
        )
        mae.to(DEVICE)
        denoising_mlp.to(DEVICE)
    else: 
        for epoch in epochs:
            loss = train_one_epoch(dataloader, mae, denoising_mlp, optimizer, args.patch_size, args.min_mask_rate, DEVICE, debug=args.is_debug)
            losses.append(loss)
            if global_rank == 0:
                epochs.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss: {loss:.4f}")
        
        if global_rank == 0:
            save_plot(losses, filename=f"{args.output_dir}/training_loss.png")
            save_checkpoint(
                path=f"{args.output_dir}/checkpoint.pt",
                mae=mae,
                denoising_mlp=denoising_mlp,
                optimizer=optimizer
            )

    if args.evaluate:
        mae.eval()
        denoising_mlp.eval()

        sampled_imgs = sample(denoising_mlp, mae, bsz=16, 
                               seq_len=(args.img_size//args.patch_size)**2,
                               embed_dim=args.patch_size**2, 
                               device=DEVICE, 
                               pred_type=args.pred_type, 
                               num_iter=64, 
                               num_timesteps=32, 
                               debug=args.is_debug
                            )
        
        sampled_imgs = sampled_imgs.cpu().numpy()

        save_multiple_imgs_as_fig(sampled_imgs, args.patch_size, filename="sampled_images.png", path=args.output_dir)

if __name__ == "__main__":
    main()
    









