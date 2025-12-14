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

from denoiser import Denoiser
from diffusion_mlp import SimpleMLPAdaLN
from models import MAE, DenoisingMLP
from utils import load_checkpoint, save_checkpoint, save_img_as_fig, save_plot, save_multiple_imgs_as_fig
from sample import sample
from train import evaluate, train_one_epoch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--is_debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--img_size", type=int, default=32, help="Image size (assumed square)")
    parser.add_argument("--output_dir", type=str, default="./output/debug", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pred_type", type=str, default="x", help="Prediction type for the diffusion", choices=["x", "eps", "v"])
    parser.add_argument("--load_check", action="store_true", help="Load model from checkpoint before training")
    parser.add_argument("--patch_size", type=int, default=8, help="Patch size for masking")
    parser.add_argument("--t_eps", type=float, default=1e-3, help="Epsilon value for time sampling in diffusion")

    # Training
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_mask_rate", type=float, default=0.7, help="Minimum mask rate")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Noise scale for diffusion")
    parser.add_argument('--P_mean', default=-0.8, type=float, help='mean for the normal distribution for time sampling')
    parser.add_argument('--P_std', default=0.8, type=float, help='std for the normal distribution for time sampling')

    # Sampling
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Batch size for sampling")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--num_images", type=int, default=10, help="Number of images to generate during evaluation")
    parser.add_argument("--num_ar_steps", type=int, default=32, help="Number of sampling steps during evaluation")
    parser.add_argument("--num_timesteps", type=int, default=100, help="Number of timesteps for diffusion process during sampling")
    parser.add_argument("--online_eval_freq", type=int, default=5, help="Frequency of online evaluation during training")
    parser.add_argument("--sampling_method", type=str, default="euler", help="Sampling method to use", choices=["euler", "heun"])


    # Dataset
    parser.add_argument("--dataset", type=str, default="MNIST", help="Dataset to use", choices=["MNIST", "CIFAR10", "IMAGENET"])
    parser.add_argument("--class_num", type=int, default=10, help="Number of classes in the dataset")
    parser.add_argument("--has_fixed_target_class", action="store_true", help="Whether to use a fixed target class for training")
    parser.add_argument("--fixed_target_class", type=int, default=0, help="Fix the target class to train on")
    parser.add_argument("--channels", type=int, default=1, help="Number of image channels")

    return parser

def main(): 
    args = create_parser().parse_args()

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
        channels = 1
    elif args.dataset == "CIFAR10":
        dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)

    if args.has_fixed_target_class:
        indices = [i for i in range(len(dataset)) if dataset[i][1] == args.fixed_target_class]
        dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    mae = MAE(args.img_size, patch_size=args.patch_size, hidden_dim=64, channels=args.channels).to(DEVICE)
    denoiser = Denoiser(args).to(DEVICE)
    optimizer = optim.Adam(list(mae.parameters()) + list(denoiser.parameters()), lr=1e-4)

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

    if args.evaluate:
        print("Starting sampling...")
        evaluate(args, mae, denoiser, DEVICE, epoch=None, global_rank=global_rank)
        return

    print("Starting training...")
    losses = []
    if global_rank == 0:
        epochs = tqdm(range(args.epochs), desc="Training Epochs")
    else:
        epochs = range(args.epochs)

    for epoch in epochs:
        epoch_losses = train_one_epoch(args, dataloader, mae, denoiser, optimizer, DEVICE, global_rank=global_rank)
        losses.extend(epoch_losses)
        if global_rank == 0:
            epochs.set_description(f"Epoch {epoch+1}/{args.epochs}, Loss: {np.mean(epoch_losses):.4f}")

            save_plot(losses, filename="loss.png", path=args.output_dir)

        if int(epoch) % args.online_eval_freq == 0 and global_rank == 0:
            print("Starting online evaluation...")
            evaluate(args, mae, denoiser, DEVICE, epoch=epoch, global_rank=global_rank)

    if global_rank == 0:
        save_checkpoint(
            path=f"{args.output_dir}/checkpoint.pt",
            mae=mae,
            denoiser=denoiser,
            optimizer=optimizer
        )
        
if __name__ == "__main__":
    main()
    









