import argparse
import datetime
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch.distributed as dist
import copy

from denoiser import Denoiser
from model.denoising_model_context import DenoisingMLP
from model.mae import MAE
from utils import ddp
from utils.utils import center_crop_arr, save_checkpoint, save_plot, write_csv
from train_eval import evaluate, train_one_epoch
from utils.wandb_utils import initialize_wandb


def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--use_logging", action="store_true", help="Enable debug mode")
    parser.add_argument("--img_size", type=int, default=64, help="Image size (assumed square)")
    parser.add_argument("--output_dir", type=str, default="./output/debug", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--pred_type", type=str, default="v", help="Prediction type for the diffusion", choices=["x", "eps", "v"])
    parser.add_argument("--load_check", action="store_true", help="Load model from checkpoint before training")
    parser.add_argument("--checkpoint_path", type=str, default="./output/checkpoint_last.pt", help="Loading path for checkpoint")
    parser.add_argument("--t_eps", type=float, default=1e-2, help="Epsilon value for time sampling in diffusion")
    parser.add_argument("--remove_ema", action="store_true", help="Use exponential moving average for the model parameters")
    parser.add_argument('--ema_decay', default=0.9999, type=float, help='EMA decay for the parameter update')
    parser.add_argument("--model", type=str, default="pixelMAR-B-04", help="Specify the model")

    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=50, help="Number of warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--min_mask_rate", type=float, default=0.7, help="Minimum mask rate")
    parser.add_argument("--noise_scale", type=float, default=1.0, help="Noise scale for diffusion")
    parser.add_argument('--P_mean', default=-0.8, type=float, help='mean for the normal distribution for time sampling')
    parser.add_argument('--P_std', default=0.8, type=float, help='std for the normal distribution for time sampling')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='Weight decay for the optimizer')
    parser.add_argument("--save_freq", type=int, default=50, help="Frequency of saving the checkpoint")
    parser.add_argument("--diffusion_batch_mult", type=int, default=4, help="Multiplicate the number of diffusion steps for each train step")

    # Sampling 
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Batch size for sampling")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--num_images", type=int, default=50000, help="Number of images to generate during evaluation")
    parser.add_argument("--num_ar_steps", type=int, default=16, help="Number of sampling steps during evaluation")
    parser.add_argument("--num_timesteps", type=int, default=50, help="Number of timesteps for diffusion process during sampling")
    parser.add_argument("--online_eval_freq", type=int, default=25, help="Frequency of online evaluation during training")
    parser.add_argument("--sampling_method", type=str, default="euler", help="Sampling method to use", choices=["euler", "heun"])

    # Dataset
    parser.add_argument("--class_num", type=int, default=1000, help="Number of classes in the dataset")
    parser.add_argument("--has_fixed_target_class", action="store_true", help="Whether to use a fixed target class for training")
    parser.add_argument("--fixed_target_class", type=int, default=0, help="Fix the target class to train on")
    parser.add_argument("--channels", type=int, default=3, help="Number of image channels")
    parser.add_argument("--data_path", type=str, default="./data/imagenet", help="Path to the imagenet train dataset")
    parser.add_argument("--fid_statistics", action="store_true", help="Use online FID calculation")
    parser.add_argument("--fid_statistics_path", type=str, default="./fid_stats/adm_in256_stats_full.npz", help="Path to fid statistic file")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_entity", type=str, default="tuebingen_diffusion")
    parser.add_argument("--wandb_project", type=str, default="tuebingen_diffusion")

    return parser

def main(): 
    args = create_parser().parse_args()

    ddp.init_distributed_mode(args)
    print('Work directory:', os.path.dirname(os.path.realpath(__file__)))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_rank = dist.get_rank() if dist.is_initialized() else 0

    if args.use_wandb and global_rank == 0 and not args.evaluate:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{args.model}_{args.img_size}px_{args.pred_type}-prediction_{now}"   
        initialize_wandb(args, 
                        entity=args.wandb_entity, 
                        exp_name=exp_name,
                        project_name=args.wandb_project
        )

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), 
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)

    if args.has_fixed_target_class:
        targets = torch.tensor(dataset.targets)
        indices = (targets == args.fixed_target_class).nonzero(as_tuple=True)[0]
        dataset = Subset(dataset, indices)
        if len(dataset) == 0:
            print("Fixed dataset class not found!")
            return
         
        print(f"Filtered dataset to class {args.fixed_target_class}. New size: {len(dataset)}")
    
    sampler_train = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_train, 
                            num_workers=10, pin_memory=True)

    model_builder = pixelMAR_models[args.model]
    mae_model, denoising_mlp_model = model_builder(
        img_size=args.img_size, 
        channels=args.channels, 
        num_classes=args.class_num, 
        ema_decay=args.ema_decay
    )

    mae = mae_model.to(device)
    denoiser = Denoiser(args, denoising_mlp_model).to(device)

    mae = torch.nn.parallel.DistributedDataParallel(mae, device_ids=[args.gpu], find_unused_parameters=True)
    denoiser = torch.nn.parallel.DistributedDataParallel(denoiser, device_ids=[args.gpu], find_unused_parameters=True)
    mae_single = mae.module
    denoiser_single = denoiser.module
    
    optimizer = optim.AdamW(
        list(mae.parameters()) + list(denoiser.parameters()), 
        lr=args.lr, 
        weight_decay=args.weight_decay, 
        betas=(0.9, 0.95)
    )

    if args.load_check or args.evaluate:
        print("Loading from checkpoint...")
        path = args.checkpoint_path
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        mae_single.load_state_dict(checkpoint['mae_state_dict'])
        denoiser_single.load_state_dict(checkpoint['denoising_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        mae_single.ema_params = checkpoint['ema_params_mae']
        denoiser_single.ema_params = checkpoint['ema_params_denoising_mlp']
    else:
        mae_single.ema_params = copy.deepcopy(list(mae.parameters()))
        denoiser_single.ema_params = copy.deepcopy(list(denoiser.parameters()))

    metrics = {k: [] for k in ['loss', 'fid', 'is', 'precision', 'recall']}
    if args.evaluate:
        print("Starting sampling...")
        evaluate(args, mae_single, denoiser_single, device, epoch=None, metrics=metrics)
        return

    print("Starting training...")
    for epoch in range(args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        epoch_losses = train_one_epoch(args, epoch, dataloader, mae, denoiser, mae_single, denoiser_single, 
                                       optimizer, device, global_rank=global_rank)
        metrics['loss'].append(sum(epoch_losses) / len(epoch_losses))

        if global_rank == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {np.mean(epoch_losses):.4f}")

        if int(epoch) % args.online_eval_freq == 0:
            print("Starting online evaluation...")
            evaluate(args, mae_single, denoiser_single, device, epoch=epoch, metrics=metrics)

        if global_rank == 0:
            if int(epoch) % args.save_freq == 0:
                print("Saving online checkpoint...")
                #save_path = os.path.join(args.output_dir, "checkpoint_{}.pt".format(epoch))
                save_path = os.path.join(args.output_dir, "checkpoint_last.pt")
                save_checkpoint(
                    path=save_path,
                    mae=mae_single,
                    denoiser=denoiser_single,
                    optimizer=optimizer,
                    mae_ema_params=mae_single.ema_params,
                    denoiser_ema_params=denoiser_single.ema_params
                )

            metrics_folder = os.path.join(args.output_dir, "quality_metrics")
            os.makedirs(metrics_folder, exist_ok=True)
            save_plot(metrics['fid'], "fid.png", metrics_folder, y_label="FID")
            save_plot(metrics['is'], "is.png", metrics_folder, y_label="IS")

            write_csv("fid.csv", metrics_folder, metrics['fid'])
            write_csv("is.csv", metrics_folder, metrics['is'])

    if global_rank == 0:
        print("Saving final checkpoint...")
        save_checkpoint(
            path=f"{args.output_dir}/checkpoint.pt",
            mae=mae_single,
            denoiser=denoiser_single,
            optimizer=optimizer,
            mae_ema_params=mae_single.ema_params,
            denoiser_ema_params=denoiser_single.ema_params
        )

def pixel_mar_S_04(**kwargs):
    patch_size = 4
    mae = MAE(hidden_dim=512, depth=6, bottleneck_dim=512, mlp_ratio=4.0, patch_size=patch_size, **kwargs)
    denoisingMLP = DenoisingMLP(hidden_dim=768, depth=6, final_bottleneck_dim=768, patch_size=patch_size, 
                                mae_hidden_dim=512, **kwargs)
    return mae, denoisingMLP

def pixel_mar_B_04(**kwargs):
    patch_size = 4
    mae = MAE(hidden_dim=768, depth=12, bottleneck_dim=768, mlp_ratio=4.0, patch_size=patch_size, **kwargs)
    denoisingMLP = DenoisingMLP(hidden_dim=1024, depth=6, final_bottleneck_dim=1024, patch_size=patch_size, 
                                mae_hidden_dim=768, **kwargs)
    return mae, denoisingMLP

def pixel_mar_B_16(**kwargs):
    patch_size = 16
    mae = MAE(hidden_dim=768, depth=12, bottleneck_dim=768, mlp_ratio=4.0, patch_size=patch_size, **kwargs)
    denoisingMLP = DenoisingMLP(hidden_dim=1024, depth=6, final_bottleneck_dim=1024, patch_size=patch_size, 
                                mae_hidden_dim=768, **kwargs)
    return mae, denoisingMLP

pixelMAR_models = {
    'pixelMAR-S-04' : pixel_mar_S_04,
    'pixelMAR-B-04' : pixel_mar_B_04,
    'pixelMAR-B-16' : pixel_mar_B_16
}

if __name__ == "__main__":
    main()
    









