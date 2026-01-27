import argparse
import datetime
import os
import numpy as np
import torch
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import datasets, transforms
import torch.distributed as dist
import copy

from denoiser import Denoiser
from model.denoising_model_context import DenoisingModel
from model.mae import MAE
from utils import ddp
from utils.configs_utils import parse_configs
from utils.utils import center_crop_arr, save_plot, write_csv
from train_eval import evaluate, train_one_epoch
from utils.wandb_utils import initialize_wandb


def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--config", type=str, default="./configs/ImageNet64/pixelMAR_base_64.yaml", help="Specify the config")
    parser.add_argument("--use_logging", action="store_true", help="Enable debug mode")
    parser.add_argument("--output_dir", type=str, default="./output/debug", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--load_check", action="store_true", help="Load model from checkpoint before training")
    parser.add_argument("--checkpoint_path", type=str, default="./output/checkpoint_last.pt", help="Loading path for checkpoint")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch from checkpoint")
    parser.add_argument('--grad_checkpointing', action='store_true')

    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=100, help="Number of warmup epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--weight_decay', default=0.02, type=float, help='Weight decay for the optimizer')
    parser.add_argument("--save_freq", type=int, default=50, help="Frequency of saving the checkpoint")

    # Sampling 
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Batch size for sampling")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--num_images", type=int, default=50000, help="Number of images to generate during evaluation")
    parser.add_argument("--online_eval_freq", type=int, default=25, help="Frequency of online evaluation during training")
    parser.add_argument("--remove_ema", action="store_true", help="Use exponential moving average for the model parameters")
    parser.add_argument("--cfg_scale", type=int, default=1.0, help="Classifier-free guidance factor")
    parser.add_argument("--cfg_min", type=int, default=0.1, help="Classifier-free guidance minimum")
    parser.add_argument("--cfg_max", type=int, default=1.0, help="Classifier-free guidance maximum")

    # Dataset
    parser.add_argument("--class_num", type=int, default=1000, help="Number of classes in the dataset")
    parser.add_argument("--has_fixed_target_class", action="store_true", help="Whether to use a fixed target class for training")
    parser.add_argument("--fixed_target_class", type=int, default=0, help="Fix the target class to train on")
    parser.add_argument("--data_path", type=str, default="./data/imagenet", help="Path to the imagenet train dataset")
    parser.add_argument("--fid_statistics", action="store_true", help="Use online FID calculation")
    parser.add_argument("--fid_statistics_path", type=str, default="./fid_stats/adm_in256_stats_full.npz", help="Path to fid statistic file")

    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_entity", type=str, default="tuebingen_diffusion")
    parser.add_argument("--wandb_project", type=str, default="pixel_mar")

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

    os.makedirs(args.output_dir, exist_ok=True)

    model_config, sampler_config = parse_configs(args.config)
    model_params = model_config.get('params', None)
    mae_params = model_config.get('mae_params', None)
    denoiser_params = model_config.get('denoiser_params', None)

    model_name = model_config.get('name', 'None')
    dataset_name = model_config.get('dataset_name', 'ImageNet')
    args.img_size = model_params.get('img_size', 256)
    args.patch_size = model_params.get('patch_size', 16)
    args.channels = model_params.get('channels', 3)

    if args.use_wandb and global_rank == 0 and not args.evaluate:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{model_name}_{dataset_name}_{args.img_size}px_{now}"   
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
    
    sampler_train = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_train, 
                            num_workers=8, pin_memory=True, persistent_workers=True)
    
    # Load MAE from config file
    mae = MAE(
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        ema_decay=model_params.get('ema_decay', 0.9999),
        encoder_dim=mae_params.get('encoder_dim', 768),
        decoder_dim=mae_params.get('decoder_dim', 768),
        encoder_depth=mae_params.get('encoder_depth', 12),
        decoder_depth=mae_params.get('decoder_depth', 12),
        encoder_num_heads=mae_params.get('encoder_num_heads', 12),
        decoder_num_heads=mae_params.get('decoder_num_heads', 12),
        mlp_ratio=mae_params.get('mlp_ratio', 4.0),
        dropout=mae_params.get('dropout', 0.1),
        buffer_size=mae_params.get('buffer_size', 64),
        min_mask_rate=mae_params.get('min_mask_rate', 0.7),
        num_classes=args.class_num,
        grad_ckpt=args.grad_checkpointing

    ).to(device)

    # Load denoising model from config file
    denoising_model = DenoisingModel(
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        hidden_dim=denoiser_params.get('hidden_dim', 1024),
        depth=denoiser_params.get('depth', 6),
        dropout=denoiser_params.get('dropout', 0.1),
        z_hidden_dim=mae_params.get('decoder_dim', 768),
        num_classes=args.class_num,
        grad_ckpt=args.grad_checkpointing
    )
    #Load denoiser from config file
    denoiser = Denoiser(
        denoising_model=denoising_model,
        output_dir=args.output_dir,
        sampling_method=sampler_config.get('method', 'euler'),
        pred_type=model_params.get('pred_type', 'v'),
        diffusion_batch_multi=model_params.get('diffusion_batch_multi', 4),
        num_timesteps=sampler_config.get('num_timesteps', 100),
        sample_t_mean=model_params.get('sample_t_mean', 0.0),
        sample_t_std=model_params.get('sample_t_std', 1.0),
        t_eps=model_params.get('t_eps', 1e-2),
        noise_scale=sampler_config.get('noise_scale', 1.0),
        ema_decay=model_params.get('ema_decay', 0.9999),
        use_logging=args.use_logging
    ).to(device)

    n_mae_params = sum(p.numel() for p in mae.parameters() if p.requires_grad)
    n_denoiser_params = sum(p.numel() for p in denoiser.parameters() if p.requires_grad)
    n_params = n_mae_params + n_denoiser_params
    print("Number of trainable parameters: {:.6f}M".format(n_params / 1e6))

    mae = torch.nn.parallel.DistributedDataParallel(mae, device_ids=[args.gpu], find_unused_parameters=False)
    denoiser = torch.nn.parallel.DistributedDataParallel(denoiser, device_ids=[args.gpu], find_unused_parameters=False)
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
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

        mae_single.load_state_dict(checkpoint['mae'])
        denoiser_single.load_state_dict(checkpoint['denoiser'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        mae_single.ema_params = checkpoint['ema_mae']
        denoiser_single.ema_params = checkpoint['ema_denoiser']

        args.start_epoch = checkpoint.get('epoch', 0)

        if 'model_config' in checkpoint and model_config != checkpoint['model_config']:
            print("Model config loaded from checkpoint is different")
            return
        if 'sampler_config' in checkpoint and sampler_config != checkpoint['sampler_config']:
            print("Sampler config loaded from checkpoint is different")
            return

        print("Loaded epoch: {}".format(checkpoint.get('epoch', None)))
        print("Total training steps: {}".format(checkpoint.get('step', None)))
        print("Checkpoint args: {} \n".format(checkpoint.get('args', None)).replace(', ', ',\n'))
    else:
        print("{}".format(args).replace(', ', ',\n'))
        mae_single.ema_params = copy.deepcopy(list(mae.parameters()))
        denoiser_single.ema_params = copy.deepcopy(list(denoiser.parameters()))

    metrics = {k: [] for k in ['loss', 'fid', 'is', 'precision', 'recall']}
    if args.evaluate:
        print("Starting sampling...")
        evaluate(args=args, mae=mae_single, denoiser=denoiser_single, device=device, model_params=model_params, sampler_params=sampler_config, epoch=None, metrics=metrics)
        return

    print("Starting training...")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        global_step = train_one_epoch(args, epoch, dataloader, mae, denoiser, mae_single, denoiser_single, 
                                       optimizer, device)

        if int(epoch) % args.online_eval_freq == 0 and int(epoch) > 0:
            print("Starting online evaluation...")
            evaluate(args=args, mae=mae_single, denoiser=denoiser_single, device=device, model_params=model_params, sampler_params=sampler_config, epoch=epoch, metrics=metrics)

        if global_rank == 0:
            if int(epoch) % args.save_freq == 0:
                print("Saving online checkpoint...")
                #save_path = os.path.join(args.output_dir, "checkpoint_{}.pt".format(epoch))
                ckpt_path = os.path.join(args.output_dir, "checkpoint_last.pt")

                checkpoint = {
                    "mae": mae_single.state_dict(),
                    "denoiser": denoiser_single.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema_mae": mae_single.ema_params,
                    "ema_denoiser": denoiser_single.ema_params,

                    "epoch": epoch,
                    "step": global_step,

                    "config_path": args.config,
                    "model_config": model_config,
                    "sampler_config": sampler_config,
                    "seed": args.seed, 
                    "args": vars(args)
                }
                torch.save(checkpoint, ckpt_path)

            metrics_folder = os.path.join(args.output_dir, "quality_metrics")
            os.makedirs(metrics_folder, exist_ok=True)
            save_plot(metrics['fid'], "fid.png", metrics_folder, y_label="FID")
            save_plot(metrics['is'], "is.png", metrics_folder, y_label="IS")

            write_csv("fid.csv", metrics_folder, metrics['fid'])
            write_csv("is.csv", metrics_folder, metrics['is'])

if __name__ == "__main__":
    main()
    









