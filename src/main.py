import argparse
import datetime
import os
import numpy as np
import yaml
from model.ar_encoder_decoder import ArEncoderDecoder
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset, DistributedSampler
from torchvision import datasets, transforms
import copy

from model.multi_step_flow.denoiser import Denoiser
from model.multi_step_flow.denoising_model import DenoisingModel
from model.one_step_flow.mean_flow import MeanFlow
from model.one_step_flow.mean_flow_model import MeanFlowModel
from model.ar_decoder import ArDecoder
from utils import ddp
from utils.configs_utils import parse_configs
from utils.logging_utils import log_model_parameters
from utils.utils import center_crop_arr, ImageNetLMDB, SingleImageDataset
from train_eval import calc_val_loss, evaluate, train_one_epoch
from utils.wandb_utils import initialize_wandb


def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--config", type=str, default="./configs/ImageNet64/pixelMAR_base_64.yaml", help="Specify the config")
    parser.add_argument("--use_logging", action="store_true", help="Enable debug mode")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--load_check", action="store_true", help="Load model from checkpoint before training")
    parser.add_argument("--checkpoint_path", type=str, default="./output/checkpoint_last.pt", help="Loading path for checkpoint")
    parser.add_argument("--start_epoch", type=int, default=0, help="Start epoch from checkpoint")
    parser.add_argument("--denoiser_type", type=str, default="ada_ln", help="Type of the denoiser", choices=["ada_ln", "ada_ln_fusion", "ada_ln_fusion_only", "cross_attn"])
    parser.add_argument("--log_parameters", action="store_true", help="Log the model parameters")

    # Training
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=1000, help="Training epochs")
    parser.add_argument("--warmup_epochs", type=int, default=100, help="Number of warmup epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (absolute lr)")
    parser.add_argument("--blr", type=float, default=1e-4, help="Base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument("--min_lr", type=float, default=0.0, help="Lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument("--lr_schedule", type=str, default="constant", choices=["constant", "cosine"], help="Learning rate schedule after warmup")
    parser.add_argument('--weight_decay', default=0.02, type=float, help='Weight decay for the optimizer')
    parser.add_argument("--grad_clip", type=float, default=3.0, help="Max grad norm for clip_grad_norm_ (0 disables clipping)")
    parser.add_argument("--grad_log_freq", type=int, default=0, help="Step interval for per-layer grad norms")
    parser.add_argument("--save_freq", type=int, default=5, help="Frequency of saving the checkpoint")
    parser.add_argument("--label_drop_prob", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--overfit_single_image", action="store_true", help="Train on a single image to validate the pipeline")

    # Auxiliary losses
    parser.add_argument("--recon_weight", type=float, default=0, help="Weight for MAR reconstruction loss (0 = disabled)")

    # Sampling
    parser.add_argument("--gen_batch_size", type=int, default=16, help="Batch size for sampling")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model after training")
    parser.add_argument("--num_images", type=int, default=50000, help="Number of images to generate during evaluation")
    parser.add_argument("--online_eval_freq", type=int, default=25, help="Frequency of online evaluation during training")
    parser.add_argument("--remove_ema", action="store_true", help="Use exponential moving average for the model parameters")
    parser.add_argument("--cfg_scale", type=float, default=1.0, help="Classifier-free guidance scale (1.0 = no guidance)")
    parser.add_argument("--cfg_interval_min", type=float, default=0.0, help="CFG interval lower bound (denoiser timestep)")
    parser.add_argument("--cfg_interval_max", type=float, default=1.0, help="CFG interval upper bound (denoiser timestep)")
    parser.add_argument("--ema_eval_decay", type=float, default=None, help="EMA decay value to use for evaluation (must match one of ema_decays in config)")

    # Dataset
    parser.add_argument("--class_num", type=int, default=1000, help="Number of classes in the dataset")
    parser.add_argument("--has_fixed_target_class", action="store_true", help="Whether to use a fixed target class for training")
    parser.add_argument("--fixed_target_class", type=int, default=0, help="Fix the target class to train on")
    parser.add_argument("--data_path", type=str, default="./data/imagenet", help="Path to the imagenet train dataset")
    parser.add_argument("--dataset_type", type=str, default="imagefolder", help="Dataset format", choices=["lmdb", "imagefolder"])
    parser.add_argument("--fid_statistics", action="store_true", help="Use online FID calculation")
    parser.add_argument("--fid_statistics_path", type=str, default="./fid_stats/adm_in256_stats_full.npz", help="Path to fid statistic file")
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument("--wandb_entity", type=str, default="tuebingen_diffusion")
    parser.add_argument("--wandb_project", type=str, default="flow_mar")

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

    model_config, sampler_config = parse_configs(args.config)
    model_params = model_config.get('params', None)
    ar_params = model_config.get('ar_params', None)
    denoiser_params = model_config.get('denoiser_params', None)
    rae_params = model_config.get('rae_params', None)

    model_name = model_config.get('name', 'None')
    dataset_name = model_config.get('dataset_name', 'ImageNet')
    args.img_size = model_params.get('img_size', 256)
    args.patch_size = model_params.get('patch_size', 16)
    args.channels = model_params.get('channels', 3)

    # The last folder of output_dir always matches the wandb run name so the
    # two stay in sync and nothing collides when the parent dir is shared.
    if global_rank == 0:
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        exp_name = f"{model_name}_{args.img_size}px_{now}"
    else:
        exp_name = None
    if dist.is_initialized():
        obj_list = [exp_name]
        dist.broadcast_object_list(obj_list, src=0)
        exp_name = obj_list[0]
    args.output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(args.output_dir, exist_ok=True)

    if global_rank == 0 and not args.evaluate:
        with open(os.path.join(args.output_dir, "args.yaml"), "w") as f:
            yaml.safe_dump(vars(args), f, sort_keys=True)
        with open(os.path.join(args.output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump({"model": model_config, "sampler": sampler_config}, f, sort_keys=False)

    if args.use_wandb and global_rank == 0 and not args.evaluate:
        initialize_wandb(args,
                        entity=args.wandb_entity,
                        exp_name=exp_name,
                        project_name=args.wandb_project,
                        model_config=model_config,
                        sampler_config=sampler_config
        )

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    if args.dataset_type == 'lmdb':
        dataset = ImageNetLMDB(os.path.join(args.data_path, 'imagenet_train.lmdb'), transform=transform)
        val_dataset = ImageNetLMDB(os.path.join(args.data_path, 'imagenet_val.lmdb'), transform=transform)
    elif args.dataset_type == 'imagefolder':
        dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)
        val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform)

    if args.has_fixed_target_class:
        if hasattr(dataset, 'targets'):
            targets = torch.tensor(dataset.targets)
            indices = (targets == args.fixed_target_class).nonzero(as_tuple=True)[0]
            dataset = Subset(dataset, indices)

            val_targets = torch.tensor(val_dataset.targets)
            val_indices = (val_targets == args.fixed_target_class).nonzero(as_tuple=True)[0]
            val_dataset = Subset(val_dataset, val_indices)
        else:
            print(f"Warning: dataset type '{args.dataset_type}' does not support fixed target class filtering, skipping.")

        if len(dataset) == 0:
            print("Fixed dataset class not found!")
            return
        print(f"Filtered dataset to class {args.fixed_target_class}. New size: {len(dataset)}")

    if args.overfit_single_image:
        img_path = "data/single_image.JPEG"
        dataset = SingleImageDataset(img_path, args.img_size, length=max(args.batch_size * world_size * 8, 512))
        val_dataset = SingleImageDataset(img_path, args.img_size, length=max(args.batch_size * world_size, 64))
        args.has_fixed_target_class = True
        args.fixed_target_class = 0
        args.label_drop_prob = 0.0
        print(f"Overfitting on single image: {img_path}")

    sampler_train = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True
    )

    sampler_val = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=global_rank, shuffle=False
    )

    num_workers = 8
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler_train,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True)

    dataloader_val = DataLoader(val_dataset, batch_size=args.batch_size, sampler=sampler_val,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True
    )
    
    # RAE tokenizer for latent space mode
    rae_tokenizer = None
    latent_dim = None
    if rae_params is not None:
        latent_type = rae_params.get('latent_type', 'dinov2')
        if latent_type == 'dinov2':
            from rae.dinov2_tokenizer import Dinov2Tokenizer
            rae_tokenizer = Dinov2Tokenizer(
                dinov2_path=rae_params.get('rae_encoder', 'facebook/dinov2-with-registers-base'),
                rae_decoder_config_path=rae_params.get('rae_decoder_config', None),
                rae_decoder_ckp=rae_params.get('rae_decoder_ckp', None),
                rae_norm_stats=rae_params.get('rae_norm_stats', None),
                encoder_input_size=rae_params.get('encoder_input_size', 224),
                decoder_patch_size=args.patch_size,
            ).to(device)
        elif latent_type == 'mae':
            from rae.mae_tokenizer import MaeTokenizer
            rae_tokenizer = MaeTokenizer(
                mae_path=rae_params.get('rae_encoder', 'facebook/vit-mae-base'),
                rae_decoder_config_path=rae_params.get('rae_decoder_config', None),
                rae_decoder_ckp=rae_params.get('rae_decoder_ckp', None),
                rae_norm_stats=rae_params.get('rae_norm_stats', None),
                encoder_input_size=rae_params.get('encoder_input_size', 256),
                decoder_patch_size=args.patch_size,
            ).to(device)
        elif latent_type == 'vae':
            from rae.vae_tokenizer import VaeTokenizer
            rae_tokenizer = VaeTokenizer(
                vae_path=rae_params.get('vae_path', 'rae_models/vae/kl16.ckpt'),
                embed_dim=rae_params.get('latent_dim', 16),
                latent_scale=rae_params.get('latent_scale', 0.2325),
                img_size=args.img_size,
            ).to(device)
        else:
            raise ValueError(f"Unknown latent_type: '{latent_type}'.")
        rae_tokenizer.eval()
        latent_dim = rae_tokenizer.latent_dim
        print(f"Latent space mode: RAE dim={latent_dim}, patches={rae_tokenizer.num_patches}")

    ema_decay = model_params.get('ema_decay', [0.9999])
    if not isinstance(ema_decay, list):
        ema_decay = [ema_decay]

    ar_type = ar_params.get('ar_type', 'decoder')
    ar_common_kwargs = dict(
        img_size=args.img_size,
        patch_size=args.patch_size,
        channels=args.channels,
        ema_decay=ema_decay,
        decoder_dim=ar_params.get('decoder_dim', 768),
        decoder_depth=ar_params.get('decoder_depth', 12),
        decoder_num_heads=ar_params.get('decoder_num_heads', 12),
        mlp_ratio=ar_params.get('mlp_ratio', 4.0),
        dropout=ar_params.get('dropout', 0.1),
        class_token_size=ar_params.get('class_token_size', 64),
        mask_rate_token_size=ar_params.get('mask_rate_token_size', 8),
        min_mask_rate=ar_params.get('min_mask_rate', 0.7),
        
        train_mask_schedule=ar_params.get('train_mask_schedule', 'truncnorm'),
        gt_noise_scale=ar_params.get('gt_noise_scale', 0.0),
        num_classes=args.class_num,
        lable_dropout=args.label_drop_prob,
        latent_dim=rae_params.get('latent_dim', None) if rae_params else None,
        mask_condition=ar_params.get('mask_condition', False),
    )
    if ar_type == 'encoder_decoder':
        ar_model = ArEncoderDecoder(
            **ar_common_kwargs,
            encoder_dim=ar_params.get('encoder_dim', 768),
            encoder_depth=ar_params.get('encoder_depth', 12),
            encoder_num_heads=ar_params.get('encoder_num_heads', 12),
        ).to(device)
    elif ar_type == 'decoder':
        ar_model = ArDecoder(**ar_common_kwargs).to(device)
    else:
        raise ValueError(f"Unknown ar_type: '{ar_type}'.")

    flow_type = model_params.get('flow_type', 'multi_step')
    if flow_type == 'multi_step':
        # Load denoising model from config file
        denoising_model = DenoisingModel(
            img_size=args.img_size,
            patch_size=args.patch_size,
            channels=args.channels,
            hidden_dim=denoiser_params.get('hidden_dim', 1024),
            depth=denoiser_params.get('depth', 6),
            hidden_ratio=denoiser_params.get('hidden_ratio', 4.0),
            dropout=denoiser_params.get('dropout', 0.1),
            z_hidden_dim=ar_params.get('decoder_dim', 768),
            num_classes=args.class_num,
            denoiser_type=args.denoiser_type,
            bottleneck_dim=model_params.get('bottleneck_dim', None),
            latent_dim=latent_dim
        )
        #Load denoiser from config file
        denoiser = Denoiser(
            denoising_model=denoising_model,
            output_dir=args.output_dir,
            sampling_method=sampler_config.get('method', 'euler'),
            pred_type=model_params.get('pred_type', 'v'),
            diffusion_batch_mul=model_params.get('diffusion_batch_mul', 1),
            num_timesteps=sampler_config.get('num_timesteps', 100),
            sample_t_mean=model_params.get('sample_t_mean', 0.0),
            sample_t_std=model_params.get('sample_t_std', 1.0),
            t_eps=model_params.get('t_eps', 1e-2),
            t_eps_sample=sampler_config.get('t_eps_sample', 1e-2),
            noise_scale=sampler_config.get('noise_scale', 1.0),
            ema_decay=ema_decay,
            use_logging=args.use_logging,
            train_shift=model_params.get('train_shift', None),
            sample_shift=model_params.get('sample_shift', None),
            t_sampling_method=model_params.get('t_sampling_method', 'logit_normal')
        ).to(device)
    elif flow_type == 'one_step':
        denoising_model = MeanFlowModel(
            img_size=args.img_size,
            patch_size=args.patch_size,
            channels=args.channels,
            hidden_dim=denoiser_params.get('hidden_dim', 1024),
            depth=denoiser_params.get('depth', 6),
            aux_head_depth=denoiser_params.get('aux_head_depth', 2),
            hidden_ratio=denoiser_params.get('hidden_ratio', 4.0),
            dropout=denoiser_params.get('dropout', 0.0),
            z_hidden_dim=ar_params.get('decoder_dim', 768),
            bottleneck_dim=model_params.get('bottleneck_dim', None),
            latent_dim=latent_dim,
            pred_type=model_params.get('pred_type', 'x'),
            t_eps=model_params.get('t_eps', 5e-2),
        )
        denoiser = MeanFlow(
            denoising_model=denoising_model,
            output_dir=args.output_dir,
            sampling_method=sampler_config.get('method', 'euler'),
            diffusion_batch_mul=model_params.get('diffusion_batch_mul', 1),
            num_timesteps=sampler_config.get('num_timesteps', 1),
            sample_t_mean=model_params.get('sample_t_mean', 0.0),
            sample_t_std=model_params.get('sample_t_std', 1.0),
            t_eps=model_params.get('t_eps', 5e-2),
            t_eps_sample=sampler_config.get('t_eps_sample', 0.0),
            noise_scale=sampler_config.get('noise_scale', 1.0),
            ema_decay=ema_decay,
            use_logging=args.use_logging,
            train_shift=model_params.get('train_shift', None),
            sample_shift=model_params.get('sample_shift', None),
            data_proportion=model_params.get('data_proportion', 0.5),
            norm_p=model_params.get('norm_p', 1.0),
            norm_eps=model_params.get('norm_eps', 1e-2),
            t_sampling_method=model_params.get('t_sampling_method', 'logit_normal'),
        ).to(device)

    if global_rank == 0 and args.log_parameters:
        log_model_parameters(ar_model, denoiser, device, args)
        torch.cuda.empty_cache()

    ar_model = torch.nn.parallel.DistributedDataParallel(ar_model, device_ids=[args.gpu], find_unused_parameters=True)
    denoiser = torch.nn.parallel.DistributedDataParallel(denoiser, device_ids=[args.gpu], find_unused_parameters=False)
    ar_model_single = ar_model.module
    denoiser_single = denoiser.module
    
    # Scale learning rate based on effective batch size (linear scaling rule)
    eff_batch_size = args.batch_size * world_size
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256
    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)
    print("effective batch size: %d" % eff_batch_size)

    opt_params = list(ar_model.parameters()) + list(denoiser.parameters())

    optimizer = optim.AdamW(
        opt_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )

    if args.load_check or args.evaluate:
        print("Loading from checkpoint...")
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)

        if 'ar_model' in checkpoint:
            missing = ar_model_single.load_state_dict(checkpoint['ar_model'], strict=False).missing_keys
            ema_ar_raw = checkpoint['ema_ar_model']
        else:
            missing = ar_model_single.load_state_dict(checkpoint['mae'], strict=False).missing_keys
            ema_ar_raw = checkpoint['ema_mae']
        #TODO: Change in the final code. Zero out params missing from old checkpoints so they act as no-ops
        for key in missing:
            print(f"  Zeroing missing key: {key}")
            nn.init.zeros_(dict(ar_model_single.named_parameters())[key])
        denoiser_single.load_state_dict(checkpoint['denoiser'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        ema_den_raw = checkpoint['ema_denoiser']

        # Backward compatibility #TODO Remove this in the future
        def _load_ema(raw, num_emas, device):
            if isinstance(raw[0], torch.Tensor):
                # Old format: single EMA
                single = [p.to(device) for p in raw]
                return [copy.deepcopy(single) for _ in range(num_emas)]
            else:
                # New format: list of EMAs
                loaded = [[p.to(device) for p in ema] for ema in raw]
                while len(loaded) < num_emas:
                    loaded.append(copy.deepcopy(loaded[-1]))
                return loaded[:num_emas]

        ar_model_single.ema_params_list = _load_ema(ema_ar_raw, len(ema_decay), device)
        denoiser_single.ema_params_list = _load_ema(ema_den_raw, len(ema_decay), device)

        args.start_epoch = checkpoint.get('epoch', 0)

        print("Loaded epoch: {}".format(checkpoint.get('epoch', None)))
        print("Total training steps: {}".format(checkpoint.get('step', None)))
        print("Checkpoint args: {} \n".format(checkpoint.get('args', None)).replace(', ', ',\n'))
    else:
        print("{}".format(args).replace(', ', ',\n'))
        ar_model_single.ema_params_list = [copy.deepcopy(list(ar_model.parameters())) for _ in ema_decay]
        denoiser_single.ema_params_list = [copy.deepcopy(list(denoiser.parameters())) for _ in ema_decay]

    metrics = {k: [] for k in ['loss', 'fid', 'is', 'precision', 'recall']}
    if args.evaluate:
        print("Starting sampling...")
        evaluate(args=args, ar_model=ar_model_single, denoiser=denoiser_single, device=device, model_params=model_params, sampler_params=sampler_config, epoch=None, metrics=metrics, rae_tokenizer=rae_tokenizer)
        return

    print("Starting training...")
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)

        global_step = train_one_epoch(args, epoch, dataloader, ar_model, denoiser, ar_model_single, denoiser_single,
                                       optimizer, device, rae_tokenizer=rae_tokenizer)

        calc_val_loss(args, ar_model_single, denoiser_single, dataloader_val, epoch, device, rae_tokenizer=rae_tokenizer)

        is_last_epoch = (epoch == args.epochs - 1)

        if int(epoch) % args.online_eval_freq == 0 or is_last_epoch:
            print("Starting online evaluation...")
            evaluate(args=args, ar_model=ar_model_single, denoiser=denoiser_single, device=device, model_params=model_params, sampler_params=sampler_config, epoch=epoch, metrics=metrics, rae_tokenizer=rae_tokenizer, global_step=global_step)
            print("Online evaluation finsihed")

        if global_rank == 0:
            if int(epoch) % args.save_freq == 0 or is_last_epoch:
                print("Saving online checkpoint...")
                #save_path = os.path.join(args.output_dir, "checkpoint_{}.pt".format(epoch))
                ckpt_path = os.path.join(args.output_dir, "checkpoint_last.pt")

                checkpoint = {
                    "ar_model": ar_model_single.state_dict(),
                    "denoiser": denoiser_single.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "ema_ar_model": ar_model_single.ema_params_list,
                    "ema_denoiser": denoiser_single.ema_params_list,

                    "epoch": epoch,
                    "step": global_step,

                    "config_path": args.config,
                    "model_config": model_config,
                    "sampler_config": sampler_config,
                    "seed": args.seed,
                    "args": vars(args)
                }
                tmp_path = ckpt_path + ".tmp"
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, ckpt_path)

                if int(epoch) % 100 == 0 and int(epoch) > 0:
                    archive_path = os.path.join(args.output_dir, f"checkpoint_epoch{int(epoch)}.pt")
                    torch.save(checkpoint, archive_path)
                    print(f"Archived checkpoint to {archive_path}")

                print("Saving online checkpoint finished")

if __name__ == "__main__":
    main()