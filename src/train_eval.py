from collections import defaultdict
import copy
import os
import random
import shutil
import cv2
from scipy import stats
from torchvision import transforms
import torch
import numpy as np
from sample import sample
import torch_fidelity
from utils.wandb_utils import log

from utils.utils import adjust_learning_rate, patchify, sample_order, save_img_as_fig, unpatchify

def random_masking(x, orders, min_mask_rate=0.7):
    bsz, seq_len, embed_dim = x.shape
    mask_rate = min_mask_rate
    mask_rate = stats.truncnorm((min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
    num_masked_tokens = int(np.ceil(seq_len * mask_rate))
    mask = torch.zeros(bsz, seq_len, device=x.device)
    mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
    return mask

def train_one_epoch(args, epoch, dataloader, mae, denoiser, mae_single, denoiser_single, optimizer, device):
    mae.train()
    denoiser.train()    

    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    
    for step, (samples, labels) in enumerate(dataloader):
        adjust_learning_rate(optimizer, step / len(dataloader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x = patchify(samples, mae_single.patch_size)
        x_gt = x.clone().detach()
        orders = sample_order(x.shape[0], x.shape[1], device)
        mask = random_masking(x, orders, mae_single.min_mask_rate)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            z = mae(x, mask, labels)
            loss = denoiser(x_gt, z, mask, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        mae_single.update_ema()
        denoiser_single.update_ema()


        optimizer_step = step + epoch*len(dataloader)
        if local_rank == 0:
            if args.use_wandb:
                stats = {
                    "train/loss": loss.item(),
                    "train/lr": optimizer.param_groups[0]['lr'],
                }
                log(stats, step=optimizer_step)
        
    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {np.mean(loss.item()):.4f}")
    return optimizer_step

def evaluate(args, mae, denoiser, device, model_params, sampler_params, epoch=None, metrics=None):
    mae.eval()
    denoiser.eval()

    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    if epoch is not None:
        save_folder_fid = os.path.join(args.output_dir, "train_fid_samples")
        class_folder = os.path.join(args.output_dir, "train_class_samples", 'epoch{}'.format(epoch))
    else:
        save_folder_fid = os.path.join(args.output_dir, "eval_fid_samples")
        class_folder = os.path.join(args.output_dir, "eval_class_samples")

    if local_rank == 0: 
        os.makedirs(class_folder, exist_ok=True)
        os.makedirs(save_folder_fid, exist_ok=True)

    if args.remove_ema == False: 
        mae_state_dict = copy.deepcopy(mae.state_dict())
        mae_ema_state_dict = copy.deepcopy(mae.state_dict())
        for i, (name, _value) in enumerate(mae.named_parameters()):
            assert name in mae_ema_state_dict
            mae_ema_state_dict[name] = mae.ema_params[i]

        denoiser_state_dict = copy.deepcopy(denoiser.state_dict())
        denoiser_ema_state_dict = copy.deepcopy(denoiser.state_dict())
        for i, (name, _value) in enumerate(denoiser.named_parameters()):
            assert name in denoiser_ema_state_dict
            denoiser_ema_state_dict[name] = denoiser.ema_params[i]
        
        print("Switch to ema")
        mae.load_state_dict(mae_ema_state_dict)
        denoiser.load_state_dict(denoiser_ema_state_dict)

    assert args.num_images % args.class_num == 0, "Number of images per class must be the same"
    if args.has_fixed_target_class:
        class_label_gen_world = np.full((args.class_num,), args.fixed_target_class)
    else:
        class_label_gen_world = np.arange(0, args.class_num).repeat(args.num_images // args.class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    num_steps = args.num_images // (args.gen_batch_size * world_size) + 1
    bsz = args.gen_batch_size
    for step in range(num_steps):
        start_idx = world_size * bsz * step + local_rank * bsz
        end_idx = start_idx + bsz
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()
        
        torch.distributed.barrier()

        sampled_images = sample(args, mae, denoiser, labels_gen, device, model_params, sampler_params)
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        for b_id in range(sampled_images.size(0)):
            img_id = step * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
            if img_id >= args.num_images:
                break
            gen_img = np.round(np.clip( sampled_images[b_id].numpy().transpose([1, 2, 0]) * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(os.path.join(save_folder_fid, 'sample_{}.png'.format(str(img_id).zfill(5))), gen_img)

    if local_rank == 0:
        all_ids = list(range(args.num_images))
        sample_size = min(len(all_ids), 20)
        selected_ids = random.sample(all_ids, sample_size)
        
        for i, img_id in enumerate(selected_ids):
            cls = int(class_label_gen_world[img_id])
            src = os.path.join(save_folder_fid, f'sample_{str(img_id).zfill(5)}.png')
            
            class_filename = f'sample_{i}_class_{cls}.png'
            dst = os.path.join(class_folder, class_filename)
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                
    torch.distributed.barrier()

    if args.remove_ema == False:
        print("Switch back from ema")
        mae.load_state_dict(mae_state_dict)
        denoiser.load_state_dict(denoiser_state_dict)

    # Evaluate the generation quality
    if args.fid_statistics:
        isc = True
        fid = True
        kid = False
        prc = False
        if os.path.exists(args.fid_statistics_path) == False:
            print("Statistic path does not exits! ")
            return
        metrics_dict = torch_fidelity.calculate_metrics(
            input1=save_folder_fid,
            input2=None,
            fid_statistics_file=args.fid_statistics_path,
            cuda=True,
            isc=isc,
            fid=fid,
            kid=kid,
            prc=prc,
            verbose=False
        )
        if fid:
            fid_score = metrics_dict['frechet_inception_distance']
            print("FID: {:.4f}".format(fid_score))
            if metrics['fid'] is not None:
                metrics['fid'].append(fid_score)
        if isc:
            inception_score = metrics_dict['inception_score_mean']
            print("Inception Score: {:.4f}".format(inception_score))
            if metrics['is'] is not None:
                metrics['is'].append(inception_score)
        if kid:
            kid_score = metrics_dict['kernel_inception_distance_mean']
            print("KID: {:.4f}".format(kid_score))
        if prc:
            precision = metrics_dict['precision']
            recall = metrics_dict['recall']
            print("Precision: {:.4f}".format(precision))
            print("Recall: {:.4f}".format(recall))      
            if metrics['precision'] is not None:
                metrics['precision'].append(precision)
            if metrics['recall'] is not None:
                metrics['recall'].append(recall)
        
        if local_rank == 0:
            if args.use_wandb and epoch is not None:
                stats = {
                    "evaluate/fid": fid_score,
                    "evaluate/is": inception_score,
                    "epoch" : epoch
                }
                log(stats)
            shutil.rmtree(save_folder_fid)

    torch.distributed.barrier()