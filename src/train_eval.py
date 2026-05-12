from collections import defaultdict

import json
import os
import random
import shutil
import time
import cv2
from scipy import stats
from torchvision import transforms
import torch
import numpy as np
import wandb
from sample import sample
import torch_fidelity
from utils.logging_utils import StepTimer
from utils.wandb_utils import log

from utils.utils import adjust_learning_rate, patchify, sample_order, save_img_as_fig, save_pca_viz, unpatchify


def _module_grad_norm(parameters):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return torch.tensor(0.0)
    return torch.norm(torch.stack([g.norm(2) for g in grads]), 2)


def train_one_epoch(args, epoch, dataloader, ar_model, denoiser, ar_model_single, denoiser_single, optimizer, device, rae_tokenizer=None):
    ar_model.train()
    denoiser.train()

    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    timer = StepTimer(log_interval=50) if args.log_parameters else None

    for step, batch in enumerate(dataloader):
        adjust_learning_rate(optimizer, step / len(dataloader) + epoch, args)

        if timer:
            timer.mark('step_start')

        samples, labels = batch
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if rae_tokenizer is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_gt = rae_tokenizer.encode(samples)
        else:
            x_gt = patchify(samples, args.patch_size)
        orders = sample_order(x_gt.shape[0], x_gt.shape[1], device)

        if timer:
            timer.mark('ar_model_start')

        recon_weight = getattr(args, 'recon_weight', 0.0)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            z, mask, x_recon = ar_model(x_gt, orders, labels)

        if timer:
            timer.mark('ar_model_end')

        if step == 0 and local_rank == 0 and getattr(args, 'use_logging', False):
            pca_folder = os.path.join(args.output_dir, "images", "pca_viz")
            with torch.no_grad():
                x_masked = x_gt * (1 - mask).unsqueeze(-1)
                pca_idx = int(np.random.randint(x_gt.shape[0]))

                # spatial visible-token mask for the chosen sample, upsampled to image resolution
                n_tok = mask.shape[1]
                h_tok = w_tok = int(n_tok ** 0.5)
                visible = (1 - mask[pca_idx]).reshape(h_tok, w_tok)
                scale = samples.shape[-1] // h_tok
                visible_img = visible.repeat_interleave(scale, 0).repeat_interleave(scale, 1)
                gt_masked = samples[pca_idx] * visible_img.to(samples.dtype)

                def _pca_path(name):
                    sub = os.path.join(pca_folder, name)
                    os.makedirs(sub, exist_ok=True)
                    return os.path.join(sub, f"epoch{epoch:03d}.png")

                save_img_as_fig(samples[pca_idx:pca_idx + 1].float(), _pca_path("gt"),        size=args.img_size)
                save_img_as_fig(gt_masked.unsqueeze(0).float(),       _pca_path("gt_masked"), size=args.img_size)
                save_pca_viz(x_gt.float(),     _pca_path("dinov2"),        args.img_size, idx=pca_idx)
                save_pca_viz(x_masked.float(),  _pca_path("dinov2_masked"), args.img_size, idx=pca_idx)
                save_pca_viz(z.float(),         _pca_path("z_cond"),        args.img_size, idx=pca_idx)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            denoiser_loss = denoiser(x_gt, z, mask, labels)

        loss = denoiser_loss

        recon_loss = None
        if recon_weight > 0:
            B, N, D = x_gt.shape
            recon_loss = ((x_recon - x_gt) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * D + 1e-8)
            loss = loss + recon_weight * recon_loss

        if timer:
            timer.mark('den_end')

        optimizer.zero_grad()
        loss.backward()

        if timer:
            timer.mark('bwd_end')

        optimizer_step = step + epoch * len(dataloader)
        log_grad_details = (
            local_rank == 0
            and args.use_wandb
            and args.grad_log_freq > 0
            and optimizer_step % args.grad_log_freq == 0
        )

        ar_grad_norm = _module_grad_norm(list(ar_model.parameters()))
        den_grad_norm = _module_grad_norm(list(denoiser.parameters()))

        per_layer_norms = {}
        grad_hist_values = None
        if log_grad_details:
            flat_grads = []
            for prefix, model in (("ar_model", ar_model), ("denoiser", denoiser)):
                for name, p in model.named_parameters():
                    if p.grad is None:
                        continue
                    g = p.grad.detach()
                    per_layer_norms[f"grad_norm_layer/{prefix}.{name}"] = g.norm(2).item()
                    flat_grads.append(g.flatten().float().cpu())
            if flat_grads:
                grad_hist_values = torch.cat(flat_grads).numpy()

        opt_params = list(ar_model.parameters()) + list(denoiser.parameters())
        if args.grad_clip > 0:
            total_grad_norm = torch.nn.utils.clip_grad_norm_(opt_params, max_norm=args.grad_clip)
        else:
            total_grad_norm = torch.sqrt(ar_grad_norm ** 2 + den_grad_norm ** 2)

        optimizer.step()

        if timer:
            timer.mark('opt_end')

        ar_model_single.update_ema()
        denoiser_single.update_ema()

        if timer:
            timer.mark('ema_end')
            timer.end_step()

        if local_rank == 0:
            if timer and timer.should_log():
                timing_stats = timer.print_and_get_stats()

            stats = {
                "train/loss": loss.item(),
                "train/denoiser_loss": denoiser_loss.item(),
                "train/lr": optimizer.param_groups[0]['lr'],
                "train/grad_norm": total_grad_norm.item(),
                "train/grad_norm_ar": ar_grad_norm.item(),
                "train/grad_norm_denoiser": den_grad_norm.item(),
            }
            if log_grad_details:
                stats.update(per_layer_norms)
                if grad_hist_values is not None:
                    stats["train/grad_hist"] = wandb.Histogram(grad_hist_values)
            if recon_loss is not None:
                stats["train/recon_loss"] = recon_loss.item()

            # Extra losses for MeanFlow objective
            denoiser_inner = denoiser_single if hasattr(denoiser_single, 'last_loss_u_raw') else None
            if denoiser_inner is not None:
                stats["train/loss_u_raw"] = denoiser_inner.last_loss_u_raw.item()
                stats["train/loss_v_raw"] = denoiser_inner.last_loss_v_raw.item()

            if args.use_wandb:
                log(stats, step=optimizer_step)

    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {np.mean(loss.item()):.4f}")
    return optimizer_step

@torch.no_grad()
def calc_val_loss(args, ar_model, denoiser, val_dataloader, epoch, device, rae_tokenizer=None):
    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    recon_weight = getattr(args, 'recon_weight', 0.0)

    losses = []
    recon_losses = []
    for step, batch in enumerate(val_dataloader):
        samples, labels = batch
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if rae_tokenizer is not None:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                x_gt = rae_tokenizer.encode(samples)
        else:
            x_gt = patchify(samples, args.patch_size)

        orders = sample_order(x_gt.shape[0], x_gt.shape[1], device)

        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            z, mask, x_recon = ar_model(x_gt, orders, labels)
            denoiser_loss = denoiser(x_gt, z, mask, labels)

        losses.append(denoiser_loss.item())

        if recon_weight > 0:
            B, N, D = x_gt.shape
            rl = ((x_recon - x_gt) ** 2 * mask.unsqueeze(-1)).sum() / (mask.sum() * D + 1e-8)
            recon_losses.append(rl.item())

    if local_rank == 0:
        if args.use_wandb:
            stats = {
                "evaluation/val_loss": sum(losses) / len(losses),
                "epoch" : epoch
            }
            if recon_losses:
                stats["evaluation/val_recon_loss"] = sum(recon_losses) / len(recon_losses)
            log(stats)

    torch.distributed.barrier()

def evaluate(args, ar_model, denoiser, device, model_params, sampler_params, epoch=None, metrics=None, rae_tokenizer=None, global_step=None):
    ar_model.eval()
    denoiser.eval()

    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    images_dir = os.path.join(args.output_dir, "images")

    cfg_scale_main = getattr(args, 'cfg_scale', 1.0)
    eval_cfg_scales = [1.0, cfg_scale_main] if cfg_scale_main != 1.0 else [cfg_scale_main]

    if args.remove_ema == False:
        ema_eval_decay = getattr(args, 'ema_eval_decay', None)
        if ema_eval_decay is not None:
            ema_idx = ar_model.ema_decays.index(ema_eval_decay)
        else:
            ema_idx = len(ar_model.ema_decays) - 1
        print(f"Switch to ema (decay={ar_model.ema_decays[ema_idx]})")
        for i, p in enumerate(ar_model.parameters()):
            ar_model.ema_params_list[ema_idx][i], p.data = p.data.clone(), ar_model.ema_params_list[ema_idx][i]
        for i, p in enumerate(denoiser.parameters()):
            denoiser.ema_params_list[ema_idx][i], p.data = p.data.clone(), denoiser.ema_params_list[ema_idx][i]

    assert args.num_images % args.class_num == 0, "Number of images per class must be the same"
    if args.has_fixed_target_class:
        class_label_gen_world = np.full((args.class_num,), args.fixed_target_class)
    else:
        class_label_gen_world = np.arange(0, args.class_num).repeat(args.num_images // args.class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    num_steps = args.num_images // (args.gen_batch_size * world_size) + 1
    bsz = args.gen_batch_size

    # Map each cfg scale to its fid folder so metrics can be computed after EMA switch-back
    fid_folders = {}

    for eval_cfg in eval_cfg_scales:
        cfg_tag = f"cfg{eval_cfg:.2g}"

        if epoch is not None:
            save_folder_fid = os.path.join(images_dir, f"train_fid_samples_{cfg_tag}")
            class_folder = os.path.join(images_dir, "train_class_samples", f"epoch{epoch}_{cfg_tag}")
        else:
            save_folder_fid = os.path.join(images_dir, f"eval_fid_samples_{cfg_tag}")
            class_folder = os.path.join(images_dir, f"eval_class_samples_{cfg_tag}")

        fid_folders[eval_cfg] = save_folder_fid

        if local_rank == 0:
            os.makedirs(class_folder, exist_ok=True)
            os.makedirs(save_folder_fid, exist_ok=True)

        args.cfg_scale = eval_cfg

        used_time = 0
        gen_img_cnt = 0
        for step in range(num_steps):
            seed = args.seed + step
            torch.manual_seed(seed)
            np.random.seed(seed)

            start_idx = world_size * bsz * step + local_rank * bsz
            end_idx = start_idx + bsz
            labels_gen = class_label_gen_world[start_idx:end_idx]
            labels_gen = torch.Tensor(labels_gen).long().cuda()

            torch.distributed.barrier()

            torch.cuda.synchronize()
            start_time = time.time()

            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                sampled_images = sample(args, ar_model, denoiser, labels_gen, device, model_params, sampler_params, rae_tokenizer=rae_tokenizer)

            if step >= 1:
                torch.cuda.synchronize()
                used_time += time.time() - start_time
                gen_img_cnt += bsz
                print(f"[{cfg_tag}] Generating {gen_img_cnt} images takes {used_time:.5f} seconds, {used_time/gen_img_cnt:.5f} sec per image")

            torch.distributed.barrier()

            sampled_images = sampled_images.detach().cpu()
            sampled_images = (sampled_images + 1) / 2

            for b_id in range(sampled_images.size(0)):
                img_id = step * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                if img_id >= args.num_images:
                    break
                gen_img = np.round(np.clip(sampled_images[b_id].float().numpy().transpose([1, 2, 0]) * 255, 0, 255))
                gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(os.path.join(save_folder_fid, f'sample_{str(img_id).zfill(5)}.png'), gen_img)

        args.cfg_scale = cfg_scale_main  # restore

        if local_rank == 0:
            all_ids = list(range(args.num_images))
            selected_ids = random.sample(all_ids, min(len(all_ids), 20))
            for i, img_id in enumerate(selected_ids):
                cls = int(class_label_gen_world[img_id])
                src = os.path.join(save_folder_fid, f'sample_{str(img_id).zfill(5)}.png')
                dst = os.path.join(class_folder, f'sample_{i}_class_{cls}.png')
                if os.path.exists(src):
                    shutil.copy2(src, dst)

        torch.distributed.barrier()

    if args.remove_ema == False:
        print("Switch back from ema")
        for i, p in enumerate(ar_model.parameters()):
            ar_model.ema_params_list[ema_idx][i], p.data = p.data.clone(), ar_model.ema_params_list[ema_idx][i]
        for i, p in enumerate(denoiser.parameters()):
            denoiser.ema_params_list[ema_idx][i], p.data = p.data.clone(), denoiser.ema_params_list[ema_idx][i]
        torch.cuda.empty_cache()

    # Compute metrics for each cfg scale
    if args.fid_statistics:
        if not os.path.exists(args.fid_statistics_path):
            print("Statistic path does not exits!")
            torch.distributed.barrier()
            return

        for eval_cfg in eval_cfg_scales:
            cfg_tag = f"cfg{eval_cfg:.2g}"
            is_main = (eval_cfg == eval_cfg_scales[-1])
            save_folder_fid = fid_folders[eval_cfg]

            metrics_dict = torch_fidelity.calculate_metrics(
                input1=save_folder_fid,
                input2=None,
                fid_statistics_file=args.fid_statistics_path,
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False
            )

            fid_score = metrics_dict['frechet_inception_distance']
            inception_score = metrics_dict['inception_score_mean']

            print(f"FID [{cfg_tag}]: {fid_score:.4f}")
            print(f"Inception Score [{cfg_tag}]: {inception_score:.4f}")

            if is_main:
                if metrics['fid'] is not None:
                    metrics['fid'].append(fid_score)
                if metrics['is'] is not None:
                    metrics['is'].append(inception_score)

            if local_rank == 0:
                if args.use_wandb and epoch is not None:
                    log({
                        f"evaluate/fid_{cfg_tag}": fid_score,
                        f"evaluate/is_{cfg_tag}": inception_score,
                        "epoch": epoch,
                    })

                if epoch is not None:
                    metrics_file = os.path.join(args.output_dir, f"eval_{cfg_tag}.jsonl")
                    row = {"epoch": epoch, "cfg_scale": eval_cfg}
                    if global_step is not None:
                        row["step"] = int(global_step)
                    row["fid"] = float(fid_score)
                    row["is"] = float(inception_score)
                    with open(metrics_file, "a") as f:
                        f.write(json.dumps(row) + "\n")

                shutil.rmtree(save_folder_fid)

    torch.distributed.barrier()