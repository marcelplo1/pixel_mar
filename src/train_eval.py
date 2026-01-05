import copy
import os
import shutil
import cv2
from scipy import stats
from torchvision import transforms
import torch
import numpy as np
from sample import sample
import torch_fidelity

from utils.utils import adjust_learning_rate, patchify, sample_order, save_multiple_imgs_as_fig

def random_masking(x, orders, min_mask_rate=0.7):
    bsz, seq_len, embed_dim = x.shape
    mask_rate = min_mask_rate
    mask_rate = stats.truncnorm((min_mask_rate - 1.0) / 0.25, 0, loc=1.0, scale=0.25).rvs(1)[0]
    num_masked_tokens = int(np.ceil(seq_len * mask_rate))
    mask = torch.zeros(bsz, seq_len, device=x.device)
    mask = torch.scatter(mask, dim=-1, index=orders[:, :num_masked_tokens],
                            src=torch.ones(bsz, seq_len, device=x.device))
    return mask

def train_one_epoch(args, epoch, dataloader, mae, denoiser, optimizer, device, global_rank=0):
    mae.train()
    denoiser.train()    
    
    losses = []
    for step, (samples, labels) in enumerate(dataloader):
        adjust_learning_rate(optimizer, step / len(dataloader) + epoch, args)
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x = patchify(samples, args.patch_size)
        orders = sample_order(x.shape[0], x.shape[1], device)
        mask = random_masking(x, orders, args.min_mask_rate)

        z = mae(x, mask, labels)
        
        loss = denoiser(x, z, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        mae.update_ema()
        denoiser.update_ema()

        losses.append(loss.item())

        if args.is_debug and global_rank == 0:
            pass
            #save_img_as_fig(unpatchify(x*mask.unsqueeze(-1), args.patch_size, x.shape[1], args.channels), filename="mask.png", path=args.output_dir)

    return losses

def evaluate(args, mae, denoiser, device, epoch=None, global_rank=0, fids=None, dataset=None):
    mae.eval()
    denoiser.eval()

    local_rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    save_folder_fid = os.path.join(args.output_dir, "generation_{}_fid".format(args.dataset))
    os.makedirs(save_folder_fid, exist_ok=True)

    folder_class = os.path.join(args.output_dir, "generation_{}".format(args.dataset))
    folder_class = os.path.join(folder_class, 'epoch{}'.format(epoch))
    os.makedirs(folder_class, exist_ok=True)

    if args.activate_ema: 
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
    class_label_gen_world = np.arange(0, args.class_num).repeat(args.num_images // args.class_num)
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    num_steps = args.num_images // (args.gen_batch_size * world_size) + 1
    bsz = args.gen_batch_size
    saved_one_per_class = {c: False for c in range(args.class_num)}
    for step in range(num_steps):
        start_idx = world_size * bsz * step + local_rank * bsz
        end_idx = start_idx + bsz
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.Tensor(labels_gen).long().cuda()
        
        sampled_images = sample(args, mae, denoiser, labels_gen, device)
        sampled_images = (sampled_images + 1) / 2
        sampled_images = sampled_images.detach().cpu()

        torch.distributed.barrier() if torch.distributed.is_initialized() else None

        if global_rank == 0:
            if epoch is not None:
                for b_id in range(sampled_images.size(0)):
                    img_id = step * sampled_images.size(0) * world_size + local_rank * sampled_images.size(0) + b_id
                    if img_id >= args.num_images:
                        break
                    resize = transforms.Resize((args.sampled_img_size, args.sampled_img_size), interpolation=transforms.InterpolationMode.BILINEAR)
                    img_down = resize(sampled_images[b_id])
                    gen_img = np.round(np.clip(img_down.numpy().transpose([1, 2, 0]) * 255, 0, 255))
                    gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(os.path.join(save_folder_fid, 'sample_{}.png'.format(str(img_id).zfill(5))), gen_img)

                    cls = int(labels_gen[b_id].item())
                  
                    if cls < args.class_num and not saved_one_per_class[cls]:
                        cv2.imwrite(os.path.join(folder_class, 'sample_{}.png'.format(str(img_id).zfill(5))), gen_img)
                        saved_one_per_class[cls] = True

            else:
                save_multiple_imgs_as_fig(sampled_images, args.patch_size, filename=f"sampled_batch_{step}.png", path=args.output_dir)

    if args.activate_ema:
        print("Switch back from ema")
        mae.load_state_dict(mae_state_dict)
        denoiser.load_state_dict(denoiser_state_dict)

    # Evaluate the generation quality
    isc = False
    fid = False
    kid = False
    prc = False
    if args.dataset == 'imagenet':
        input2 = None
        fid_statistics_file = 'fid_stats/adm_in256_stats.npz'
        fid = True
        isc = True
    elif args.dataset == 'cifar10':
        input2 = Input2Dataset(dataset)
        fid_statistics_file = None
        fid = True
        isc = True
    elif args.dataset == 'mnist':
        input2 = Input2Dataset(dataset)
        fid_statistics_file = None
        fid = True
    else: # TODO
        input2 = Input2Dataset(dataset)
        fid_statistics_file = None
        fid = True
    metrics_dict = torch_fidelity.calculate_metrics(
        input1=save_folder_fid,
        input2=input2,
        fid_statistics_file=fid_statistics_file,
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
        if fids is not None:
            fids.append(fid_score)
    if isc:
        inception_score = metrics_dict['inception_score_mean']
        print("Inception Score: {:.4f}".format(inception_score))
    if kid:
        kid_score = metrics_dict['kernel_inception_distance_mean']
        print("KID: {:.4f}".format(kid_score))
    if prc:
        pass

    shutil.rmtree(save_folder_fid)

    
class Input2Dataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __getitem__(self, index):
        img, _ = self.base_dataset[index]
        img = img.mul(0.5).add_(0.5)
          
        # Scale to [0, 255] and convert to uint8
        img = (img * 255).clamp(0, 255).to(torch.uint8)
        
        return img

    def __len__(self):
        return len(self.base_dataset)