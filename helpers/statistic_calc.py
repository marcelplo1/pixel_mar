import argparse
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3
import numpy as np
from PIL import Image

def create_parser():
    parser = argparse.ArgumentParser(description="Argument Parser for toy example")
    parser.add_argument("--img_size", type=int, default=64, help="Image size (assumed square)")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size")
    parser.add_argument("--data_path", type=str, default="../data/imagenet", help="Directory to the ImageNet dataset")
    parser.add_argument("--save_path", type=str, default="../fid_stats", help="Directory where to save the statistics")
    parser.add_argument("--single_class", action="store_true", help="Calculate the statistics for a single class")
    parser.add_argument("--single_class_index", type=int, default=207, help="Class index for a single class")

    return parser

def main():
    args = create_parser().parse_args()

    args.data_path = "/mnt/lustre/datasets/ImageNet2012"

    transform = transforms.Compose([
        transforms.Lambda(lambda img: center_crop_arr(img, args.img_size)),
        transforms.ToTensor(), 
    ])
    
    dataset = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform)

    if args.single_class:
        targets = torch.tensor(dataset.targets)
        indices = (targets == args.single_class_index).nonzero(as_tuple=True)[0]
        dataset = Subset(dataset, indices)
        if len(dataset) == 0:
            print("Fixed dataset class not found!")
            return

    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=10, shuffle=False)

    model = FeatureExtractorInceptionV3(name='inception-v3-compat', features_list=['2048'])
    model = model.cuda().eval()

    features = []
    print(f"Extracting features for {len(dataset)} images...")

    for i, (img, _) in enumerate(loader):
        img = (img * 255).clamp(0, 255).to(torch.uint8).cuda()
        with torch.no_grad():
            out = model(img)[0] 
            features.append(out.cpu())
        if i % 100 == 0:
            print(f"Batch {i}/{len(loader)}")

    features = torch.cat(features, dim=0).numpy()
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)

    if args.single_class:
        filename = f"adm_in{args.img_size}_stats_class{args.single_class_index}.npz"
    else:
        filename = f"adm_in{args.img_size}_stats_full.npz"

    os.makedirs(args.save_path, exist_ok=True)
    final_path = os.path.join(args.save_path, filename)

    np.savez(final_path, mu=mu, sigma=sigma)
    print(f"Success! Saved to {final_path}")

    
def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])

if __name__ == '__main__':
    main()