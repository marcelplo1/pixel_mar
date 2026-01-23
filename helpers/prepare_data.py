import os
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default='../data/mnist-64', help='Folder where transformed images will be saved')
    parser.add_argument('--img_size', type=int, default=64, help='Resolution to upscale (e.g., 224 for ViT)')
    parser.add_argument('--dataset', type=str, default='mnist', help='Select dataset to store', choices=['mnist','cifar10'])
    
    args = parser.parse_args()

    transform_train = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])

    dataset_train = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform_train
    )
    
    class_names = dataset_train.classes
    data_loader = DataLoader(
        dataset_train,
        batch_size=128,
        num_workers=8,
        shuffle=False
    )

    for name in class_names:
        os.makedirs(os.path.join(args.output_path, name), exist_ok=True)

    to_pil = transforms.ToPILImage()
    counters = {name: 0 for name in class_names}

    print(f"Upscaling CIFAR-10 to {args.img_size}x{args.img_size}...")

    for batch_images, batch_labels in tqdm(data_loader):
        for i in range(batch_images.size(0)):
            img_tensor = batch_images[i]
            label_idx = batch_labels[i].item()
            class_name = class_names[label_idx]

            pil_img = to_pil(img_tensor)
            
            file_name = f"{class_name}_{counters[class_name]:05d}.png"
            out_path = os.path.join(args.output_path, class_name, file_name)
            
            pil_img.save(out_path, format='PNG', compress_level=0)
            counters[class_name] += 1

    print(f"Finished! Dataset saved to {args.output_path}")

if __name__ == "__main__":
    main()