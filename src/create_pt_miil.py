import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./data/Gender_latent_20k")
    parser.add_argument("--image_dir", type=str, default="./data/Gender_image_20k")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    accelerator = Accelerator()
    device = accelerator.device

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load VAE
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
    
    image_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Dataset
    class ImageDataset(Dataset):
        def __init__(self, image_dir):
            self.image_dir = image_dir
            self.file_names = [f for f in os.listdir(image_dir) if f.endswith(".png")]

        def __len__(self):
            return len(self.file_names)

        def __getitem__(self, idx):
            file_name = self.file_names[idx]
            image_path = os.path.join(self.image_dir, file_name)
            image = Image.open(image_path).convert("RGB")
            image = image_transforms(image)
            base_name = os.path.splitext(file_name)[0]
            return image, base_name

    dataset = ImageDataset(args.image_dir)

    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
    )

    # Prepare
    dataloader, vae = accelerator.prepare(dataloader, vae)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    vae.eval()
    for images, base_names in tqdm(dataloader, desc="Processing"):
        images = images.to(device)
        with torch.no_grad():
            latents = vae.module.encode(images).latent_dist.sample() if isinstance(vae, torch.nn.parallel.DistributedDataParallel) else vae.encode(image_tensors).latent_dist.sample()
            latents = latents * vae.module.config.scaling_factor  # Stable Diffusion의 scaling factor

        for latent, base_name in zip(latents, base_names):
            save_path = os.path.join(args.output_dir, f"{base_name}_latent.pt")
            torch.save(latent.cpu(), save_path)

if __name__ == "__main__":
    main()
