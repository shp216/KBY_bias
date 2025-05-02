import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None, required=False,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--max_train_samples", type=int, default=None, help="Truncate the number of training examples.")
    parser.add_argument("--output_dir", type=str, default="./data/x0_occupation_gender_240k_latent",
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--seed", type=int, default=1234, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512, help="The resolution for input images.")
    parser.add_argument("--center_crop", action="store_true", help="Whether to center crop the input images.")
    parser.add_argument("--random_flip", action="store_true", help="Whether to randomly flip images horizontally.")
    parser.add_argument("--train_batch_size", type=int, default=64, help="Batch size (per device).")
    parser.add_argument("--dataloader_num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    args = parser.parse_args()
    print('args')
    return args

def main():
    args = parse_args()
    
    # Set up Accelerator
    accelerator = Accelerator()
    device = accelerator.device

    # Set random seed for reproducibility
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    
    print('seed')
    # Load models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer", revision=args.revision)
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)

    print('models')
    # Image preprocessing
    image_transforms = transforms.Compose([
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    class PngDataset(Dataset):
        def __init__(self, image_dir, transform=None):
            self.image_dir = image_dir
            self.transform = transform
            self.image_paths = []

            for root, _, files in os.walk(image_dir):  # 모든 폴더를 탐색하여 PNG 파일 경로 수집
                for file in files:
                    if file.endswith(".png"):
                        self.image_paths.append(os.path.join(root, file))
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            image_path = self.image_paths[idx]
            image = Image.open(image_path).convert("RGB")
            
            if self.transform:
                image = self.transform(image)
            
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            return image, base_name  # 이미지와 파일 이름 반환


    # DataLoader 설정
    img_dir = "./data/x0_occupation_gender_240k"
    
    
    print('before dataset')
    
    dataset = PngDataset(img_dir, transform=image_transforms)
    
    print('dataset')

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers,
        collate_fn=lambda x: list(zip(*x))
    )

    # Prepare models for distributed training
    train_dataloader, text_encoder, vae = accelerator.prepare(train_dataloader, text_encoder, vae)
     
    print('accelerator prepare')
    def process_batch(batch):
        images, base_names = batch

        # 이미지를 배치 단위로 처리
        if images:
            image_tensors = torch.stack(images).to(device)  # [B, C, H, W] 형태
            with torch.no_grad():
                latents = vae.module.encode(image_tensors).latent_dist.sample() if isinstance(vae, torch.nn.parallel.DistributedDataParallel) else vae.encode(image_tensors).latent_dist.sample()
                latents = latents * vae.module.config.scaling_factor  # Stable Diffusion의 scaling factor
        else:
            latents = None

        return latents, base_names
    
    print("output_dir: ", args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 배치 단위로 데이터 처리
    for batch in tqdm(train_dataloader, desc="Processing Batches"):
        latents, base_names = process_batch(batch)

        # 결과 저장
        for latent, base_name in zip(latents, base_names):
            if latent is not None:
                latent_path = os.path.join(args.output_dir, f"{base_name}_latent.pt")
                torch.save(latent.cpu(), latent_path)
                print('save',latent_path)
        

if __name__ == "__main__":
    main()