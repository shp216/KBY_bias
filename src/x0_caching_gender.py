import os
import pandas as pd
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time

# 새로운 CSV 기반 Dataset
class PromptFromCSV(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["prompt"], row["file_name"]


# 이미지 생성 함수
def generate_images_from_csv(pipeline, dataloader, save_dir, device, accelerator):
    seed = 1234 + accelerator.process_index
    generator = torch.Generator(device=device).manual_seed(seed)

    for batch in dataloader:
        prompts, filenames = batch
        prompts = list(prompts)
        filenames = list(filenames)

        images = pipeline(prompts, num_inference_steps=25, generator=generator).images

        for i, image in enumerate(images):
            save_path = os.path.join(save_dir, filenames[i])
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)

    accelerator.wait_for_everyone()


def main():
    accelerator = Accelerator()
    device = accelerator.device

    # 설정
    csv_path = "./data/x0_occupation_gender_miil_24k.csv"
    save_dir = "./x0_gender_image"
    os.makedirs(save_dir, exist_ok=True)

    # Dataset & DataLoader
    dataset = PromptFromCSV(csv_path)
    dataloader = DataLoader(dataset, batch_size=60, shuffle=False)

    # Stable Diffusion 파이프라인 로드
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
    )
    pipeline = pipeline.to(device)

    # Accelerator 준비
    pipeline, dataloader = accelerator.prepare(pipeline, dataloader)

    start_time = time.time()
    # 이미지 생성
    generate_images_from_csv(pipeline, dataloader, save_dir, device, accelerator)

    # 시간 측정 끝
    elapsed_time = time.time() - start_time
    accelerator.print(f" 이미지 생성에 걸린 시간: {elapsed_time:.2f}초 ({elapsed_time/60:.2f}분)")

if __name__ == "__main__":
    main()
