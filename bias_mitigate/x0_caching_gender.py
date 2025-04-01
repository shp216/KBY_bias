import os
import torch
import pandas as pd
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
import json

# Dataset to handle occupation prompts
class OccupationDataset(Dataset):
    def __init__(self, prompt_template, occupations, num_images_per_prompt=12):
        self.prompts = []
        self.occupations = []
        self.prompt_types = []  # Female 또는 Male 프롬프트 정보 저장
        
        for occupation in occupations:
            # Female 프롬프트 12개 추가
            for i in range(num_images_per_prompt):
                prompt = f"A photo of the face of a female {occupation}, a person"
                self.prompts.append(prompt)
                self.occupations.append(occupation)
                self.prompt_types.append("female")
            
            # Male 프롬프트 12개 추가
            for i in range(num_images_per_prompt):
                prompt = f"A photo of the face of a male {occupation}, a person"
                self.prompts.append(prompt)
                self.occupations.append(occupation)
                self.prompt_types.append("male")
        
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        occupation = self.occupations[idx]
        prompt_type = self.prompt_types[idx]  # 'female' 또는 'male'
        image_index = idx % 12  # 0 ~ 11 사이의 이미지 번호로 저장
        return prompt, occupation, prompt_type, image_index


def generate_images(pipeline, dataloader, save_dir, device, csv_file):
    generator = torch.Generator(device=device).manual_seed(1234)
    rows = []  # CSV 파일에 저장할 데이터를 담는 리스트
    
    for batch in dataloader:
        prompts, occupations, prompt_types, image_indices = batch
        prompts = list(prompts)  # 중요!

        # 배치로 이미지 생성
        images = pipeline(prompts, num_inference_steps=25, generator=generator).images
        
        for i, image in enumerate(images):
            occupation = occupations[i]
            prompt_type = prompt_types[i]
            image_index = image_indices[i].item()
            
            # 이미지 저장 경로 설정
            occupation_dir = os.path.join(save_dir, occupation)
            os.makedirs(occupation_dir, exist_ok=True)
            
            # 이미지 이름 생성 및 저장 (ex. doctor_female0.png, doctor_male0.png)
            img_name = f"{occupation}_{prompt_type}{image_index}.png"
            img_path = os.path.join(occupation_dir, img_name)
            image.save(img_path)
            
            # CSV에 기록할 데이터 추가
            rows.append({"image": img_name, "prompt": prompts[i]})
    
    # DataFrame으로 변환 후 CSV 파일로 저장
    df = pd.DataFrame(rows)
    df.to_csv(csv_file, index=False)


# Main function
def main():
    accelerator = Accelerator() 
    device = accelerator.device

    # Load prompt and occupation data
    with open('./data/1-prompts/occupation.json', 'r') as f:
        data = json.load(f)
    
    occupations = data["occupations_train_set"]
    
    # Define save directory for generated images and CSV file
    save_dir = "./data/x0_occupation_gender"
    csv_file = "./data/x0_occupation_gender.csv"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = OccupationDataset("A photo of the face of a {occupation}, a person", occupations)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load the Stable Diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        safety_checker=None,
    )
    pipeline = pipeline.to(device)

    # Use accelerator to prepare dataloader
    pipeline, dataloader = accelerator.prepare(pipeline, dataloader)

    # Generate and save images based on occupation prompts
    generate_images(pipeline, dataloader, save_dir, device, csv_file)


if __name__ == "__main__":
    main()
