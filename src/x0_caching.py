import os
import torch
from accelerate import Accelerator
from diffusers import StableDiffusionPipeline
from torch.utils.data import DataLoader, Dataset
import json

# Dataset to handle occupation prompts
class OccupationDataset(Dataset):
    def __init__(self, prompt_template, occupations, num_images_per_prompt=24):
        self.prompts = [
            prompt_template.format(occupation=occupation)
            for occupation in occupations
            for _ in range(num_images_per_prompt)
        ]
        self.occupations = occupations
        self.num_images_per_prompt = num_images_per_prompt

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        occupation_idx = idx // self.num_images_per_prompt  # 해당 프롬프트의 occupation 인덱스
        occupation = self.occupations[occupation_idx]
        image_index = idx % self.num_images_per_prompt  # 0 ~ 23 사이의 이미지 번호
        return prompt, occupation, image_index


def generate_images(pipeline, dataloader, save_dir, device):
    generator = torch.Generator(device=device).manual_seed(1234)
    
    for batch in dataloader:
        prompts, occupations, image_indices = batch  # 배치 단위로 데이터 받기
        prompts = list(prompts)  # 중요 수정!

        # 배치로 이미지 생성 (pipeline 호출을 배치 단위로 수행)
        images = pipeline(prompts, num_inference_steps=25, generator=generator).images  # Batch 생성
        
        for i, image in enumerate(images):
            # 저장 경로 설정
            occupation = occupations[i]
            image_index = image_indices[i].item()
            occupation_dir = os.path.join(save_dir, occupation)
            os.makedirs(occupation_dir, exist_ok=True)

            # 이미지 이름 저장 (ex. doctor0.png, doctor1.png, ...)
            img_name = f"{occupation}{image_index}.png"
            image.save(os.path.join(occupation_dir, img_name))


# Main function
def main():
    accelerator = Accelerator() 
    device = accelerator.device

    # Load prompt and occupation data
    with open('./data/1-prompts/occupation.json', 'r') as f:
        data = json.load(f)
    
    prompt_template = data["prompt_templates_train"][0]  # "A photo of the face of a {occupation}, a person"
    occupations = data["occupations_train_set"]
    
    # Define save directory for generated images
    save_dir = "./x0_occupation"
    os.makedirs(save_dir, exist_ok=True)

    # Initialize dataset and dataloader
    dataset = OccupationDataset(prompt_template, occupations)
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
    generate_images(pipeline, dataloader, save_dir, device)


if __name__ == "__main__":
    main()














# import os
# import torch
# from accelerate import Accelerator
# from diffusers import StableDiffusionPipeline
# from torch.utils.data import DataLoader, Dataset

# # Dataset to handle txt files and their corresponding content
# class TxtDataset(Dataset):
#     def __init__(self, txt_dir):
#         self.txt_dir = txt_dir
#         self.txt_files = [f for f in os.listdir(txt_dir) if f.endswith('.txt')]

#     def __len__(self):
#         return len(self.txt_files)

#     def __getitem__(self, idx):
#         txt_file = self.txt_files[idx]
#         with open(os.path.join(self.txt_dir, txt_file), 'r') as f:
#             text = f.read().strip()
#         return text, txt_file

# # Function to generate images from the given dataset
# def generate_images(pipeline, dataloader, save_dir, device):
#     generator = torch.Generator(device=device).manual_seed(1234)
#     for batch in dataloader:
#         texts, txt_files = batch
#         for i, text in enumerate(texts):
#             image = pipeline(text, num_inference_steps=25, generator=generator).images[0]
#             img_name = txt_files[i].replace('.txt', '.jpg')
#             image.save(os.path.join(save_dir, img_name))

# # Main function
# def main():
#     accelerator = Accelerator() 
#     device = accelerator.device

#     # Define directories
#     txt_dir = "./data/laion_aes/train"  # Directory where txt files are located
#     save_dir = "./data/laion_aes/x0_cache_212k/train"  # Directory where generated images will be saved
#     os.makedirs(save_dir, exist_ok=True)

#     # Initialize dataset and dataloader
#     dataset = TxtDataset(txt_dir)
#     dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

#     # Load the Stable Diffusion pipeline
#     pipeline = StableDiffusionPipeline.from_pretrained(
#         "CompVis/stable-diffusion-v1-4",
#         safety_checker=None,
#     )
#     pipeline = pipeline.to(device)

#     # Use accelerator to prepare dataloader
#     pipeline, dataloader = accelerator.prepare(pipeline, dataloader)

#     # Generate and save images based on txt file content
#     generate_images(pipeline, dataloader, save_dir, device)

# if __name__ == "__main__":
#     main()
