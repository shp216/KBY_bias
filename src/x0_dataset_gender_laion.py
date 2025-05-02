import os
import math
import torch
import pandas as pd
from torch.utils.data import Dataset
from safetensors.torch import load_file
import random
import numpy as np
import json
import pickle

class x0_dataset_laion(Dataset):
    def __init__(self, data_dir, extra_text_dir=None, n_T=1000, random_conditioning = False, 
                 random_conditioning_lambda=5, world_size=1, rank=0, drop_text=True, drop_text_p=0.1, 
                 use_unseen_setting=False, gpt_caption=False, max_extra_text_samples=None, safe_tensor=None,
                 use_linear_timestep=None, use_exp_timestep=None
                ):
        """
        Args:
            data_dir (str): 데이터가 저장된 폴더의 경로.
        """
        self.data_dir = data_dir
        self.n_T = n_T
        self.random_conditioning = random_conditioning
        self.random_conditioning_lambda = random_conditioning_lambda
        self.world_size = world_size
        self.rank = rank
        
        self.drop_text = drop_text
        self.drop_text_p = drop_text_p
        self.gpt_caption = gpt_caption
        self.max_extra_text_samples = max_extra_text_samples
        self.safe_tensor = safe_tensor
        self.use_linear_timestep = use_linear_timestep
        self.use_exp_timestep = use_exp_timestep

        if self.use_exp_timestep:
            print("use exp timestep!!")
        
        # select metadata
        metadata_path = os.path.join(data_dir, "metadata.csv")
        self.metadata = pd.read_csv(metadata_path)
        # metadata_path2 = os.path.join(data_dir, "bias_gender_miil.csv")
        # self.metadata2 = pd.read_csv(metadata_path2)
        #metadata_path2 = os.path.join("./data/bias_train/bias_gender_miil.csv")
        #metadata_path1 = os.path.join("./data/x0_occupation_gender_miil_24k.csv")
        #self.metadata1 = pd.read_csv(metadata_path1)
        self.text_data = []
        
        with open('./data/1-prompts/occupation.json', 'r') as f:
            self.occupation_json = json.load(f)
        self.occupations = self.occupation_json["occupations_train_set"]
             
    
    def __len__(self):
        # 유효한 인덱스 개수를 반환합니다.+
        return len(self.metadata)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 파일들을 로드하여 반환합니다.
        #random_idx = torch.randint(0, len(self.metadata2), (1,)).item()
        metadata_row = self.metadata.iloc[idx]
        file_name = metadata_row['file_name']
        if idx % 2 == 0:
            gender = "male"
        else:
            gender = "female"
        occup_index = torch.randint(0, len(self.occupations), (1,)).long()
        occupation = self.occupations[occup_index]

        prompt_template_teacher = self.occupation_json["prompt_templates_train_teacher"][0]  # 템플릿 불러오기
        prompt_template_student = self.occupation_json["prompt_templates_train_student"][0]  # 템플릿 불러오기
    
        if self.safe_tensor:
            latent_file_name = file_name.replace('.jpg', '_latent.safetensors')
            latent_path = os.path.join(self.data_dir, latent_file_name)
            data = load_file(latent_path)
            latent_tensor = data["tensor"]

        else:
            # Modify the file_name to get latent file name
            latent_file_name = file_name.replace('.jpg', '_latent.pt')
            latent_path = os.path.join(self.data_dir, latent_file_name)
            
            latent_tensor = torch.load(latent_path, map_location=torch.device('cpu'))
            

        #timestep = torch.randint(0, self.n_T, (1,)).long()  
        if self.use_linear_timestep:
            weights = torch.arange(1, self.n_T + 1, dtype=torch.float)
            timestep = torch.multinomial(weights / weights.sum(), 1).long()
            #print("Use Linear timestep!")
        elif self.use_exp_timestep:
            timesteps = torch.arange(0, self.n_T, dtype=torch.float)
            lambda_exp = 5.0
            weights = torch.exp(-lambda_exp * (1 - timesteps / self.n_T))  # 후반 몰림
            probs = weights / weights.sum()
            timestep = torch.multinomial(probs, 1).long()
        else:
            timestep = torch.randint(0, self.n_T, (1,)).long()
        
        paired = torch.tensor(1).unsqueeze(0)
        if self.random_conditioning:
            t_value = timestep.item()
            #p = math.exp(-self.random_conditioning_lambda * (1 - t_value / self.n_T)) #random conditioning
            p = 1
            if torch.rand(1).item() < p:
                #rand_index = torch.randint(0, len(self.data_indices), (1,)).item()
                #random_idx = torch.randint(0, len(self.metadata2), (1,)).item()
                selected_row = self.metadata1.iloc[idx]  # .iloc 사용해서 행 접근
                gender = selected_row['gender']
                occupation = selected_row['occupation']
                teacher_text = prompt_template_teacher.format(gender=gender, occupation=occupation)
                student_text = prompt_template_student.format(occupation=occupation)
                #print(f"student text: {student_text}\ntimestep: {t_value}")
                # #선택된 행의 정보 가져오기 (DataFrame 형태를 가정)
                # selected_row = self.metadata2.iloc[random_idx]  # .iloc 사용해서 행 접근

                # teacher_text = selected_row['teacher_prompt']
                # student_text = selected_row['student_prompt'] 
        
                paired = torch.tensor(0).unsqueeze(0)

        teacher_text = prompt_template_teacher.format(gender=gender, occupation=occupation)
        student_text = prompt_template_student.format(occupation=occupation)
        
        if self.drop_text:
            if random.random() < self.drop_text_p:  # 10% 확률
                teacher_text = ""
                student_text = ""                

        return latent_tensor, teacher_text, student_text, timestep, paired

def collate_fn(tokenizer):
    def collate(batch):
        latents, teacher_texts, student_texts, timesteps, paireds = zip(*batch)
        
        latent_tensors = torch.stack(latents)
        timesteps = torch.cat(timesteps)
        paireds = torch.cat(paireds)

        # teacher_texts와 student_texts에 대해 각각 토크나이징
        teacher_captions = []
        student_captions = []
        
        for teacher_text, student_text in zip(teacher_texts, student_texts):
            # Teacher Caption 처리
            if isinstance(teacher_text, str):
                teacher_captions.append(teacher_text)
            elif isinstance(teacher_text, (list, np.ndarray)):
                teacher_captions.append(random.choice(teacher_text))
            else:
                raise ValueError(
                    f"Teacher Caption `{teacher_text}` should be a string or a list of strings."
                )
            
            # Student Caption 처리
            if isinstance(student_text, str):
                student_captions.append(student_text)
            elif isinstance(student_text, (list, np.ndarray)):
                student_captions.append(random.choice(student_text))
            else:
                raise ValueError(
                    f"Student Caption `{student_text}` should be a string or a list of strings."
                )

        # Tokenization
        teacher_inputs = tokenizer(
            teacher_captions, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        student_inputs = tokenizer(
            student_captions, 
            max_length=tokenizer.model_max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )

        # Tokenized Inputs
        teacher_input_ids = teacher_inputs.input_ids
        student_input_ids = student_inputs.input_ids

        return {
            "latents": latent_tensors,
            "teacher_input_ids": teacher_input_ids,
            "student_input_ids": student_input_ids,
            "timesteps": timesteps,
            "paireds": paireds
        }
    return collate


# def collate_fn(tokenizer, tokenizer_2=None):
#     def collate(batch):
#         latents, texts, timesteps, paireds = zip(*batch)
        
#         latent_tensors = torch.stack(latents)
#         timesteps = torch.cat(timesteps)
#         paireds = torch.cat(paireds)

#         captions = []
#         for caption in texts:
#             if isinstance(caption, str):
#                 captions.append(caption)
#             elif isinstance(caption, (list, np.ndarray)):
#                 # take a random caption if there are multiple
#                 captions.append(random.choice(caption))
#             else:
#                 raise ValueError(
#                     f"Caption column `{caption}` should contain either strings or lists of strings."
#                 )
#         inputs = tokenizer(
#             captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
#         )
        
#         input_ids = inputs.input_ids

#         if tokenizer_2 is not None:
#             inputs_2 = tokenizer(
#                 captions, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
#             )      
#             input_ids_2 = inputs_2.input_ids
#         else:
#             input_ids_2=None
        
#         return {
#             "latents": latent_tensors,
#             "input_ids": input_ids,
#             "input_ids_2": input_ids_2,
#             "timesteps": timesteps,
#             "paireds":paireds
#         }
#     return collate