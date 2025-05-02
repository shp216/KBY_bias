#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import itertools
import math
import os


import torch
from tqdm.auto import tqdm

from diffusers import (
    AutoencoderKL,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.loaders import LoraLoaderMixin
from diffusers.models.attention_processor import LoRAAttnProcessor
from transformers import CLIPTextModel, CLIPTokenizer
import json
import pytz

from torch import nn
from torchvision import transforms
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from utils.misc import get_file_list_from_csv, change_img_size, change_img_size_ddp, change_img_size_bias
import hashlib

# os.environ["gpu_ids"] = "1"
my_timezone = pytz.timezone("Asia/Singapore")

os.environ["WANDB__SERVICE_WAIT"] = "300"  # set to DETAIL for runtime logging.

def stable_hash(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest(), 16) % (10**8)

@torch.no_grad()
def generate_image(prompt, noises, tokenizer, text_encoder, unet, vae, noise_scheduler, num_denoising_steps=25, guidance_scale=7.5, device="cuda:0", weight_dtype=torch.float16, weight_dtype_high_precision=torch.float32):
    """
    prompts: str
    noises: [N,4,64,64], N is number images to be generated for the prompt
    """

    N = noises.shape[0]
    prompts = [prompt] * N
    
    prompts_token = tokenizer(prompts, return_tensors="pt", padding=True)
    prompts_token["input_ids"] = prompts_token["input_ids"].to(device)
    prompts_token["attention_mask"] = prompts_token["attention_mask"].to(device)

    prompt_embeds = text_encoder(
        prompts_token["input_ids"],
        prompts_token["attention_mask"],
    )
    prompt_embeds = prompt_embeds[0]

    batch_size = prompt_embeds.shape[0]
    uncond_tokens = [""] * batch_size
    max_length = prompt_embeds.shape[1]
    uncond_input = tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
    uncond_input["input_ids"] = uncond_input["input_ids"].to(device)
    uncond_input["attention_mask"] = uncond_input["attention_mask"].to(device)
    negative_prompt_embeds = text_encoder(
        uncond_input["input_ids"],
        uncond_input["attention_mask"],
    )
    negative_prompt_embeds = negative_prompt_embeds[0]

    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
    prompt_embeds = prompt_embeds.to(weight_dtype)
    
    noise_scheduler.set_timesteps(num_denoising_steps)
    latents = noises
    for i, t in enumerate(noise_scheduler.timesteps):
    
        # scale model input
        latent_model_input = torch.cat([latents.to(weight_dtype)] * 2)
        latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
        
        noises_pred = unet(
            latent_model_input,
            t,
            encoder_hidden_states=prompt_embeds,
        ).sample
        noises_pred = noises_pred.to(weight_dtype_high_precision)
        
        noises_pred_uncond, noises_pred_text = noises_pred.chunk(2)
        noises_pred = noises_pred_uncond + guidance_scale * (noises_pred_text - noises_pred_uncond)
        
        latents = noise_scheduler.step(noises_pred, t, latents).prev_sample

    latents = 1 / vae.config.scaling_factor * latents
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1,1) # in range [-1,1]
    
    return images

def eval_gender_images(args, accelerator, tokenizer, text_encoder, unet, vae, noise_scheduler, i=None, mode="eval_score"):
    device = accelerator.device
    
    unet.eval()

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # # Move unet, vae and text_encoder to device and cast to weight_dtype
    # text_encoder.to(device, dtype=weight_dtype)
    # unet.to(device, dtype=weight_dtype)
    # vae.to(device, dtype=weight_dtype)    
    
    # Load test prompts
    with open(args.prompts_path, 'r') as f:
        experiment_data = json.load(f)
    
    if mode == "eval_images":
        test_prompts = experiment_data['val_prompts']
        num_imgs_per_prompt = args.num_imgs_per_prompt_valid
    else:
        template = experiment_data["prompt_templates_test"][i]
        test_prompts = [template.format(occupation=occ) for occ in experiment_data["occupations_test_set"]]
        num_imgs_per_prompt = args.num_imgs_per_prompt
    
        
    # Split prompts across GPUs
    num_gpus = accelerator.num_processes
    prompts_per_gpu = torch.tensor_split(torch.tensor(range(len(test_prompts))), num_gpus)
    local_rank = accelerator.local_process_index
    print(f"prompts_per_gpu: {prompts_per_gpu}, local_rank: {local_rank}")
    local_prompts = [test_prompts[idx] for idx in prompts_per_gpu[local_rank].tolist()]

    # Prepare noise for each prompt
    noise_all = []
    # for prompt in local_prompts:
    #     noise_per_prompt = []
    #     for i in range(num_imgs_per_prompt):
    #         torch.manual_seed(args.eval_random_seed + hash(prompt) + i)
    #         noise_single = torch.randn([1, 4, 64, 64], dtype=weight_dtype_high_precision).to(accelerator.device)
    #         noise_per_prompt.append(noise_single)
    #     noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
    #     noise_all.append(noise_per_prompt)
    # noise_all = torch.cat(noise_all)
    for prompt in local_prompts:
        noise_per_prompt = []
        for i in range(num_imgs_per_prompt):
            seed = args.eval_random_seed + hash(prompt) + i
            gen = torch.Generator()
            gen.manual_seed(seed)
            noise_single = torch.randn([1, 4, 64, 64], generator=gen, dtype=weight_dtype_high_precision).to(accelerator.device)
            noise_per_prompt.append(noise_single)
        noise_per_prompt = torch.cat(noise_per_prompt).unsqueeze(0)
        noise_all.append(noise_per_prompt)
    noise_all = torch.cat(noise_all)

    for i, prompt_i in tqdm(enumerate(local_prompts), total=len(local_prompts), desc='Generating Images', leave=True):
        
        global_prompt_idx = prompts_per_gpu[local_rank][i].item()
        save_dir_prompt_i = os.path.join(args.eval_save_dir, f"prompt_{global_prompt_idx}")
        os.makedirs(save_dir_prompt_i, exist_ok=True)

        noises_to_use = []
        img_save_paths_to_use = []
        for j in range(num_imgs_per_prompt):
            img_save_path = os.path.join(save_dir_prompt_i, f"img_{j}.jpg")
            if not os.path.exists(img_save_path):
                noises_to_use.append(noise_all[i, j].unsqueeze(dim=0))
                img_save_paths_to_use.append(img_save_path)
        noises_to_use = torch.cat(noises_to_use)

        N = math.ceil(noises_to_use.shape[0] / args.eval_batch_size)
        for j in tqdm(range(N), desc='Images per prompt', leave=False):
            noises_ij = noises_to_use[args.eval_batch_size * j:args.eval_batch_size * (j + 1)]
            img_save_paths_ij = img_save_paths_to_use[args.eval_batch_size * j:args.eval_batch_size * (j + 1)]

            images_ij = generate_image(
                prompt_i,
                noises_ij,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                unet=unet,
                vae=vae,
                noise_scheduler=noise_scheduler,
                num_denoising_steps=args.num_denoising_steps,
                guidance_scale=args.guidance_scale,
                device=accelerator.device,
                weight_dtype=weight_dtype,
                weight_dtype_high_precision=weight_dtype_high_precision
            )

            for img, img_save_path in itertools.zip_longest(images_ij, img_save_paths_ij):
                img_pil = transforms.ToPILImage()(img * 0.5 + 0.5)
                img_pil.save(img_save_path)
    
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        change_img_count = change_img_size_bias(args.eval_save_dir, args.eval_save_dir_256, args.img_resz)
        accelerator.print(f"resized images saved.")
        accelerator.print(f"Total resized images: {change_img_count}")
    accelerator.wait_for_everyone()

    unet.train()
