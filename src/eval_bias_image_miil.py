import os
import math
import pandas as pd
import torch
from tqdm.auto import tqdm
from torchvision import transforms
from diffusers import StableDiffusionPipeline
from utils.misc import change_img_size_bias
from hashlib import sha256
def stable_hash(text: str) -> int:
    return int(sha256(text.encode()).hexdigest(), 16) % (10**8)
def get_noise(prompt: str, image_index: int, seed_offset: int = 42, device="cuda", dtype=torch.float32):
    seed = seed_offset + stable_hash(prompt) + image_index
    gen = torch.Generator()  # 🔁 여기에서 device 제거 (CPU 전용)
    gen.manual_seed(seed)
    return torch.randn([1, 4, 64, 64], generator=gen, dtype=dtype).to(device)
@torch.no_grad()
def generate_image(prompts, noises, tokenizer, text_encoder, unet, vae, noise_scheduler, num_denoising_steps=25, guidance_scale=7.5, device="cuda:0", weight_dtype=torch.float16, weight_dtype_high_precision=torch.float32):
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
    for t in noise_scheduler.timesteps:
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
    images = vae.decode(latents.to(vae.dtype)).sample.clamp(-1, 1)  # in range [-1, 1]
    return images
def eval_from_csv(args, accelerator, tokenizer, text_encoder, unet, vae, noise_scheduler):
    device = accelerator.device
    weight_dtype_high_precision = torch.float32
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    accelerator.print("Loading CSV and generating all noises...")
    df = pd.read_csv(args.eval_csv_path)
    prompts = df["prompt"].tolist()
    image_indices = df["image_index"].tolist()
    file_names = df["file_name"].tolist()
    # 모든 노이즈 미리 생성
    all_noises = torch.cat([
        get_noise(p, i, seed_offset=args.eval_random_seed, device=device, dtype=weight_dtype_high_precision)
        for p, i in zip(prompts, image_indices)
    ], dim=0)  # shape: [N, 4, 64, 64]
    # 분산 처리 분할
    df_split = df.iloc[accelerator.process_index::accelerator.num_processes].reset_index(drop=True)
    noises_split = all_noises[accelerator.process_index::accelerator.num_processes]
    unet.eval()
    for i in range(0, len(df_split), args.eval_batch_size):
        batch_df = df_split.iloc[i:i+args.eval_batch_size]
        batch_prompts = batch_df["prompt"].tolist()
        batch_file_names = batch_df["file_name"].tolist()
        batch_noises = noises_split[i:i+args.eval_batch_size]
        images = generate_image(
            batch_prompts,
            batch_noises,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            unet=unet,
            vae=vae,
            noise_scheduler=noise_scheduler,
            num_denoising_steps=args.num_denoising_steps,
            guidance_scale=args.guidance_scale,
            device=device,
            weight_dtype=weight_dtype,
            weight_dtype_high_precision=weight_dtype_high_precision,
        )
        for img, file_name in zip(images, batch_file_names):
            save_path = os.path.join(args.eval_save_dir, file_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img_pil = transforms.ToPILImage()(img * 0.5 + 0.5)
            img_pil.save(save_path)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        change_img_count = change_img_size_bias(args.eval_save_dir, args.eval_save_dir_256, args.img_resz)
        accelerator.print(f"Resized images saved.")
        accelerator.print(f"Total resized images: {change_img_count}")
    accelerator.wait_for_everyone()
    unet.train()