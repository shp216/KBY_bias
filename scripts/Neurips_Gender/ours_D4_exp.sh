# ------------------------------------------------------------------------------------
# Copyright 2023. Nota Inc. All Rights Reserved.
# Code modified from https://github.com/huggingface/diffusers/tree/v0.15.0/examples/text_to_image
# ------------------------------------------------------------------------------------
MODEL_NAME="runwayml/stable-diffusion-v1-5"

#TRAIN_DATA_DIR="./data/laion_aes/pt_cache_212k" # please adjust it if needed
TRAIN_DATA_DIR="./data/x0_occupation_gender_240k_latent" # 절대 경로로 설정]
EXTRA_TEXT_DIR=None

UNET_CONFIG_PATH=None

UNET_NAME="teacher" # option: ["teacher", "bk_base", "bk_small", "bk_tiny"]
#OUTPUT_DIR="./results/bksdm_feature/GTimg_rand_cond_copy_"$UNET_NAME # please adjust it if needed
OUTPUT_DIR="./results_neurips/Gender/ours_D4_exp" # please adjust it if needed

MODEL_ID="runwayml/stable-diffusion-v1-5"

DATA_SIZE="4"
BATCH_SIZE=12
GRAD_ACCUMULATION=1
NUM_GPUS=2
StartTime=$(date +%s)

#CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --multi_gpu --num_processes ${NUM_GPUS} src/kd_train_text_to_image_gender.py \
#CUDA_VISIBLE_DEVICES=0 accelerate launch --num_processes ${NUM_GPUS} src/kd_train_gender_multitmp.py \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --num_processes ${NUM_GPUS} --main_process_port 29655 src/kd_train_text_to_image_gender.py \
  --pretrained_model_name_or_path $MODEL_NAME \
  --train_data_dir $TRAIN_DATA_DIR\
  --extra_text_dir $EXTRA_TEXT_DIR\
  --num_train_x0 $DATA_SIZE \
  --resolution 512 --center_crop --random_flip \
  --train_batch_size $BATCH_SIZE \
  --gradient_checkpointing \
  --mixed_precision="fp16" \
  --learning_rate 5e-05 \
  --max_grad_norm 1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --report_to="wandb" \
  --seed 1234 \
  --gradient_accumulation_steps $GRAD_ACCUMULATION \
  --checkpointing_steps 400 \
  --evaluation_step 400 \
  --valid_steps 400 \
  --lambda_sd 1.0 --lambda_kd_output 1.0 --lambda_kd_feat 1.0 \
  --unet_config_path $UNET_CONFIG_PATH --unet_config_name $UNET_NAME \
  --output_dir $OUTPUT_DIR \
  --max_train_steps 4000 \
  --model_id $MODEL_ID \
  --drop_text \
  --random_conditioning \
  --use_copy_weight_from_teacher \
  --dataloader_num_workers 2 \
  --use_exp_timestep \
  --resume_from_checkpoint "latest" \

EndTime=$(date +%s)
echo "** KD training takes $(($EndTime - $StartTime)) seconds."