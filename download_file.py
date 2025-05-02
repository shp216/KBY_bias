from huggingface_hub import hf_hub_download
import os
import shutil

# 정확한 다운로드 경로
cached_file_path = hf_hub_download(
    repo_id="shp216/bias_data",
    filename="exp-1-debias-gender/outputs/from-paper_finetune-text-encoder_09190215/checkpoint-9800_exported/text_encoder_lora_EMA.pth",
    repo_type="dataset"
)

# 복사할 위치 지정
target_dir = "./results_miil/Gender"
os.makedirs(target_dir, exist_ok=True)

# 복사 (이제 깨진 링크 대신 실제 파일이 저장됨)
target_path = os.path.join(target_dir, "text_encoder_lora_EMA.pth")
shutil.copy(cached_file_path, target_path)

print("✅ Copied to:", target_path)

# from huggingface_hub import hf_hub_download
# import shutil
# import os

# # 다운로드 (cache_dir 내부에 huggingface가 내부 폴더 구조 유지함)
# cached_file_path = hf_hub_download(
#     repo_id="shp216/bias_data",
#     filename="1k3d68.onnx",
#     repo_type="dataset",
# )

# # 복사 위치
# target_dir = "./data/buffalo_l"
# os.makedirs(target_dir, exist_ok=True)

# # 복사 (파일 이름 유지)
# target_path = os.path.join(target_dir, "1k3d68.onnx")
# shutil.copy(cached_file_path, target_path)

# print("Copied to:", target_path)



# from huggingface_hub import snapshot_download

# snapshot_download(
#     repo_id="shp216/bias_data",
#     repo_type="dataset",
#     local_dir="../"
# )


# from huggingface_hub import hf_hub_download

# # 원하는 파일만 다운로드
# file_path = hf_hub_download(
#     repo_id="shp216/bias_data",
#     filename="w600k_r50.onnx",  # 정확한 파일 이름
#     repo_type="dataset",
#     cache_dir="./data/buffalo"  # optional: 저장 경로
# )

# print("Downloaded to:", file_path)

