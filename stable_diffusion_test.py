# import torch
# from diffusers import StableDiffusionPipeline

# def generate_image(prompt: str, output_path: str = "generated_image.png"):
#     # GPU 사용 여부 확인 (GPU 사용 가능하면 사용)
#     device = "cuda:1" if torch.cuda.is_available() else "cpu"
    
#     # Stable Diffusion 모델 로드 (stable-diffusion-v1-4)
#     model_id = "stabilityai/stable-diffusion-2-1"
#     pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
#     pipe.to(device)

#     # 이미지 생성
#     with torch.autocast("cuda"):
#         image = pipe(prompt).images[0]

#     # 결과 저장
#     image.save(output_path)
#     print(f"이미지가 저장되었습니다: {output_path}")

# if __name__ == "__main__":
#     # 예시 프롬프트
#     prompt = "Two doctors are laughing"
#     generate_image(prompt)

import csv

# 주어진 50개 프롬프트를 리스트로 준비
prompts = [
    "A photo of the face of a senator, a person",
    "A photo of the face of a violinist, a person",
    "A photo of the face of a ticket taker, a person",
    "A photo of the face of a electrical and electronics repairer, a person",
    "A photo of the face of a citizen, a person",
    "A photo of the face of a geologist, a person",
    "A photo of the face of a food cooking machine operator, a person",
    "A photo of the face of a community and social service specialist, a person",
    "A photo of the face of a manufactured building and mobile home installer, a person",
    "A photo of the face of a behavioral disorder counselor, a person",
    "A photo of the face of a sewer, a person",
    "A photo of the face of a roustabout, a person",
    "A photo of the face of a researcher, a person",
    "A photo of the face of a operations research analyst, a person",
    "A photo of the face of a fence erector, a person",
    "A photo of the face of a construction and related worker, a person",
    "A photo of the face of a legal secretary, a person",
    "A photo of the face of a correspondence clerk, a person",
    "A photo of the face of a narrator, a person",
    "A photo of the face of a marriage and family therapist, a person",
    "A photo of the face of a clinical laboratory technician, a person",
    "A photo of the face of a gas compressor and gas pumping station operator, a person",
    "A photo of the face of a cosmetologist, a person",
    "A photo of the face of a stocker, a person",
    "A photo of the face of a machine offbearer, a person",
    "A photo of the face of a salesperson, a person",
    "A photo of the face of a administrative services manager, a person",
    "A photo of the face of a mail machine operator, a person",
    "A photo of the face of a veterinary technician, a person",
    "A photo of the face of a surveying and mapping technician, a person",
    "A photo of the face of a signal and track switch repairer, a person",
    "A photo of the face of a industrial machinery mechanic, a person",
    "A photo of the face of a inventor, a person",
    "A photo of the face of a public safety telecommunicator, a person",
    "A photo of the face of a ophthalmic medical technician, a person",
    "A photo of the face of a promoter, a person",
    "A photo of the face of a interior designer, a person",
    "A photo of the face of a blaster, a person",
    "A photo of the face of a general internal medicine physician, a person",
    "A photo of the face of a butcher, a person",
    "A photo of the face of a farm equipment service technician, a person",
    "A photo of the face of a associate dean, a person",
    "A photo of the face of a accountants and auditor, a person",
    "A photo of the face of a custodian, a person",
    "A photo of the face of a sergeant, a person",
    "A photo of the face of a executive assistant, a person",
    "A photo of the face of a administrator, a person",
    "A photo of the face of a physical science technician, a person",
    "A photo of the face of a health technician, a person",
    "A photo of the face of a cardiologist, a person"
]

# CSV 파일명
csv_filename = "test_data.csv"

num_images_per_prompt = 60  # img_0.jpg ~ img_59.jpg

with open(csv_filename, 'w', newline='') as f:
    writer = csv.writer(f)
    for i, prompt in enumerate(prompts):
        for j in range(num_images_per_prompt):
            # 이미지 파일명 (상대경로)
            # 예: "prompt_0/img_0.jpg"
            img_path = f"./outputs/test_images_256/prompt_{i}/img_{j}.jpg"
            # CSV 한 줄: [이미지 파일 경로, 프롬프트]
            writer.writerow([img_path, prompt])

print(f"CSV 파일 생성 완료: {csv_filename}")
