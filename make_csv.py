import pandas as pd
import json
import os

# 경로 설정
eval_template_path = "./data/bias_eval/bias_eval_template.csv"
occupation_json_path = "./data/1-prompts/occupation.json"
output_csv_path = "./bias_eval_gender.csv"

# 불러오기
template_df = pd.read_csv(eval_template_path)
with open(occupation_json_path, 'r') as f:
    occupations = json.load(f)["occupations_test_set"]

prompt_templates = template_df["prompt"].tolist()
assert len(prompt_templates) == 20

records = []

for occupation_id, occupation in enumerate(occupations):
    img_counter = 0  # img0.jpg ~ img39.jpg
    for prompt_id, template in enumerate(prompt_templates):
        formatted_prompt = template.replace("{occupation}", occupation)
        for image_index in range(2):  # 2장씩
            file_name = os.path.join(
                f"prompt_{occupation_id}",
                f"img_{img_counter}.jpg"
            )
            records.append({
                "file_name": file_name,
                "occupation": occupation,
                "prompt_id": prompt_id,
                "image_index": image_index,
                "prompt": formatted_prompt
            })
            img_counter += 1

# 저장
df = pd.DataFrame(records)[["file_name", "occupation", "prompt_id", "image_index", "prompt"]]
df.to_csv(output_csv_path, index=False)
print(f"✅ CSV 생성 완료: {output_csv_path} (총 {len(df)} rows)")


# import pandas as pd
# import json
# import random

# # 설정
# template_csv_path = "./bias_gender_race_template.csv"  # gender+race 템플릿
# occupation_json_path = "./data/1-prompts/occupation.json"
# output_csv_path = "./bias_gender_race_miil.csv"

# # 불러오기
# template_df = pd.read_csv(template_csv_path)
# with open(occupation_json_path, "r") as f:
#     occupations = json.load(f)["occupations_train_set"]

# student_templates = template_df["student_prompt"].tolist()
# teacher_templates = template_df["teacher_prompt"].tolist()

# genders = ["male", "female"]
# races = ["White", "Asian", "Black", "Indian"]

# # 준비: 각 템플릿 인덱스를 100번씩 반복 → 10000개 리스트
# template_indices = list(range(100)) * 100
# random.shuffle(template_indices)

# assert len(template_indices) == 10000
# assert len(occupations) == 1000

# records = []
# filename_index = 0

# # 각 occupation에 대해 10개씩 템플릿 할당
# for i, occupation in enumerate(occupations):
#     assigned_indices = template_indices[i * 10: (i + 1) * 10]

#     for template_idx in assigned_indices:
#         student_template = student_templates[template_idx]
#         teacher_template = teacher_templates[template_idx]

#         for gender in genders:
#             for race in races:
#                 student_prompt = student_template.replace("{occupation}", occupation)
#                 teacher_prompt = teacher_template.replace("{occupation}", occupation).replace("{gender}", gender).replace("{race}", race)

#                 records.append({
#                     "filename": f"{filename_index}.png",
#                     "gender": gender,
#                     "race": race,
#                     "occupation": occupation,
#                     "template_id": template_idx,
#                     "student_prompt": student_prompt,
#                     "teacher_prompt": teacher_prompt
#                 })
#                 filename_index += 1

# # 저장
# df = pd.DataFrame(records)
# df.to_csv(output_csv_path, index=False)
# print(f"✅ 완료: {output_csv_path} (총 {len(df)}개, template 100개 × 100회 × gender 2 × race 4)")
