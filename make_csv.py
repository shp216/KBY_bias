import pandas as pd
import re

# CSV 파일 경로 설정
csv_file = './data/x0_occupation_gender.csv'

# CSV 파일 로드
df = pd.read_csv(csv_file)

# gender와 occupation 열 추가하기
genders = []
occupations = []

# 정규표현식 패턴 정의
pattern = r"A photo of the face of a (female|male) (.*), a person"

for prompt in df['prompt']:
    match = re.match(pattern, prompt)
    if match:
        gender = match.group(1)  # female 또는 male
        occupation = match.group(2)  # occupation 이름
    else:
        gender = "Unknown"
        occupation = "Unknown"
    
    genders.append(gender)
    occupations.append(occupation)

# 새로운 열 추가
df['gender'] = genders
df['occupation'] = occupations

# 저장할 파일 이름 설정
output_file = './data/x0_occupation_gender_miil.csv'
df.to_csv(output_file, index=False)

print(f"새로운 CSV 파일이 저장되었습니다: {output_file}")
