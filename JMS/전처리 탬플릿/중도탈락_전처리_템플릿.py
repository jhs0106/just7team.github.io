
import pandas as pd
import numpy as np

# === 1. 파일 로딩 ===
file_path = "파일경로.xlsx 또는 csv"
if file_path.endswith(".csv"):
    df = pd.read_csv(file_path, encoding="utf-8-sig")
else:
    df = pd.read_excel(file_path)

# === 2. 병합 해제 및 결측 채우기 ===
fill_cols = ['기준년도', '학교', '단과대학']  # 단과대학은 선택사항
for col in fill_cols:
    if col in df.columns:
        df[col] = df[col].ffill()

# === 3. 불필요한 행 제거 ===
if '학교종류' in df.columns:
    df = df[~df['학교종류'].str.contains('사이버대학', na=False)]

if '구분' in df.columns:
    df = df[~df['구분'].isin(['야간', '원격'])]

for col in ['재적학생', '학생정원']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] >= 10]

for col in ['중도탈락률', '재학생충원율']:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df[df[col] > 0]

# === 4. 기준연도 정제 ===
if '기준년도' in df.columns:
    df['기준년도'] = df['기준년도'].astype(str)
    df['기준년도'] = df['기준년도'].str.replace(r'상반기|상', '-1', regex=True)
    df['기준년도'] = df['기준년도'].str.replace(r'하반기|하', '-2', regex=True)
    df['기준년도'] = df['기준년도'].str.extract(r'(20\d{2})(?:-1|-2)?').fillna(method='ffill') + df['기준년도'].str.extract(r'(-1|-2)').fillna('')

# === 5. 계열 분류 ===
standard_major_dict = {
    "국어국문학": "인문사회계열",
    "기계공학": "공학계열",
    # ... 생략
}

second_major_df = pd.read_excel("통합_학과_계열정보_정리.xlsx")
second_major_dict = second_major_df.set_index("학과명")["대학자체대계열"].to_dict()

third_major_df = pd.read_excel("기타_학과_전수_분류.xlsx")
third_major_df = third_major_df.rename(columns={
    '기타 계열로 남은 학과명': '학과명',
    '계열분류': '대학자체대계열'
})
third_major_dict = third_major_df.set_index("학과명")["대학자체대계열"].to_dict()

if '대학자체대계열' not in df.columns:
    df['대학자체대계열'] = None

df['대학자체대계열'] = df['학과'].map(standard_major_dict).combine_first(
    df['학과'].map(second_major_dict)).combine_first(
    df['학과'].map(third_major_dict)
)

# === 6. 정렬 ===
if '기준년도' in df.columns and '학교' in df.columns:
    df = df.sort_values(by=['기준년도', '학교'])

# === 7. 저장 ===
df.to_csv("최종결과.csv", index=False, encoding="utf-8-sig")
