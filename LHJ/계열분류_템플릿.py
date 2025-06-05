import pandas as pd
import re
from rapidfuzz import process, fuzz

# 데이터 불러오기
df_scholarship = pd.read_csv("학과별_장학금_10-20.csv")
df_mapping = pd.read_csv("대학자체대계열분류.csv")

# 학과명 정제
def normalize_major(text):
    if pd.isna(text):
        return ''
    text = str(text)
    text = re.sub(r'\(.*?\)', '', text)  # 괄호 제거
    text = re.sub(r'[^가-힣A-Za-z0-9]', '', text)  # 한글, 영문, 숫자가 아닌 문자 제거
    return text.strip()

df_scholarship['학과명'] = df_scholarship['학과명'].apply(normalize_major)
df_scholarship['학교명'] = df_scholarship['학교명'].astype(str).str.strip()

df_mapping['학과명'] = df_mapping['학과명'].apply(normalize_major)
df_mapping['학교명'] = df_mapping['학교명'].astype(str).str.strip()

# 학과명 기준 분류
major_group = df_mapping.groupby('학과명')['대학자체대계열'].nunique()
single_majors = major_group[major_group == 1].index.tolist()

df_mapping_single = df_mapping[df_mapping['학과명'].isin(single_majors)].drop_duplicates(subset=['학과명'])[['학과명', '대학자체대계열']]
df_merged = pd.merge(df_scholarship, df_mapping_single, how='left', on='학과명')

# 분류 실패한 학과는 학교명,학과명 기준으로 분류
df_remaining = df_merged[df_merged['대학자체대계열'].isna()].drop(columns=['대학자체대계열'])
df_mapping_multi = df_mapping[~df_mapping['학과명'].isin(single_majors)].drop_duplicates(subset=['학교명', '학과명'])

df_remaining_merged = pd.merge(df_remaining, df_mapping_multi, how='left', on=['학교명', '학과명'])

# 분류 결과 통합
df_combined = pd.concat([
    df_merged[df_merged['대학자체대계열'].notna()],
    df_remaining_merged
], ignore_index=True)

# fuzzy 매칭을 위한 유사 학과 자동 매핑
df_unmatched = df_combined[df_combined['대학자체대계열'].isna()].copy()
print(f"병합 실패 학과 수 (fuzzy 대상): {len(df_unmatched)}")

# 매핑용 리스트 및 딕셔너리 생성
mapping_majors_list = df_mapping['학과명'].dropna().unique().tolist()
major_to_dept = df_mapping.drop_duplicates('학과명').set_index('학과명')['대학자체대계열'].to_dict()

# 유사도 매칭 함수
def get_closest_major_and_dept_fast(major_name):
    match, score, _ = process.extractOne(major_name, mapping_majors_list, scorer=fuzz.token_sort_ratio)
    if score >= 80:
        dept = major_to_dept.get(match, None)
        return pd.Series([dept, match, score])
    else:
        return pd.Series([None, None, score])

# fuzzy matching 수행
df_unmatched[['대학자체대계열', '추천_학과명', '유사도']] = df_unmatched['학과명'].apply(get_closest_major_and_dept_fast)

# 자동 분류 결과 통합
df_unmatched_matched = df_unmatched[df_unmatched['대학자체대계열'].notna()]
df_unmatched_failed = df_unmatched[df_unmatched['대학자체대계열'].isna()]

df_final = pd.concat([
    df_combined[df_combined['대학자체대계열'].notna()],
    df_unmatched_matched,
    df_unmatched_failed
], ignore_index=True)

# csv 저장
df_final.to_csv('학과별_장학금_최종_계열포함.csv', index=False, encoding = "UTF-8-sig")

# 분류 실패 학과 저장
if not df_unmatched_failed.empty:
    df_unmatched_failed.to_csv('최종_계열_미지정.csv', index=False, encoding = "UTF-8-sig")