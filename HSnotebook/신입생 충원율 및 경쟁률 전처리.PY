
import pandas as pd
import os
from glob import glob

# 모든 연도별 엑셀 파일 경로 가져오기
file_paths = sorted(glob("데이터_폴더_경로/*.xlsx"))

all_dfs = []

for file_path in file_paths:
    try:
        # 엑셀 파일에서 시트 읽기
        df_raw = pd.read_excel(file_path, sheet_name='Sheet1', header=None)

        # 헤더: 보통 4~5행 정도에 컬럼 정보가 분산됨
        multi_header = df_raw.iloc[3:5].fillna(method='ffill', axis=1)
        flat_columns = multi_header.apply(lambda x: f"{x.iloc[0]}_{x.iloc[1]}" if x.iloc[1] else x.iloc[0], axis=0)

        # 실제 데이터는 5행 아래에서 시작
        df = df_raw.iloc[5:].copy()
        df.columns = flat_columns
        df = df.reset_index(drop=True)

        # 필요한 컬럼만 선택 (인덱스가 연도마다 다를 경우는 컬럼명으로 지정)
        df_sub = df[[
            [col for col in df.columns if "기준" in col][0],
            [col for col in df.columns if "학교" in col][0],
            [col for col in df.columns if "충원율" in col][0],
            [col for col in df.columns if "경쟁률" in col][0],
        ]]
        df_sub.columns = ['기준년도', '학교', '정원내 신입생 충원율(%)', '경쟁률']

        # 결측값 처리 및 숫자형 변환
        df_sub['학교'] = df_sub['학교'].fillna(method='ffill')
        df_sub['기준년도'] = df_sub['기준년도'].fillna(method='ffill')
        df_sub['정원내 신입생 충원율(%)'] = pd.to_numeric(df_sub['정원내 신입생 충원율(%)'], errors='coerce')
        df_sub['경쟁률'] = pd.to_numeric(df_sub['경쟁률'], errors='coerce')

        # 이상치 제거
        df_sub = df_sub[
            (df_sub['학교'] != '대학교') &
            (df_sub['정원내 신입생 충원율(%)'] > 0) &
            (df_sub['경쟁률'] > 0)
        ]

        # 기준년도 정제
        df_sub['기준년도'] = df_sub['기준년도'].astype(str).str.extract(r'(\\d{4})')[0]

        all_dfs.append(df_sub)

    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")

# 통합
final_df = pd.concat(all_dfs, ignore_index=True)

# 2023년 기준 유지할 학교 추출 (사이버 대학교는 제거)
schools_2023 = set(final_df[final_df['기준년도'] == '2023']['학교'].unique())
final_df = final_df[final_df['학교'].isin(schools_2023)]
final_df = final_df[~final_df['학교'].str.contains('사이버')]

# 저장 (CP949 for Excel)
final_df.to_csv("통합_충원률_경쟁률_2010_2023_CP949.csv", index=False, encoding='cp949')
