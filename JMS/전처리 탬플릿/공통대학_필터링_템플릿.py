
import pandas as pd

def filter_by_common_schools(data_path, output_path, school_column_name="학교", encoding="utf-8-sig", is_excel=False):
    """
    공통대학 기준에 따라 데이터를 필터링하고 저장하는 함수.

    Parameters:
    - data_path: 처리할 CSV 또는 Excel 파일 경로
    - output_path: 결과를 저장할 CSV 파일 경로
    - school_column_name: 원본 데이터의 학교명 컬럼명 ("학교" 또는 "학교명")
    - encoding: 파일 인코딩 (CSV용, 기본값 'utf-8-sig')
    - is_excel: 엑셀 파일 여부 (True면 Excel로 읽기)
    """
    # 1. 공통대학 기준 불러오기
    common_school_df = pd.read_csv("공통학교명_리스트.csv", encoding="utf-8-sig")
    common_school_set = set(common_school_df["공통학교명"].unique())

    # 2. 대상 파일 불러오기
    if is_excel:
        df = pd.read_excel(data_path)
    else:
        df = pd.read_csv(data_path, encoding=encoding)

    # 3. 학교명 정규화
    df["학교명_정규화"] = df[school_column_name].replace({
        "경남과학기술대학교": "경상국립대학교"
    })

    # 4. 공통대학 기준 필터링
    filtered_df = df[df["학교명_정규화"].isin(common_school_set)].copy()

    # 5. 저장
    filtered_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f">>> 저장 완료: {output_path}")

# 사용 예시:
# filter_by_common_schools("입력파일.csv", "결과파일.csv")
# filter_by_common_schools("입력파일.xlsx", "결과파일.csv", is_excel=True)
