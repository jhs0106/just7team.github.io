
import pandas as pd

# === 1. 파일 로딩 ===
def load_excel_with_dynamic_header(filepath):
    for header_row in range(4, 8):  # 헤더 후보 행
        df = pd.read_excel(filepath, header=header_row)
        if any('학부' in str(col) and '전공' in str(col) for col in df.columns):
            return df
    raise ValueError("적절한 헤더를 찾지 못했습니다.")

# === 2. 컬럼 자동 탐색 및 매핑 ===
def extract_relevant_columns(df):
    column_map = {
        '학교명': None,
        '학부·과(전공)명': None,
        '대계열분류': None,
        '대학자체대계열': None
    }
    for col in df.columns:
        if '학교명' in col:
            column_map['학교명'] = col
        elif '학부·과' in col or '전공' in col:
            column_map['학부·과(전공)명'] = col
        elif '대계열' in col and '표준' in col:
            column_map['대계열분류'] = col
        elif '대학자체대계열' in col:
            column_map['대학자체대계열'] = col

    # 추출 및 컬럼명 정리
    filtered = df[[column_map[key] for key in column_map if column_map[key] is not None]].copy()
    filtered.columns = ['학교명', '학부·과(전공)명', '대계열분류', '대학자체대계열']
    return filtered

# === 3. 통합 및 중복 제거 ===
def merge_and_deduplicate(dfs):
    merged = pd.concat(dfs, ignore_index=True)
    deduped = merged.drop_duplicates(subset=['학부·과(전공)명', '대계열분류', '대학자체대계열'])
    return deduped

# === 4. 실행 ===
if __name__ == "__main__":
    filepaths = [
        "학교명_전공명_계열정보_추출.xlsx",
        "학교별_교육편제단위_정보_230228_계열추출.xlsx",
        "학교별_교육편제단위_정보_231005_계열추출.xlsx"
    ]

    dfs = []
    for path in filepaths:
        df = load_excel_with_dynamic_header(path)
        filtered = extract_relevant_columns(df)
        dfs.append(filtered)

    result = merge_and_deduplicate(dfs)
    result.to_excel("통합_학과_계열정보_정리.xlsx", index=False)
