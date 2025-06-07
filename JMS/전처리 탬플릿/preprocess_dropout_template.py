
import pandas as pd

def preprocess_dropout_data(file_path: str, output_path: str) -> None:
    """
    중도탈락 데이터 전처리 템플릿
    1~2행 유지, 기준년도 복원, 사이버대학/야간/원격 제거 포함

    Parameters:
    - file_path: 원본 .xlsx 파일 경로
    - output_path: 저장할 전처리 결과 파일 경로
    """

    # === 1. 파일 로드 ===
    xls = pd.ExcelFile(file_path)
    df_raw = xls.parse(xls.sheet_names[0], dtype=str)

    # === 2. 1~2행 분리 ===
    df_head = df_raw.iloc[:2].copy()
    df_data = df_raw.iloc[2:].copy()

    # === 3. 병합 해제 가정 후 빈 셀 채우기 ===
    df_data.ffill(inplace=True)

    # === 4. 열 이름 수동 지정 ===
    df_data.columns = ['기준년도', '학교', '학부과전공명', '구분', '학과특성', '학과상태', '재적학생', '중도탈락비율']

    # === 5. 기준년도 복구 ===
    # '가야대학교 인터넷보안학과' 기준으로 위는 2011, 아래는 2010
    target_idx = df_data[
        (df_data['학교'].str.contains('가야대학교', na=False)) &
        (df_data['학부과전공명'].str.contains('인터넷보안학과', na=False))
    ].index.min()

    df_data.loc[:target_idx, '기준년도'] = '2011'
    df_data.loc[target_idx + 1:, '기준년도'] = '2010'

    # === 6. 필터링 ===
    if '학교종류' in df_data.columns:
        df_data = df_data[~df_data['학교종류'].str.contains('사이버대학', na=False)]
    else:
        df_data = df_data[~df_data['학교'].str.contains('사이버대학', na=False)]

    df_data = df_data[~df_data['구분'].isin(['야간', '원격'])]

    # === 7. 결과 저장 ===
    with pd.ExcelWriter(output_path) as writer:
        df_data.to_excel(writer, index=False, sheet_name="전처리데이터")
        df_head.to_excel(writer, index=False, sheet_name="원본상단2행")

# === 사용 예시 ===
# preprocess_dropout_data("2010~2011 중도탈락.xlsx", "전처리완료_2010_2011_최종수정본.xlsx")
