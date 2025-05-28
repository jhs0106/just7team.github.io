
import pandas as pd
from rapidfuzz import process

# === 파일 경로 설정 ===
manual_path = "기타_학과_전수_분류.xlsx"
etc_path = "최종_미분류_학과_목록_불분명제거_정제본.xlsx"
output_path = "유사도기반_자동매핑_결과.xlsx"

# === 파일 로딩 ===
manual_df = pd.read_excel(manual_path)
etc_df = pd.read_excel(etc_path)

# === 수동 분류 학과명 → 계열 매핑 dict 생성 ===
manual_map = dict(zip(
    manual_df['기타 계열로 남은 학과명'].astype(str).str.strip(),
    manual_df['계열분류'].astype(str).str.replace("계열", "").str.strip()
))

manual_names = list(manual_map.keys())
etc_majors = etc_df['학과명'].astype(str).str.strip().tolist()

# === 유사도 기반 매핑 ===
matches = []
for major in etc_majors:
    match, score, _ = process.extractOne(major, manual_names)
    if score >= 90:
        계열 = manual_map[match]
        matches.append((major, match, score, 계열))

# === 저장 ===
result_df = pd.DataFrame(matches, columns=['기타 학과명', '매칭된 수동 학과명', '유사도', '자동 지정 계열'])
result_df.to_excel(output_path, index=False)
print(f"자동 매핑 결과 {len(result_df)}건 저장 완료 → {output_path}")
