# mdtest2.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# === 데이터 로딩 및 병합 ===
df = pd.read_csv("Zscore_data.csv")
money = pd.read_csv("money_data.csv", encoding="cp949")

money = money.rename(columns={"기준년도": "기준연도"})
money["재학생"] = money["재학생"].astype(str).str.replace(",", "").astype(float)

money_grouped = (
    money.groupby(["기준연도", "학교명"], as_index=False)[
        ["학자금대출이용학생비율(%) 전체", "학자금대출이용학생비율(%) 등록금"]
    ].mean()
)

df["기준연도"] = df["기준연도"].astype(int)
money_grouped["기준연도"] = money_grouped["기준연도"].astype(int)
df = pd.merge(df, money_grouped, how="left", on=["기준연도", "학교명"])

# === 피처 목록 ===
all_features = [
    "정원내 신입생 충원율(%)",
    "기숙사 수용률",
    "1인당장학금",
    "소규모 강좌 비율",
    "전임교원 1인당 학생수(정원기준)",
    "취업률",
    "진학률",
    "가중 평균 재학생 충원율(%)",
    "학자금대출이용학생비율(%) 전체",
    "학자금대출이용학생비율(%) 등록금"
]

top5_features = [
    "가중 평균 재학생 충원율(%)",
    "학자금대출이용학생비율(%) 전체",
    "정원내 신입생 충원율(%)",
    "진학률",
    "1인당장학금"
]

target = "중도탈락학생(신입생)비율(%)"

# === 공통 함수: 모델링 & 평가 ===
def train_and_evaluate(X, y, name):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    xgb_r2 = r2_score(y_test, xgb_pred)

    # 선형 회귀
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    lr_r2 = r2_score(y_test, lr_pred)

    # 결과 출력
    print(f"\n[{name}]")
    print(f"XGBoost:    RMSE = {xgb_rmse:.2f}, R² = {xgb_r2:.2f}")
    print(f"선형 회귀: RMSE = {lr_rmse:.2f}, R² = {lr_r2:.2f}")

# === 전체 변수 모델 ===
df_all = df.dropna(subset=all_features + [target])
print("[전체 변수 기반 모델 성능]")
train_and_evaluate(df_all[all_features], df_all[target], "전체 변수")

# === 상위 5개 변수 모델 ===
df_top5 = df.dropna(subset=top5_features + [target])
print("\n[상위 5개 변수 기반 모델 성능]")
train_and_evaluate(df_top5[top5_features], df_top5[target], "상위 5개 변수")
