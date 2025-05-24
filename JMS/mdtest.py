# mdtest.py
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# 1. 데이터 불러오기
df = pd.read_csv("Zscore_data.csv")
money = pd.read_csv("money_data.csv", encoding="cp949")

# 2. 전처리: 병합을 위한 학자금 데이터 정리
money = money.rename(columns={"기준년도": "기준연도"})
money["재학생"] = money["재학생"].astype(str).str.replace(",", "").astype(float)

money_grouped = (
    money.groupby(["기준연도", "학교명"], as_index=False)[
        ["학자금대출이용학생비율(%) 전체", "학자금대출이용학생비율(%) 등록금"]
    ].mean()
)

# 3. 병합
df["기준연도"] = df["기준연도"].astype(int)
money_grouped["기준연도"] = money_grouped["기준연도"].astype(int)
df = pd.merge(df, money_grouped, how="left", on=["기준연도", "학교명"])

# 4. 피처 및 타겟 정의
feature_columns = [
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
target_column = "중도탈락학생(신입생)비율(%)"

df = df.dropna(subset=feature_columns + [target_column])

X = df[feature_columns]
y = df[target_column]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. XGBoost 모델 학습
xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
xgb_r2 = r2_score(y_test, xgb_pred)

# 6. 선형 회귀 (표준화)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_pred = lr.predict(X_test_scaled)
lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
lr_r2 = r2_score(y_test, lr_pred)

# 7. 결과 출력
print("[XGBoost 성능]")
print(f"RMSE: {xgb_rmse:.2f}")
print(f"R²: {xgb_r2:.2f}")

print("\n[XGBoost 중요 변수 상위 5]")
xgb_importance = dict(zip(feature_columns, xgb.feature_importances_))
for name, score in sorted(xgb_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{name}: {score:.4f}")

print("\n[표준화된 선형 회귀 성능]")
print(f"RMSE: {lr_rmse:.2f}")
print(f"R²: {lr_r2:.2f}")

print("\n[표준화된 선형 회귀 중요 변수 상위 5 (절댓값 기준)]")
lr_coef = dict(zip(feature_columns, lr.coef_))
for name, coef in sorted(lr_coef.items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
    print(f"{name}: {coef:.4f}")
