# file: model_Zscore_data.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# === 데이터 로드 ===
df = pd.read_csv("Zscore_data.csv", encoding="cp949")
label = "재학생_Zscore"
target = "중도탈락율"

features = [
    "정원내 신입생 충원율(%)", "기숙사 수용률", "1인당장학금", "소규모 강좌 비율",
    "전임교원 1인당 학생수(정원기준)", "취업률", "진학률", "가중 평균 재학생 충원율(%)"
]

required = features + [target, "학교명", "기준연도"]
df_clean = df.dropna(subset=required)

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 선형 회귀 ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
lr_train_pred = lr.predict(X_train_scaled)
lr_test_pred = lr.predict(X_test_scaled)

# === XGBoost ===
xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                   tree_method="hist", random_state=42, verbosity=0)
xgb.fit(X_train, y_train)
xgb_train_pred = xgb.predict(X_train)
xgb_test_pred = xgb.predict(X_test)

# === 평가
lr_rmse_test = rmse(y_test, lr_test_pred)
xgb_rmse_test = rmse(y_test, xgb_test_pred)
lr_r2_test = r2_score(y_test, lr_test_pred)
xgb_r2_test = r2_score(y_test, xgb_test_pred)

print(f"\n✅ {label} 모델링 결과")
print(f"[선형회귀] RMSE: {lr_rmse_test:.4f}, R²: {lr_r2_test:.4f}")
print(f"[XGBoost] RMSE: {xgb_rmse_test:.4f}, R²: {xgb_r2_test:.4f}")

# === 전체 예측
X_all_scaled = scaler.transform(X)
df_clean["선형회귀_예측"] = lr.predict(X_all_scaled)
df_clean["XGBoost_예측"] = xgb.predict(X)
df_clean["실제값"] = y
df_clean["선형회귀_오차"] = abs(df_clean["실제값"] - df_clean["선형회귀_예측"])
df_clean["XGBoost_오차"] = abs(df_clean["실제값"] - df_clean["XGBoost_예측"])

# 결과 출력
result_cols = ["학교명", "기준연도", "실제값", "선형회귀_예측", "XGBoost_예측", "선형회귀_오차", "XGBoost_오차"]
result = df_clean[result_cols].sort_values(by=["학교명", "기준연도"])
print(result.head(10).round(3))

# 저장
# result.to_csv(f"{label}_예측결과.csv", index=False, encoding="utf-8-sig")
# print(f"💾 저장 완료: {label}_예측결과.csv")
