import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# 데이터 로드 (예: 재학생 결측치 제거 기준)
df = pd.read_csv("Zscore_data.csv", encoding="cp949")
target = "중도탈락율"
label = "재학생_결측치제거"

features = [
    "정원내 신입생 충원율(%)", "기숙사 수용률", "1인당장학금", "소규모 강좌 비율",
    "전임교원 1인당 학생수(정원기준)", "취업률", "진학률", "가중 평균 재학생 충원율(%)"
]

required = features + [target, "학교명", "기준연도"]
df_clean = df.dropna(subset=required)

# 결과 저장용 리스트
results = []

print(f"📊 개별 피처 성능 비교 ({label})")
print("=" * 50)

for f in features:
    print(f"\n🔍 Feature: {f}")
    
    X = df_clean[[f]]
    y = df_clean[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 선형 회귀
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_rmse = rmse(y_test, lr_pred)
    lr_r2 = r2_score(y_test, lr_pred)

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1,
                       tree_method="hist", random_state=42, verbosity=0)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    xgb_rmse = rmse(y_test, xgb_pred)
    xgb_r2 = r2_score(y_test, xgb_pred)

    # 결과 저장
    results.append({
        "Feature": f,
        "Linear_RMSE": lr_rmse,
        "Linear_R2": lr_r2,
        "XGB_RMSE": xgb_rmse,
        "XGB_R2": xgb_r2
    })

# 결과 정리 및 출력
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="XGB_RMSE")

print("\n📋 성능 비교 결과 (XGBoost RMSE 기준 오름차순)")
print(results_df.round(4))
