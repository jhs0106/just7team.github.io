import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# === 데이터 로드 ===
df = pd.read_csv("new_data.csv", encoding="utf-8")
target = "중도탈락학생(신입생)비율(%)"

features = [
    "정원내 신입생 충원율(%)", "기숙사 수용률", "1인당장학금", "소규모 강좌 비율",
    "전임교원 1인당 학생수(정원기준)", "취업률", "진학률", "가중 평균 재학생 충원율(%)"
]

required = features + [target]
df_clean = df.dropna(subset=required)

X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 하이퍼파라미터 후보
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.05, 0.1, 0.2]
}

# === GridSearchCV 적용
grid = GridSearchCV(
    XGBRegressor(tree_method='hist', random_state=42),
    param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

# === 최적 모델로 평가
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)

# === 결과 출력
print("\n📊 [최적 하이퍼파라미터]")
for param, value in grid.best_params_.items():
    print(f" - {param}: {value}")

print("\n📈 [테스트 데이터 기준 성능]")
print(f"✅ RMSE (Root Mean Squared Error): {rmse(y_test, y_pred):.4f}")
print(f"✅ R² (설명력): {r2_score(y_test, y_pred):.4f}")
