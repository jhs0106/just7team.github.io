# RandomizedSearchCV
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# === 데이터 로드 ===
df = pd.read_csv("data.csv", encoding="cp949")  # 필요 시 신입생 파일로 변경
target = "중도탈락율" # 중도탈락학생(신입생)비율(%) / 중도탈락율

features = [
    "정원내 신입생 충원율(%)", "기숙사 수용률", "1인당장학금", "소규모 강좌 비율",
    "전임교원 1인당 학생수(정원기준)", "취업률", "진학률", "가중 평균 재학생 충원율(%)"
]

df_clean = df.dropna(subset=features + [target])
X = df_clean[features]
y = df_clean[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === 탐색할 하이퍼파라미터 분포 정의
param_dist = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# === RandomizedSearchCV 실행
random_search = RandomizedSearchCV(
    estimator=XGBRegressor(tree_method='hist', random_state=42),
    param_distributions=param_dist,
    n_iter=30,                      # 무작위 30조합만 탐색
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train, y_train)

# === 결과 출력
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n📌 Best Params:", random_search.best_params_)
print(f"✅ 최적 모델 RMSE: {rmse(y_test, y_pred):.4f}")
print(f"✅ 최적 모델 R²: {r2_score(y_test, y_pred):.4f}")
