import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- 설정 ---
USE_FAKE_DATA = True
REAL_DATA_PATH = "real_dropout_data.csv"

# --- 데이터 로딩 ---
def load_data(use_fake=True, path=None):
    if use_fake:
        np.random.seed(42)
        n = 300
        employment_rate = np.round(np.random.uniform(40, 95, n), 1)
        dropout_rate = (
            100 - employment_rate * np.random.uniform(0.4, 0.6) + np.random.normal(0, 3, n)
        )
        dropout_rate = np.clip(dropout_rate, 0, 100)
        return pd.DataFrame({
            "employment_rate": employment_rate,
            "dropout_rate": dropout_rate
        })
    else:
        return pd.read_csv(path)

# --- 모델 훈련 및 평가 ---
def train_linear_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    coef = model.coef_[0]
    intercept = model.intercept_

    print("\nDropout Rate Prediction Based on Employment Rate")
    print(f" - RMSE      : {round(rmse, 2)}")
    print(f" - R² Score  : {round(r2, 4)}")
    print(f" - Equation  : dropout_rate = {round(coef, 3)} * employment_rate + {round(intercept, 3)}")

    return model, X_test, y_test, y_pred

# --- 시각화 ---
def plot_results(X_test, y_test, y_pred):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, alpha=0.6, label="Actual")
    plt.plot(X_test, y_pred, color="red", label="Prediction Line")
    plt.xlabel("Employment Rate (%)")
    plt.ylabel("Dropout Rate (%)")
    plt.title("Linear Regression: Employment vs Dropout Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# --- 실행 ---
df = load_data(USE_FAKE_DATA, REAL_DATA_PATH)
X = df[["employment_rate"]]
y = df["dropout_rate"]

model, X_test, y_test, y_pred = train_linear_model(X, y)
plot_results(X_test, y_test, y_pred)
