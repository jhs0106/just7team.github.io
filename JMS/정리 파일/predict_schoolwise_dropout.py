import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# RMSE 계산 함수
def rmse(y_true, y_pred):
    """RMSE (Root Mean Squared Error) 계산"""
    return np.sqrt(mean_squared_error(y_true, y_pred))

# === 대상 파일 목록 ===
files = [
    ("data.csv", "재학생_결측치제거"),
    ("Zscore_data.csv", "재학생_Zscore"),
    ("new_data.csv", "신입생_결측치제거"),
    ("new_Zscore_data.csv", "신입생_Zscore")
]

# === 특성 변수 정의 ===
features = [
    "정원내 신입생 충원율(%)", 
    "기숙사 수용률", 
    "1인당장학금", 
    "소규모 강좌 비율",
    "전임교원 1인당 학생수(정원기준)", 
    "취업률", 
    "진학률", 
    "가중 평균 재학생 충원율(%)"
]

# === 모델 성능 결과 저장용 ===
results_summary = []

print("🎯 취업률에 따른 중도탈락률 예측 모델링")
print("=" * 60)

for file_name, label in files:
    print(f"\n🔍 {label} 데이터 분석 중...")
    print("-" * 50)

    # === 데이터 로드 ===
    try:
        df = pd.read_csv(file_name, encoding="utf-8")
        print(f"✅ 파일 로드 성공: {file_name}")
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_name, encoding="cp949")
            print(f"✅ 파일 로드 성공 (cp949): {file_name}")
        except Exception as e:
            print(f"❌ 파일 로드 실패: {file_name} - {e}")
            continue
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {file_name}")
        continue

    # === 타겟 변수 자동 설정 ===
    if "중도탈락율" in df.columns:
        target = "중도탈락율"
    elif "중도탈락학생(신입생)비율(%)" in df.columns:
        target = "중도탈락학생(신입생)비율(%)"
    else:
        print(f"❌ 타겟 컬럼이 존재하지 않습니다: {file_name}")
        continue
    
    print(f"🎯 타겟 변수: {target}")

    # === 데이터 전처리 ===
    # 필요한 컬럼들이 모두 있는지 확인
    required_columns = features + [target, "학교명", "기준연도"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        print(f"❌ 누락된 컬럼들: {missing_columns}")
        continue
    
    # 결측치 제거
    df_clean = df.dropna(subset=required_columns)
    print(f"📊 데이터 크기: {len(df)} → {len(df_clean)} (결측치 제거 후)")
    
    if len(df_clean) < 10:  # 최소 데이터 개수 확인
        print(f"❌ 데이터가 너무 적습니다 ({len(df_clean)}개). 최소 10개 이상 필요.")
        continue

    # === 특성 변수와 타겟 변수 분리 ===
    X = df_clean[features]
    y = df_clean[target]
    
    print(f"📈 특성 변수: {len(features)}개")
    print(f"📊 타겟 변수 통계:")
    print(f"   - 평균: {y.mean():.2f}")
    print(f"   - 표준편차: {y.std():.2f}")
    print(f"   - 최솟값: {y.min():.2f}")
    print(f"   - 최댓값: {y.max():.2f}")

    # === Train-Test Split (핵심!) ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2,      # 테스트 데이터 20%
        random_state=42,    # 재현 가능성을 위한 시드 고정
        stratify=None       # 회귀 문제이므로 stratify 사용 안함
    )
    
    print(f"🔄 데이터 분할:")
    print(f"   - 훈련 데이터: {len(X_train)}개 ({len(X_train)/len(X)*100:.1f}%)")
    print(f"   - 테스트 데이터: {len(X_test)}개 ({len(X_test)/len(X)*100:.1f}%)")

    # === 데이터 스케일링 ===
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # 훈련 데이터로 fit
    X_test_scaled = scaler.transform(X_test)        # 테스트 데이터는 transform만
    
    print("📏 데이터 스케일링 완료 (StandardScaler)")

    # === 모델 1: 선형 회귀 ===
    print("\n🤖 모델 1: 선형 회귀")
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    
    # 예측
    lr_train_pred = lr_model.predict(X_train_scaled)
    lr_test_pred = lr_model.predict(X_test_scaled)
    
    # 성능 평가
    lr_train_rmse = rmse(y_train, lr_train_pred)
    lr_test_rmse = rmse(y_test, lr_test_pred)
    lr_train_r2 = r2_score(y_train, lr_train_pred)
    lr_test_r2 = r2_score(y_test, lr_test_pred)
    
    print(f"   📊 훈련 성능 - RMSE: {lr_train_rmse:.4f}, R²: {lr_train_r2:.4f}")
    print(f"   📊 테스트 성능 - RMSE: {lr_test_rmse:.4f}, R²: {lr_test_r2:.4f}")

    # === 모델 2: XGBoost ===
    print("\n🚀 모델 2: XGBoost")
    xgb_model = XGBRegressor(
        n_estimators=100,       # 트리 개수 증가
        max_depth=6,           # 트리 깊이
        learning_rate=0.1,     # 학습률
        tree_method="hist",    # 빠른 학습을 위한 방법
        random_state=42,       # 재현 가능성
        verbosity=0           # 출력 메시지 최소화
    )
    xgb_model.fit(X_train, y_train)  # XGBoost는 스케일링 불필요
    
    # 예측
    xgb_train_pred = xgb_model.predict(X_train)
    xgb_test_pred = xgb_model.predict(X_test)
    
    # 성능 평가
    xgb_train_rmse = rmse(y_train, xgb_train_pred)
    xgb_test_rmse = rmse(y_test, xgb_test_pred)
    xgb_train_r2 = r2_score(y_train, xgb_train_pred)
    xgb_test_r2 = r2_score(y_test, xgb_test_pred)
    
    print(f"   📊 훈련 성능 - RMSE: {xgb_train_rmse:.4f}, R²: {xgb_train_r2:.4f}")
    print(f"   📊 테스트 성능 - RMSE: {xgb_test_rmse:.4f}, R²: {xgb_test_r2:.4f}")

    # === 모델 비교 ===
    print("\n🏆 모델 성능 비교 (테스트 데이터 기준)")
    print(f"   선형회귀: RMSE = {lr_test_rmse:.4f}, R² = {lr_test_r2:.4f}")
    print(f"   XGBoost:  RMSE = {xgb_test_rmse:.4f}, R² = {xgb_test_r2:.4f}")
    
    # 더 좋은 모델 선택
    if lr_test_rmse < xgb_test_rmse:
        best_model = "선형회귀"
        best_rmse = lr_test_rmse
        best_r2 = lr_test_r2
    else:
        best_model = "XGBoost"
        best_rmse = xgb_test_rmse
        best_r2 = xgb_test_r2
    
    print(f"   🥇 최고 성능: {best_model} (RMSE: {best_rmse:.4f})")

    # === 과적합 검사 ===
    print("\n🔍 과적합 검사")
    lr_overfit = lr_train_rmse - lr_test_rmse
    xgb_overfit = xgb_train_rmse - xgb_test_rmse
    
    print(f"   선형회귀 과적합 정도: {lr_overfit:.4f} (훈련-테스트 RMSE 차이)")
    print(f"   XGBoost 과적합 정도: {xgb_overfit:.4f} (훈련-테스트 RMSE 차이)")
    
    if abs(lr_overfit) > 1.0:
        print("   ⚠️  선형회귀 과적합 의심")
    if abs(xgb_overfit) > 1.0:
        print("   ⚠️  XGBoost 과적합 의심")

    # === 전체 데이터에 대한 예측 (참고용) ===
    print("\n📋 전체 데이터 예측 결과 생성 중...")
    
    # 전체 데이터 스케일링 (훈련 데이터로 fit된 scaler 사용)
    X_all_scaled = scaler.transform(X)
    
    # 전체 데이터 예측
    df_clean["선형회귀_예측"] = lr_model.predict(X_all_scaled)
    df_clean["XGBoost_예측"] = xgb_model.predict(X)
    df_clean["실제값"] = y
    
    # 예측 오차 계산
    df_clean["선형회귀_오차"] = abs(df_clean["실제값"] - df_clean["선형회귀_예측"])
    df_clean["XGBoost_오차"] = abs(df_clean["실제값"] - df_clean["XGBoost_예측"])
    
    # 결과 정리
    result_columns = ["학교명", "기준연도", "실제값", "선형회귀_예측", "XGBoost_예측", 
                     "선형회귀_오차", "XGBoost_오차"]
    result = df_clean[result_columns].sort_values(by=["학교명", "기준연도"])
    
    print("📊 예측 결과 (상위 10개 학교):")
    print(result.head(10).round(3))

    # === 결과 저장 ===
    results_summary.append({
        "데이터셋": label,
        "샘플수": len(df_clean),
        "선형회귀_RMSE": lr_test_rmse,
        "선형회귀_R2": lr_test_r2,
        "XGBoost_RMSE": xgb_test_rmse,
        "XGBoost_R2": xgb_test_r2,
        "최고모델": best_model,
        "최고_RMSE": best_rmse
    })
    
    # CSV 저장 (선택사항)
    # output_filename = f"{label}_예측결과.csv"
    # result.to_csv(output_filename, index=False, encoding="utf-8-sig")
    # print(f"💾 결과 저장: {output_filename}")

# === 전체 결과 요약 ===
print("\n" + "=" * 60)
print("📈 전체 모델링 결과 요약")
print("=" * 60)

summary_df = pd.DataFrame(results_summary)
if not summary_df.empty:
    print(summary_df.round(4))
    
    # 최고 성능 모델 찾기
    best_overall = summary_df.loc[summary_df['최고_RMSE'].idxmin()]
    print(f"\n🏆 전체 최고 성능:")
    print(f"   데이터셋: {best_overall['데이터셋']}")
    print(f"   모델: {best_overall['최고모델']}")
    print(f"   RMSE: {best_overall['최고_RMSE']:.4f}")
    
    # 요약 저장
    # summary_df.to_csv("모델링_결과_요약.csv", index=False, encoding="utf-8-sig")
    # print(f"\n💾 전체 요약 저장: 모델링_결과_요약.csv")

print("\n✅ 모델링 완료!")
print("🎯 주요 개선사항:")
print("   - Train-Test Split으로 올바른 성능 평가")
print("   - RMSE로 직관적인 성능 측정")
print("   - 과적합 검사 추가")
# print("   - 상세한 로그 및 결과 저장")