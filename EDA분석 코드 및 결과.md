# ===============================================
# 계획서 기술개발 내용 기반 EDA 분석
# ===============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, chi2_contingency, f_oneway
import warnings
warnings.filterwarnings('ignore')

print(" 대학 중도탈락률 예측 프로젝트 - EDA 분석")
print("=" * 60)

# 1. 데이터 로딩 및 기본 정보
df_original = pd.read_csv('최종 전처리 파일(final).csv', encoding='utf-8')
df = df_original.copy()

print(" 데이터 로딩 성공!")
print(" 원본 데이터 기본 정보")
print("데이터 크기:", df.shape)
print("기간:", df['기준년도'].min(), "~", df['기준년도'].max())

# 변수 분류 (계획서 요구사항)
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if '기준년도' in numerical_cols:
    numerical_cols.remove('기준년도')
    categorical_cols.append('기준년도')

print("수치형 변수:", len(numerical_cols), "개")
print("범주형 변수:", len(categorical_cols), "개")

# 2. 이상치 및 결측치 처리 (계획서 개선 방식)
print("\n 이상치 및 결측치 처리")
print("=" * 30)

def handle_missing_and_outliers(df):
    """계획서 개선 방식: percentile 캡핑 + 중앙값/최빈값 대체"""
    df_clean = df.copy()
    
    missing_summary = df_clean.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0]
    print("결측치가 있는 변수:")
    for col, count in missing_summary.items():
        pct = count / len(df_clean) * 100
        print(f"  {col}: {count}개 ({pct:.1f}%)")
    
    # 이상치 처리 (99% percentile 캡핑)
    print("\n이상치 처리 (99% percentile 캡핑):")
    outlier_count = 0
    for col in numerical_cols:
        if col in df_clean.columns:
            upper_cap = df_clean[col].quantile(0.99)
            lower_cap = df_clean[col].quantile(0.01)
            
            extreme_count = ((df_clean[col] > upper_cap) | (df_clean[col] < lower_cap)).sum()
            
            if extreme_count > 0:
                df_clean[col] = df_clean[col].clip(lower=lower_cap, upper=upper_cap)
                print(f"  {col}: {extreme_count}개 조정")
                outlier_count += extreme_count
    
    # 결측치 처리
    print("\n결측치 처리:")
    # 수치형: 중앙값으로 대체
    for col in numerical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            missing_count = df_clean[col].isnull().sum()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  {col}: {missing_count}개 중앙값({median_val:.2f})으로 대체")
    
    # 범주형: 최빈값으로 대체
    for col in categorical_cols:
        if col in df_clean.columns and df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else '기타'
            missing_count = df_clean[col].isnull().sum()
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"  {col}: {missing_count}개 최빈값({mode_val})으로 대체")
    
    print(f" 총 이상치 {outlier_count}개 조정 및 결측치 처리 완료")
    return df_clean

df_clean = handle_missing_and_outliers(df)

# 3. 탐색적 데이터 분석 (EDA) - 계획서 요구사항
print("\n 탐색적 데이터 분석 (EDA)")
print("=" * 40)

# 3-1. 수치형 피처의 분포 확인 (히스토그램, 박스플롯)
print("3-1. 수치형 피처 분포 확인")

# 중도탈락률 기본 통계
dropout_rates = df_clean['중도탈락률(%)'].dropna()
print(f"\n중도탈락률(%) 기본 통계:")
print(f"  개수: {len(dropout_rates)}")
print(f"  평균: {dropout_rates.mean():.2f}%")
print(f"  중앙값: {dropout_rates.median():.2f}%")
print(f"  표준편차: {dropout_rates.std():.2f}%")
print(f"  최소값: {dropout_rates.min():.2f}%")
print(f"  최대값: {dropout_rates.max():.2f}%")

# 주요 수치형 변수들의 분포 시각화
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

key_numeric_vars = ['중도탈락률(%)', '취업률(%)', '1인당장학금', '신입생_충원율(%)', '전임교원1인당재학생', '등록금']

for i, var in enumerate(key_numeric_vars):
    if var in df_clean.columns and i < 6:
        # 히스토그램
        axes[i].hist(df_clean[var].dropna(), bins=30, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{var} Distribution')
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('주요 수치형 변수 분포', y=1.02, fontsize=16)
plt.show()

# 박스플롯으로 이상치 확인
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.ravel()

for i, var in enumerate(key_numeric_vars):
    if var in df_clean.columns and i < 6:
        axes[i].boxplot(df_clean[var].dropna())
        axes[i].set_title(f'{var} Boxplot')
        axes[i].set_ylabel(var)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('주요 수치형 변수 박스플롯', y=1.02, fontsize=16)
plt.show()

# 3-2. 기준년도별 중도탈락률 변화 추이 분석 (선 그래프)
print("\n3-2. 기준년도별 중도탈락률 변화 추이")

yearly_stats = df_clean.groupby('기준년도')['중도탈락률(%)'].agg(['mean', 'median', 'std', 'count']).round(2)
print("연도별 중도탈락률 통계:")
print(yearly_stats)

# 트렌드 계산
years = yearly_stats.index.values
means = yearly_stats['mean'].values
slope = np.polyfit(years, means, 1)[0]
print(f"\n 연평균 증가율: {slope:.3f}%p/년")

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 연도별 추이 (선 그래프)
axes[0].plot(yearly_stats.index, yearly_stats['mean'], 'o-', linewidth=3, markersize=8, label='평균')
axes[0].plot(yearly_stats.index, yearly_stats['median'], 's-', linewidth=2, markersize=6, label='중앙값')
axes[0].fill_between(yearly_stats.index, 
                     yearly_stats['mean'] - yearly_stats['std'], 
                     yearly_stats['mean'] + yearly_stats['std'], 
                     alpha=0.3, label='±1 표준편차')
axes[0].set_title('기준년도별 중도탈락률 변화 추이')
axes[0].set_xlabel('기준년도')
axes[0].set_ylabel('중도탈락률 (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 연도별 분포 (박스플롯)
df_clean.boxplot(column='중도탈락률(%)', by='기준년도', ax=axes[1])
axes[1].set_title('연도별 중도탈락률 분포')
axes[1].set_xlabel('기준년도')
axes[1].set_ylabel('중도탈락률 (%)')

plt.tight_layout()
plt.show()

# 3-3. 피처 간 상관관계 분석 (상관계수 히트맵)
print("\n3-3. 피처 간 상관관계 분석")

# 주요 수치형 변수들의 상관관계
correlation_vars = ['중도탈락률(%)', '신입생_충원율(%)', '신입생_경쟁률', 
                   '신입생_충원율_평균_지역기준', '신입생_충원율_표준편차_지역기준',
                   '신입생_충원안정성비율_지역기준', '전임교원1인당재학생',
                   '취업률(%)', '진로진출률(%)', '졸업자대비진로성취(%)',
                   '재학생1인당전임교원', '1인당장학금', '진학률(%)', '등록금',
                   '신입생_충원율_평균_수도권여부기준', '신입생_충원율_편차_수도권여부기준',
                   '신입생_충원율_표준편차_수도권여부기준', '신입생_충원안정성비율_수도권여부기준',
                   '재학생충원율_평균_지역기준_', '재학생충원율_표준편차_지역기준_',
                   '재학생충원율_편차_지역기준', '재학생충원안정성비율_지역기준',
                   '재학생충원율_평균_수도권여부기준', '재학생충원율_표준편차_수도권여부기준',
                   '재학생충원율_편차_수도권여부기준', '재학생충원안정성비율_수도권여부기준',
                   '재학생충원율', '신입생_충원율_편차_지역구분', '1인당장학금대비취업률(단위보정)']

available_corr_vars = [var for var in correlation_vars if var in df_clean.columns]
corr_matrix = df_clean[available_corr_vars].corr()

# 상관계수 히트맵
plt.figure(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0, 
            square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('주요 변수 간 상관관계 히트맵')
plt.tight_layout()
plt.show()

# 중도탈락률과 다른 변수들의 상관관계
dropout_corr = corr_matrix['중도탈락률(%)'].drop('중도탈락률(%)')
print("중도탈락률과의 상관관계 (절댓값 순):")
for var, corr in dropout_corr.abs().sort_values(ascending=False).items():
    original_corr = dropout_corr[var]
    print(f"  {var}: {original_corr:.3f}")

# 3-4. 중도탈락률과 각 변수 간 산점도 분석
print("\n3-4. 중도탈락률과 주요 변수 간 관계 분석")

# 주요 변수들과의 산점도
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

scatter_vars = ['취업률(%)', '1인당장학금', '신입생_충원율(%)', '등록금', 
               '전임교원1인당재학생', '진학률(%)', '재학생1인당전임교원',
               '신입생_충원안정성비율_지역기준', '재학생충원안정성비율_지역기준',
               '1인당장학금대비취업률(단위보정)', '졸업자대비진로성취(%)', '재학생충원율']

for i, var in enumerate(scatter_vars):
    if var in df_clean.columns and i < 6:
        # 산점도
        axes[i].scatter(df_clean[var], df_clean['중도탈락률(%)'], alpha=0.5)
        
        # 회귀선 추가
        if df_clean[var].notna().sum() > 10:
            z = np.polyfit(df_clean[var].dropna(), 
                          df_clean.loc[df_clean[var].notna(), '중도탈락률(%)'], 1)
            p = np.poly1d(z)
            axes[i].plot(df_clean[var].dropna().sort_values(), 
                        p(df_clean[var].dropna().sort_values()), "r--", alpha=0.8)
        
        axes[i].set_xlabel(var)
        axes[i].set_ylabel('중도탈락률 (%)')
        axes[i].set_title(f'중도탈락률 vs {var}')
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('중도탈락률과 주요 변수 간 관계', y=1.02, fontsize=16)
plt.show()

# 4. 범주형 변수 분석
print("\n 범주형 변수별 분석")
print("=" * 30)

# 4-1. 대계열별 분석
print("4-1. 대계열별 중도탈락률 분석")

major_stats = df_clean.groupby('대계열')['중도탈락률(%)'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
major_stats = major_stats.sort_values('mean', ascending=False)
major_stats['위험도'] = major_stats['mean'] / df_clean['중도탈락률(%)'].mean()

print("대계열별 중도탈락률 통계:")
print(major_stats)

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 계열별 평균 (막대그래프)
major_means = major_stats['mean'].sort_values()
axes[0].barh(range(len(major_means)), major_means.values, color='skyblue', edgecolor='navy')
axes[0].set_yticks(range(len(major_means)))
axes[0].set_yticklabels(major_means.index)
axes[0].set_title('계열별 평균 중도탈락률')
axes[0].set_xlabel('평균 중도탈락률 (%)')
axes[0].grid(True, alpha=0.3)

# 계열별 분포 (박스플롯)
df_clean.boxplot(column='중도탈락률(%)', by='대계열', ax=axes[1])
axes[1].set_title('계열별 중도탈락률 분포')
axes[1].set_xlabel('대계열')
axes[1].set_ylabel('중도탈락률 (%)')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# 4-2. 설립구분별 분석
print("\n4-2. 설립구분별 중도탈락률 분석")

establishment_stats = df_clean.groupby('설립구분')['중도탈락률(%)'].agg(['count', 'mean', 'std']).round(2)
establishment_stats = establishment_stats.sort_values('mean', ascending=False)

print("설립구분별 중도탈락률 통계:")
print(establishment_stats)

# 4-3. 수도권 vs 비수도권 분석
print("\n4-3. 수도권 vs 비수도권 분석")

region_stats = df_clean.groupby('수도권여부')['중도탈락률(%)'].agg(['count', 'mean', 'std']).round(2)
print("지역별 중도탈락률 통계:")
print(region_stats)

# 설립구분 + 수도권 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 설립구분별
establishment_stats['mean'].plot(kind='bar', ax=axes[0], color='lightcoral', edgecolor='darkred')
axes[0].set_title('설립구분별 평균 중도탈락률')
axes[0].set_ylabel('평균 중도탈락률 (%)')
axes[0].tick_params(axis='x', rotation=45)
axes[0].grid(True, alpha=0.3)

# 수도권 여부별
region_stats['mean'].plot(kind='bar', ax=axes[1], color='lightgreen', edgecolor='darkgreen')
axes[1].set_title('지역별 평균 중도탈락률')
axes[1].set_ylabel('평균 중도탈락률 (%)')
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\n4-4. 지역별 중도탈락률 분석")

region_detail_stats = df_clean.groupby('지역')['중도탈락률(%)'].agg(['count', 'mean', 'std']).round(2)
region_detail_stats = region_detail_stats.sort_values('mean', ascending=False)

print("지역별 중도탈락률 통계 (상위 10개):")
print(region_detail_stats.head(10))

# 지역별 시각화
plt.figure(figsize=(14, 8))
top_regions = region_detail_stats.head(10)
plt.barh(range(len(top_regions)), top_regions['mean'], color='orange', edgecolor='red')
plt.yticks(range(len(top_regions)), top_regions.index)
plt.title('지역별 평균 중도탈락률 (상위 10개)')
plt.xlabel('평균 중도탈락률 (%)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. 연도별/유형별 비교 시각화
print("\n📈 연도별/유형별 비교 분석")
print("=" * 30)

# 5-1. 연도별 계열 분석
print("5-1. 연도별 계열별 변화")

# 계열별 연도 추이
major_yearly = df_clean.groupby(['기준년도', '대계열'])['중도탈락률(%)'].mean().unstack()

plt.figure(figsize=(14, 8))
for major in major_yearly.columns:
    plt.plot(major_yearly.index, major_yearly[major], marker='o', linewidth=2, label=major)

plt.title('연도별 계열별 중도탈락률 변화')
plt.xlabel('기준년도')
plt.ylabel('평균 중도탈락률 (%)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5-2. 연도별 설립구분 분석
print("\n5-2. 연도별 설립구분별 변화")

establishment_yearly = df_clean.groupby(['기준년도', '설립구분'])['중도탈락률(%)'].mean().unstack()

plt.figure(figsize=(12, 6))
for est in establishment_yearly.columns:
    plt.plot(establishment_yearly.index, establishment_yearly[est], marker='s', linewidth=3, label=est)

plt.title('연도별 설립구분별 중도탈락률 변화')
plt.xlabel('기준년도')
plt.ylabel('평균 중도탈락률 (%)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. 통계적 검정 - 계획서 요구사항
print("\n 통계적 검정")
print("=" * 30)

# 6-1. 유의성 검정 (상관관계)
print("6-1. 피어슨 상관관계 유의성 검정:")
for col in available_corr_vars:
    if col != '중도탈락률(%)':
        valid_data = df_clean[['중도탈락률(%)', col]].dropna()
        if len(valid_data) > 30:
            corr_coef, p_value = pearsonr(valid_data['중도탈락률(%)'], valid_data[col])
            sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
            print(f"  {col}: r={corr_coef:.3f}, p={p_value:.4f} {sig}")

# 6-2. ANOVA (계열별 차이)
print("\n6-2. 계열별 평균 차이 검정 (ANOVA):")
groups = [group['중도탈락률(%)'].dropna() for name, group in df_clean.groupby('대계열')]
f_stat, p_value = f_oneway(*groups)
print(f"F-statistic: {f_stat:.2f}, p-value: {p_value:.4f}")
if p_value < 0.05:
    print(" 계열별로 유의미한 차이가 있습니다.")
else:
    print(" 계열별 차이가 유의미하지 않습니다.")

# 6-3. 카이제곱 검정 (고위험군 vs 계열)
print("\n6-3. 고위험군과 계열 간 독립성 검정:")
df_clean['고위험군'] = df_clean['중도탈락률(%)'] >= df_clean['중도탈락률(%)'].quantile(0.9)
contingency = pd.crosstab(df_clean['고위험군'], df_clean['대계열'])
chi2, p_chi, dof, expected = chi2_contingency(contingency)
print(f"Chi-square: {chi2:.2f}, p-value: {p_chi:.4f}")
if p_chi < 0.05:
    print(" 고위험군과 계열 간 유의미한 관련성이 있습니다.")
else:
    print(" 고위험군과 계열 간 관련성이 유의미하지 않습니다.")

# 7. EDA 결과 요약 및 모델링 준비
print("\n EDA 결과 요약")
print("=" * 30)

# 주요 발견사항 정리
key_findings = {
    "기본정보": {
        "총_데이터수": len(df_clean),
        "분석기간": f"{df_clean['기준년도'].min()}-{df_clean['기준년도'].max()}",
        "평균_중도탈락률": f"{df_clean['중도탈락률(%)'].mean():.2f}%",
        "연평균_증가율": f"{slope:.3f}%p/년"
    },
    "계열별_위험도": {
        major: f"{stats['mean']:.2f}%" 
        for major, stats in major_stats.iterrows()
    },
    "주요_상관관계": {
        var: f"{corr:.3f}" 
        for var, corr in dropout_corr.abs().sort_values(ascending=False).head(5).items()
    }
}

# 결과 저장
import json
with open('eda_results_summary.json', 'w', encoding='utf-8') as f:
    json.dump(key_findings, f, ensure_ascii=False, indent=2)

# 전처리된 데이터 저장
df_clean.to_csv('eda_preprocessed_data.csv', index=False, encoding='utf-8-sig')

print(" EDA 분석 완료!")
print("=" * 40)
print("주요 결과:")
print(f"  - 연평균 중도탈락률 증가: {slope:.3f}%p/년")
print(f"  - 최고위험 계열: {major_stats.index[0]} ({major_stats.iloc[0]['mean']:.2f}%)")
print(f"  - 최강 예측변수: {dropout_corr.abs().idxmax()} (r={dropout_corr[dropout_corr.abs().idxmax()]:.3f})")

print("\n 생성된 파일:")
print("   eda_preprocessed_data.csv (전처리된 데이터)")
print("   eda_results_summary.json (EDA 결과 요약)")

print("\n 다음 단계:")
print("  → 이 전처리된 데이터로 모델링 진행")
print("  → Feature Importance 및 SHAP 분석")
print("  → 정책 제언 도출")

print("\n 계획서 요구사항 기반 EDA 완료!")


//결과
 대학 중도탈락률 예측 프로젝트 - EDA 분석
 계획서 기술개발 내용 요구사항 기반
============================================================
 데이터 로딩 성공!
 원본 데이터 기본 정보
데이터 크기: (7480, 35)
기간: 2014 ~ 2023
수치형 변수: 29 개
범주형 변수: 6 개

 이상치 및 결측치 처리
==============================
결측치가 있는 변수:
  신입생_충원율(%): 662개 (8.9%)
  신입생_경쟁률: 662개 (8.9%)
  신입생_충원율_평균_지역기준: 662개 (8.9%)
  신입생_충원율_표준편차_지역기준: 662개 (8.9%)
  신입생_충원안정성비율_지역기준: 662개 (8.9%)
  전임교원1인당재학생: 436개 (5.8%)
  취업률(%): 878개 (11.7%)
  진로진출률(%): 878개 (11.7%)
  졸업자대비진로성취(%): 869개 (11.6%)
  재학생1인당전임교원: 436개 (5.8%)
  1인당장학금: 102개 (1.4%)
  진학률(%): 222개 (3.0%)
  등록금: 190개 (2.5%)
  신입생_충원율_평균_수도권여부기준: 662개 (8.9%)
  신입생_충원율_편차_수도권여부기준: 662개 (8.9%)
  신입생_충원율_표준편차_수도권여부기준: 662개 (8.9%)
  신입생_충원안정성비율_수도권여부기준: 662개 (8.9%)
  재학생충원율_평균_지역기준_: 389개 (5.2%)
  재학생충원율_표준편차_지역기준_: 1080개 (14.4%)
  재학생충원율_편차_지역기준: 389개 (5.2%)
  재학생충원안정성비율_지역기준: 1080개 (14.4%)
  재학생충원율_평균_수도권여부기준: 389개 (5.2%)
  재학생충원율_표준편차_수도권여부기준: 389개 (5.2%)
  재학생충원율_편차_수도권여부기준: 389개 (5.2%)
  재학생충원안정성비율_수도권여부기준: 389개 (5.2%)
  재학생충원율: 389개 (5.2%)
  신입생_충원율_편차_지역구분: 662개 (8.9%)
  1인당장학금대비취업률(단위보정): 951개 (12.7%)

이상치 처리 (99% percentile 캡핑):
  중도탈락률(%): 74개 조정
  신입생_충원율(%): 69개 조정
  신입생_경쟁률: 138개 조정
  신입생_충원율_평균_지역기준: 136개 조정
  신입생_충원율_표준편차_지역기준: 69개 조정
  신입생_충원안정성비율_지역기준: 64개 조정
  전임교원1인당재학생: 141개 조정
  취업률(%): 65개 조정
  진로진출률(%): 134개 조정
  졸업자대비진로성취(%): 133개 조정
  재학생1인당전임교원: 142개 조정
  1인당장학금: 148개 조정
  진학률(%): 73개 조정
  등록금: 141개 조정
  신입생_충원율_편차_수도권여부기준: 96개 조정
  신입생_충원안정성비율_수도권여부기준: 132개 조정
  재학생충원율_평균_지역기준_: 127개 조정
  재학생충원율_표준편차_지역기준_: 126개 조정
  재학생충원율_편차_지역기준: 142개 조정
  재학생충원안정성비율_지역기준: 128개 조정
  재학생충원율_편차_수도권여부기준: 139개 조정
  재학생충원안정성비율_수도권여부기준: 137개 조정
  재학생충원율: 71개 조정
  신입생_충원율_편차_지역구분: 138개 조정
  1인당장학금대비취업률(단위보정): 66개 조정

결측치 처리:
  신입생_충원율(%): 662개 중앙값(99.74)으로 대체
  신입생_경쟁률: 662개 중앙값(6.71)으로 대체
  신입생_충원율_평균_지역기준: 662개 중앙값(97.24)으로 대체
  신입생_충원율_표준편차_지역기준: 662개 중앙값(8.71)으로 대체
  신입생_충원안정성비율_지역기준: 662개 중앙값(0.09)으로 대체
  전임교원1인당재학생: 436개 중앙값(29.16)으로 대체
  취업률(%): 878개 중앙값(62.53)으로 대체
  진로진출률(%): 878개 중앙값(68.58)으로 대체
  졸업자대비진로성취(%): 869개 중앙값(63.04)으로 대체
  재학생1인당전임교원: 436개 중앙값(0.03)으로 대체
  1인당장학금: 102개 중앙값(2982240.03)으로 대체
  진학률(%): 222개 중앙값(4.12)으로 대체
  등록금: 190개 중앙값(7599484.95)으로 대체
  신입생_충원율_평균_수도권여부기준: 662개 중앙값(95.74)으로 대체
  신입생_충원율_편차_수도권여부기준: 662개 중앙값(3.82)으로 대체
  신입생_충원율_표준편차_수도권여부기준: 662개 중앙값(15.80)으로 대체
  신입생_충원안정성비율_수도권여부기준: 662개 중앙값(0.26)으로 대체
  재학생충원율_평균_지역기준_: 389개 중앙값(102.30)으로 대체
  재학생충원율_표준편차_지역기준_: 1080개 중앙값(14.89)으로 대체
  재학생충원율_편차_지역기준: 389개 중앙값(0.00)으로 대체
  재학생충원안정성비율_지역기준: 1080개 중앙값(0.65)으로 대체
  재학생충원율_평균_수도권여부기준: 389개 중앙값(100.92)으로 대체
  재학생충원율_표준편차_수도권여부기준: 389개 중앙값(23.59)으로 대체
  재학생충원율_편차_수도권여부기준: 389개 중앙값(2.08)으로 대체
  재학생충원안정성비율_수도권여부기준: 389개 중앙값(0.37)으로 대체
  재학생충원율: 389개 중앙값(103.30)으로 대체
  신입생_충원율_편차_지역구분: 662개 중앙값(1.11)으로 대체
  1인당장학금대비취업률(단위보정): 951개 중앙값(20.79)으로 대체
 총 이상치 2829개 조정 및 결측치 처리 완료

 탐색적 데이터 분석 (EDA)
========================================
3-1. 수치형 피처 분포 확인

중도탈락률(%) 기본 통계:
  개수: 7480
  평균: 5.33%
  중앙값: 4.64%
  표준편차: 3.90%
  최소값: 0.00%
  최대값: 22.86%



3-2. 기준년도별 중도탈락률 변화 추이
연도별 중도탈락률 통계:
      mean  median   std  count
기준년도                           
2014  4.64    4.01  3.49    748
2015  4.71    4.10  3.52    748
2016  4.80    4.28  3.44    748
2017  4.94    4.40  3.52    748
2018  5.28    4.76  3.71    748
2019  5.39    4.78  3.76    748
2020  5.28    4.71  3.74    748
2021  5.79    4.94  4.15    748
2022  6.10    5.06  4.38    748
2023  6.42    5.22  4.68    748

연평균 증가율: 0.193%p/년


3-3. 피처 간 상관관계 분석

중도탈락률과의 상관관계 (절댓값 순):
  재학생충원율: -0.416
  재학생충원율_편차_수도권여부기준: -0.382
  재학생충원율_평균_지역기준_: -0.357
  신입생_경쟁률: -0.351
  신입생_충원율(%): -0.296
  졸업자대비진로성취(%): -0.274
  신입생_충원율_편차_수도권여부기준: -0.271
  진로진출률(%): -0.268
  재학생충원율_편차_지역기준: -0.266
  신입생_충원안정성비율_수도권여부기준: 0.265
  신입생_충원율_평균_지역기준: -0.255
  재학생충원율_평균_수도권여부기준: -0.213
  신입생_충원율_평균_수도권여부기준: -0.200
  재학생충원안정성비율_수도권여부기준: 0.198
  취업률(%): -0.188
  신입생_충원율_편차_지역구분: -0.175
  신입생_충원안정성비율_지역기준: 0.152
  신입생_충원율_표준편차_수도권여부기준: 0.152
  신입생_충원율_표준편차_지역기준: 0.151
  재학생1인당전임교원: -0.147
  재학생충원안정성비율_지역기준: 0.122
  진학률(%): -0.122
  등록금: -0.120
  1인당장학금대비취업률(단위보정): -0.107
  재학생충원율_표준편차_지역기준_: 0.102
  전임교원1인당재학생: 0.042
  1인당장학금: 0.011
  재학생충원율_표준편차_수도권여부기준: 0.004

3-4. 중도탈락률과 주요 변수 간 관계 분석


 범주형 변수별 분석
==============================
4-1. 대계열별 중도탈락률 분석
대계열별 중도탈락률 통계:
        count  mean   std  min    max       위험도
대계열                                            
예체능계열    1690  6.68  4.48  0.0  22.86  1.252219
인문사회계열   2100  5.74  3.90  0.0  22.86  1.076008
공학계열     1610  5.64  3.46  0.0  22.86  1.057263
자연과학계열   1630  4.46  2.73  0.0  22.86  0.836062
의학계열      450  0.47  1.47  0.0  22.86  0.088105


4-2. 설립구분별 중도탈락률 분석
설립구분별 중도탈락률 통계:
       count  mean   std
설립구분                    
사립      5830  5.68  3.98
국립      1380  4.39  3.41
특별법법인    110  2.92  3.26
공립        40  2.86  0.97
국립대법인     90  2.49  1.61
특별법국립     30  2.32  0.96

4-3. 수도권 vs 비수도권 분석
지역별 중도탈락률 통계:
       count  mean   std
수도권여부                   
비수도권    4700  5.88  4.11
수도권     2780  4.42  3.32


4-4. 지역별 중도탈락률 분석
지역별 중도탈락률 통계 (상위 10개):
    count  mean   std
지역                   
제주     90  9.62  6.95
전남    310  7.47  4.26
경북    630  7.29  5.27
광주    380  6.66  4.35
전북    350  6.51  4.17
충북    470  6.00  4.82
경남    300  5.68  3.15
충남    540  5.19  2.37
경기   1120  5.19  3.06
강원    460  5.18  3.86


 연도별/유형별 비교 분석
==============================
5-1. 연도별 계열별 변화


5-2. 연도별 설립구분별 변화

통계적 검정
==============================
6-1. 피어슨 상관관계 유의성 검정:
  신입생_충원율(%): r=-0.296, p=0.0000 ***
  신입생_경쟁률: r=-0.351, p=0.0000 ***
  신입생_충원율_평균_지역기준: r=-0.255, p=0.0000 ***
  신입생_충원율_표준편차_지역기준: r=0.151, p=0.0000 ***
  신입생_충원안정성비율_지역기준: r=0.152, p=0.0000 ***
  전임교원1인당재학생: r=0.042, p=0.0003 ***
  취업률(%): r=-0.188, p=0.0000 ***
  진로진출률(%): r=-0.268, p=0.0000 ***
  졸업자대비진로성취(%): r=-0.274, p=0.0000 ***
  재학생1인당전임교원: r=-0.147, p=0.0000 ***
  1인당장학금: r=0.011, p=0.3385 
  진학률(%): r=-0.122, p=0.0000 ***
  등록금: r=-0.120, p=0.0000 ***
  신입생_충원율_평균_수도권여부기준: r=-0.200, p=0.0000 ***
  신입생_충원율_편차_수도권여부기준: r=-0.271, p=0.0000 ***
  신입생_충원율_표준편차_수도권여부기준: r=0.152, p=0.0000 ***
  신입생_충원안정성비율_수도권여부기준: r=0.265, p=0.0000 ***
  재학생충원율_평균_지역기준_: r=-0.357, p=0.0000 ***
  재학생충원율_표준편차_지역기준_: r=0.102, p=0.0000 ***
  재학생충원율_편차_지역기준: r=-0.266, p=0.0000 ***
  재학생충원안정성비율_지역기준: r=0.122, p=0.0000 ***
  재학생충원율_평균_수도권여부기준: r=-0.213, p=0.0000 ***
  재학생충원율_표준편차_수도권여부기준: r=0.004, p=0.7591 
  재학생충원율_편차_수도권여부기준: r=-0.382, p=0.0000 ***
  재학생충원안정성비율_수도권여부기준: r=0.198, p=0.0000 ***
  재학생충원율: r=-0.416, p=0.0000 ***
  신입생_충원율_편차_지역구분: r=-0.175, p=0.0000 ***
  1인당장학금대비취업률(단위보정): r=-0.107, p=0.0000 ***

6-2. 계열별 평균 차이 검정 (ANOVA):
F-statistic: 293.95, p-value: 0.0000
 계열별로 유의미한 차이가 있습니다.

6-3. 고위험군과 계열 간 독립성 검정:
Chi-square: 281.60, p-value: 0.0000
고위험군과 계열 간 유의미한 관련성이 있습니다.

 EDA 결과 요약
==============================
# EDA 분석 완료!
========================================
 주요 결과:
  - 연평균 중도탈락률 증가: 0.193%p/년
  - 최고위험 계열: 예체능계열 (6.68%)
  - 최강 예측변수: 재학생충원율 (r=-0.416)

생성된 파일:
   eda_preprocessed_data.csv (전처리된 데이터)
   eda_results_summary.json (EDA 결과 요약)

 다음 단계:
  → 이 전처리된 데이터로 모델링 진행
  → Feature Importance 및 SHAP 분석
  → 정책 제언 도출

 계획서 요구사항 기반 EDA 완료!