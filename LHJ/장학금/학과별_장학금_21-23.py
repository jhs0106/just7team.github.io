import pandas as pd

df = pd.read_csv("통합_학과별_장학금_21-23.csv")
df = df.copy()

# 컬럼 설정, 필요 데이터만 남기기
df = df.iloc[:,:8]
df.columns = df.iloc[0]
df = df.iloc[4:,:]
df = df.reset_index(drop = True)

df.columns = ['기준연도', '학교명', '단과대학', '학과', '구분', '학기', '재학생', '1인당장학금']

# 비어있는 값 이전 행 값 삽입
df['기준연도'] = df['기준연도'].fillna(method='ffill')
df['학교명'] = df['학교명'].fillna(method='ffill')
df['단과대학'] = df['단과대학'].fillna(method='ffill')
df['학과'] = df['학과'].fillna(method='ffill')
df['구분'] = df['구분'].fillna(method='ffill')

# 쉼표 제거
df['재학생']= df['재학생'].str.replace(',', '', regex=False)
df['1인당장학금'] = df['1인당장학금'].astype(str).str.replace(',', '', regex=False) # 문자열 변환후 제거

df['기준연도'] = df['기준연도'].astype(int)
df['재학생'] = df['재학생'].astype(int)
df['1인당장학금'] = df['1인당장학금'].astype(float)

# 재학생이 10 이하인 행 삭제
df = df[~((df['재학생'] >= 0) & (df['재학생'] <= 10))]

df.to_csv('학과별_장학금_21-23.csv', index = False, encoding = 'utf-8-sig')