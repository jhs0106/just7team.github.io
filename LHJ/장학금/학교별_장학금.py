# 데이터 불러오기
import pandas as pd

df_raw = pd.read_csv("scholarship_univer.csv")

df_raw.head()

# 컬럼 설정
df_cl = df_raw.copy()
df_cl.columns = df_cl.iloc[0]

# 결측치 제거
df_cl = df_cl[3:]
df_cl = df_cl.iloc[:,:4]

# 컬럼명 변경
df_cl.columns = ['기준연도', '학교명', '재학생', '1인당장학금']

# 인덱스 재설정
df_cl = df_cl.reset_index(drop = True)

# 비어있는 값에 이전 값을 삽입
df_cl['기준연도'] = df_cl['기준연도'].fillna(method='ffill')

# 쉼표(,) 제거
df_cl['재학생']= df_cl['재학생'].str.replace(',', '', regex=False)
df_cl['1인당장학금']= df_cl['1인당장학금'].str.replace(',', '', regex=False)

# 타입 변경
df_cl['기준연도'] = df_cl['기준연도'].astype(int)
df_cl['재학생'] = df_cl['재학생'].astype(int)
df_cl['1인당장학금'] = df_cl['1인당장학금'].astype(float)

#데이터프레임을 csv 파일로 저장
df_cl.to_csv('cleaned_scholarship_univer.csv', index = False, encoding = 'utf-8-sig')