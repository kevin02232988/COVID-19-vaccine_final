import pandas as pd

# CSV 불러오기 (필요한 컬럼만)
df = pd.read_csv('FINAL_DATA_ROWS_#DELETED.csv', usecols=['text', 'created_at'])

# 10% 샘플링
sampled_df = df.sample(frac=0.2, random_state=42)

# UTF-8 BOM 포함하여 저장 (Excel에서 바로 열어도 깨지지 않음)
sampled_df.to_csv('FINAL_DATA_SAMPLE_20_percent_utf8.csv', index=False, encoding='utf-8-sig')

print("샘플링 및 저장 완료!")
