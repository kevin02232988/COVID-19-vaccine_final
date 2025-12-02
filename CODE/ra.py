import pandas as pd

# 1. CSV 파일 불러오기
df = pd.read_csv("FINAL_DATA_FILTERED_#TRUE.csv")

# 2. 데이터 크기 확인
print(f"전체 데이터 개수: {len(df)}")

# 3. 10% 무작위 추출 (재현 가능하도록 random_state 고정)
sample_df = df.sample(frac=0.1, random_state=42)

# 4. 필요한 열만 선택 ('text'와 'created_at')
sample_df = sample_df[['text', 'created_at']]

# 5. 샘플 데이터 저장
sample_df.to_csv("sampled_for_labeling.csv", index=False, encoding='utf-8-sig')

print(f"추출된 샘플 개수: {len(sample_df)}")
print("✅ 'sampled_for_labeling.csv' 파일 저장 완료!")
