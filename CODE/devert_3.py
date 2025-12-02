import pandas as pd

# ----------------------------
# 1) 트윗 라벨링 데이터 불러오기
# ----------------------------
df = pd.read_csv("FINAL_DATA_ROWS_labeled#.csv")   # ← 네 실제 파일 이름으로 교체

# 날짜 변환
df["created_at"] = pd.to_datetime(df["created_at"]).dt.date


# ----------------------------
# 2) 공포지수 불러오기
# ----------------------------
fear = pd.read_csv("Fear.csv")

# 컬럼명 확인용
print("Fear.csv columns:", fear.columns)

fear = fear.rename(columns={
    "Date": "date",
    "Fear Greed Index": "fear_index"
})

fear["date"] = pd.to_datetime(fear["date"]).dt.date


# ----------------------------
# 3) 날짜 기준 병합
# ----------------------------
merged = pd.merge(
    df,
    fear,
    left_on="created_at",
    right_on="date",
    how="left"
)

# ----------------------------
# 4) 저장
# ----------------------------
merged.to_csv("tweet_with_fear_index.csv", index=False)
print("병합 완료!")
print(merged.head())
