import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from textblob import TextBlob  # 간단한 감정 분석용

# 1. CSV 불러오기
df = pd.read_csv('covid_vaccine_reviews_3.csv')

# 2. 날짜 컬럼을 datetime 형식으로 변환
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# 3. 연/월 단위로 그룹화
df['year_month'] = df['date'].dt.to_period('M')

# 4. 리뷰 수 집계
monthly_counts = df.groupby('year_month').size()

# 5. 간단한 감정 점수 계산 (polarity: -1~1)
df['polarity'] = df['text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
monthly_sentiment = df.groupby('year_month')['polarity'].mean()

# 6. 시각화
fig, ax1 = plt.subplots(figsize=(12,6))

# 리뷰 수
color = 'tab:blue'
ax1.set_xlabel('Year-Month')
ax1.set_ylabel('Number of Reviews', color=color)
ax1.plot(monthly_counts.index.to_timestamp(), monthly_counts.values, color=color, marker='o', label='Review Count')
ax1.tick_params(axis='y', labelcolor=color)
ax1.xaxis.set_major_formatter(DateFormatter("%Y-%m"))

# 감정 점수 (같은 그래프에 추가)
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Average Sentiment (Polarity)', color=color)
ax2.plot(monthly_sentiment.index.to_timestamp(), monthly_sentiment.values, color=color, marker='x', label='Avg Sentiment')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('COVID-19 Vaccine Reviews: Count & Sentiment Over Time')
fig.tight_layout()
plt.show()
