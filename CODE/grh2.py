import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from asd import BytesIO
import base64

# 파일 로드 (BERT 예측 결과 파일)
OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"

try:
    df_final = pd.read_csv(OUTPUT_PREDICTED_FILE)
except FileNotFoundError:
    print(f"[ERROR] 최종 예측 파일 ('{OUTPUT_PREDICTED_FILE}')을 찾을 수 없습니다.")
    exit()

# ------------------- 1. 데이터 클리닝 및 그룹화 -------------------

# 'created_at' 컬럼을 datetime 객체로 변환
df_final['created_at'] = pd.to_datetime(df_final['created_at'], errors='coerce')
# 유효한 날짜가 있는 행만 사용
df_final = df_final.dropna(subset=['created_at'])

# 월별로 그룹화
df_final['year_month'] = df_final['created_at'].dt.to_period('M')

# 부정(Negative) 건수와 전체 건수 계산
sentiment_trend = df_final.groupby('year_month')['Predicted_Sentiment'].value_counts().unstack(fill_value=0)
sentiment_trend['Total'] = sentiment_trend.sum(axis=1)

# 부정 비율 계산
sentiment_trend['Negative_Ratio'] = (sentiment_trend.get('Negative', 0) / sentiment_trend['Total']) * 100

# 인덱스를 datetime 형식으로 변환하여 플로팅 준비
sentiment_trend.index = sentiment_trend.index.to_timestamp()

# ------------------- 2. 도표 생성 (선 그래프) -------------------

# Matplotlib 한글 폰트 설정 (Windows 환경 가정)
try:
    plt.rc('font', family='Malgun Gothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(14, 6))
# Negative Ratio를 시각화합니다.
sns.lineplot(data=sentiment_trend, x=sentiment_trend.index, y='Negative_Ratio', marker='o', color='#E34A33', linewidth=2)

# 그래프 제목과 축 라벨 설정
plt.title('월별 부정 감정 비율 변화 추이 (2019년 4월 ~ 현재)', fontsize=16, pad=15)
plt.xlabel('기간 (월)', fontsize=12)
plt.ylabel('부정적 여론 비율 (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.savefig("monthly_negative_ratio_longterm.png")
plt.close()

print("✅ 시각화 파일 저장 완료: 'monthly_negative_ratio_longterm.png'")