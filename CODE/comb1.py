import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates

# 파일 경로 정의
OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"

try:
    df_final = pd.read_csv(OUTPUT_PREDICTED_FILE)
except FileNotFoundError:
    print(f"[ERROR] 최종 예측 파일 ('{OUTPUT_PREDICTED_FILE}')을 찾을 수 없습니다.")
    exit()

# ------------------- 1. 데이터 클리닝 및 그룹화 -------------------

df_final['created_at'] = pd.to_datetime(df_final['created_at'], errors='coerce')
df_final = df_final.dropna(subset=['created_at'])

df_final['year_month'] = df_final['created_at'].dt.to_period('M')

sentiment_trend = df_final.groupby('year_month')['Predicted_Sentiment'].value_counts().unstack(fill_value=0)
sentiment_trend['Total_Posts'] = sentiment_trend.sum(axis=1)
sentiment_trend['Negative_Ratio'] = (sentiment_trend.get('Negative', 0) / sentiment_trend['Total_Posts']) * 100

# 인덱스를 datetime 형식으로 변환하여 플로팅 준비
sentiment_trend.index = sentiment_trend.index.to_timestamp()

# X축 레이블용 명확한 문자열 컬럼 생성 (오류 방지 핵심)
sentiment_trend['Date_Label'] = sentiment_trend.index.strftime('%Y-%m')


# ------------------- 2. 도표 생성 (이중 축 그래프) -------------------

# Matplotlib 한글 폰트 설정 (Windows 환경 가정)
try:
    plt.rc('font', family='Malgun Gothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

fig, ax1 = plt.subplots(figsize=(14, 6))

# 축 1: 부정 비율 (선 그래프)
color_ratio = '#E34A33' # 빨간색
sns.lineplot(data=sentiment_trend, x='Date_Label', y='Negative_Ratio', ax=ax1, marker='o', color=color_ratio, linewidth=2, label='부정 여론 비율 (%)')
ax1.set_xlabel('기간 (월)', fontsize=12)
ax1.set_ylabel('부정 여론 비율 (%)', color=color_ratio, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_ratio)
ax1.set_ylim(bottom=0, top=105) # y축 하한선을 0으로 고정
ax1.grid(True, linestyle='--', alpha=0.6)


# 축 2: 게시글 규모 (막대 그래프)
ax2 = ax1.twinx()
color_count = '#4393C3' # 파란색
# X축에 명확한 문자열 컬럼 ('Date_Label')을 사용하여 타입 충돌 방지
sns.barplot(x='Date_Label', y='Total_Posts', data=sentiment_trend, ax=ax2, color=color_count, alpha=0.3, label='월별 게시글 수 (규모)')
ax2.set_ylabel('월별 게시글 수 (규모)', color=color_count, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_count)


# X축 눈금 조정 (너무 많아지지 않도록 6개월 간격으로 설정)
tick_spacing = 6
ax1.set_xticks(np.arange(0, len(sentiment_trend), tick_spacing))
ax1.set_xticklabels(sentiment_trend['Date_Label'].iloc[::tick_spacing], rotation=45, ha='right')
ax2.set_xticks(np.arange(0, len(sentiment_trend), tick_spacing))


# 범례 통합
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')


plt.title('온라인 논란의 규모와 강도 변화 추이 (이중 축 분석)', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig("controversy_scale_and_intensity_comparison_final.png")
plt.close()

print("✅ 시각화 파일 저장 완료: 'controversy_scale_and_intensity_comparison_final.png'")