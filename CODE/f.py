import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
import re
from matplotlib.ticker import FuncFormatter

# 파일 로드 (BERT 예측 결과 파일)
OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"
try:
    df_final = pd.read_csv(OUTPUT_PREDICTED_FILE)
except FileNotFoundError:
    print(f"[ERROR] 최종 예측 파일 ('{OUTPUT_PREDICTED_FILE}')을 찾을 수 없습니다. 분석 파일이 존재하는지 확인해 주세요.")
    exit()

# ------------------- 1. Matplotlib 한글 폰트 설정 (Windows 환경) -------------------
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False
except:
    print("[WARNING] Malgun Gothic 폰트 설정 실패. 그래프 제목이 깨질 수 있습니다.")

# ------------------- 2. 도표 2번: 논란 규모와 강도 결합 도표 -------------------

# 데이터 클리닝 및 그룹화
df_final['created_at'] = pd.to_datetime(df_final['created_at'], errors='coerce')
df_final = df_final.dropna(subset=['created_at'])
df_final['year_month'] = df_final['created_at'].dt.to_period('M')

sentiment_trend = df_final.groupby('year_month')['Predicted_Sentiment'].value_counts().unstack(fill_value=0)
sentiment_trend['Total_Posts'] = sentiment_trend.sum(axis=1)
sentiment_trend['Negative_Ratio'] = (sentiment_trend.get('Negative', 0) / sentiment_trend['Total_Posts']) * 100
sentiment_trend.index = sentiment_trend.index.to_timestamp()
sentiment_trend['Date_Label'] = sentiment_trend.index.strftime('%Y-%m')

fig, ax1 = plt.subplots(figsize=(14, 6))

# 축 1: 부정 비율 (선 그래프 - 강도)
color_ratio = '#E34A33'
sns.lineplot(data=sentiment_trend, x='Date_Label', y='Negative_Ratio', ax=ax1, marker='o', color=color_ratio, linewidth=2, label='부정 여론 비율 (%)')
ax1.set_xlabel('기간 (월)', fontsize=12)
ax1.set_ylabel('부정 여론 비율 (%)', color=color_ratio, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color_ratio)
ax1.set_ylim(bottom=60, top=105)
ax1.grid(True, linestyle='--', alpha=0.6)

# 축 2: 게시글 규모 (막대 그래프 - 규모)
ax2 = ax1.twinx()
color_count = '#4393C3'
sns.barplot(x='Date_Label', y='Total_Posts', data=sentiment_trend, ax=ax2, color=color_count, alpha=0.3, label='월별 게시글 수 (규모)')
ax2.set_ylabel('월별 게시글 수 (규모)', color=color_count, fontsize=12)
ax2.tick_params(axis='y', labelcolor=color_count)

# X축 눈금 조정 (6개월 간격)
tick_spacing = 6
ax1.set_xticks(np.arange(0, len(sentiment_trend), tick_spacing))
ax1.set_xticklabels(sentiment_trend['Date_Label'].iloc[::tick_spacing], rotation=45, ha='right')
ax2.set_xticks(np.arange(0, len(sentiment_trend), tick_spacing))

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper left')

plt.title('2. 온라인 논란의 규모와 강도 변화 추이 (이중 축 분석)', fontsize=16, pad=15)
plt.tight_layout()
plt.savefig("controversy_scale_and_intensity_comparison_final.png")
plt.close()
print("✅ 도표 2 저장 완료: 'controversy_scale_and_intensity_comparison_final.png'")


# ------------------- 3. 도표 3번: 감정별 토픽 비교 도표 -------------------

# 3-1. 불용어 및 전처리 설정 (이전 단계와 동일)
STOP_WORDS_LIST = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
    'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
    'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but',
    'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further',
    'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each',
    'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
    'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'covid', 'vaccine', 'get', 'would', 'could', 'one', 'take', 'need', 'people', 'us', 'say',
    'make', 'go', 'know', 'see', 'many', 'like', 'think', 'dont', 'im', 'ive', 'said', 'thats',
    'really', 'back', 'much', 'still', 'even', 'want', 'time', 'also', 'something', 'going',
    'look', 'lot', 'way', 'got', 'didnt', 'anyone', 'new', 'ever', 'may', 'tell', 'last',
    'week', 'every', 'things', 'using', 'way', 'since', 'first', 'getting', 'without'
]

def preprocess_text_topic(text):
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [word for word in words if word not in STOP_WORDS_LIST and len(word) > 2]
    return words

# 3-2. 감정별 텍스트 필터링
df_negative = df_final[df_final['Predicted_Sentiment'] == 'Negative'].copy()
df_positive = df_final[df_final['Predicted_Sentiment'] == 'Positive'].copy()

# 3-3. Top 10 키워드 추출 함수
def get_top_words(df, n=10):
    all_words = df['text'].astype(str).str.lower().apply(preprocess_text_topic).explode().dropna()
    word_counts = Counter(all_words)
    return pd.DataFrame(word_counts.most_common(n), columns=['word', 'count'])

top_neg = get_top_words(df_negative)
top_pos = get_top_words(df_positive)

# 3-4. 도표 생성 (병렬 막대 그래프)
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('3. 감정별 핵심 논란 토픽 비교 (Negative vs. Positive)', fontsize=16)

# Negative Plot
sns.barplot(x='count', y='word', data=top_neg.sort_values(by='count', ascending=False), ax=axes[0], color='#E34A33')
axes[0].set_title(f'부정 여론 TOP {len(top_neg)} 키워드 ({len(df_negative)}건)', fontsize=14)
axes[0].set_xlabel('언급 횟수')
axes[0].set_ylabel('키워드')

# Positive Plot
sns.barplot(x='count', y='word', data=top_pos.sort_values(by='count', ascending=False), ax=axes[1], color='#74C476')
axes[1].set_title(f'긍정 여론 TOP {len(top_pos)} 키워드 ({len(df_positive)}건)', fontsize=14)
axes[1].set_xlabel('언급 횟수')
axes[1].set_ylabel('') # Y축 라벨 제거 (겹침 방지)

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Suptitle 공간 확보
plt.savefig("sentiment_topic_comparison_final.png")
plt.close()

print("✅ 도표 3 저장 완료: 'sentiment_topic_comparison_final.png'")