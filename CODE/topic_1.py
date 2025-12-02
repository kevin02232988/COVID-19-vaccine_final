import pandas as pd
from collections import Counter
import re

# 파일 로드
OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"
try:
    df_final = pd.read_csv(OUTPUT_PREDICTED_FILE)
except FileNotFoundError:
    print(f"[ERROR] 최종 예측 파일 ('{OUTPUT_PREDICTED_FILE}')을 찾을 수 없습니다. 분석 파일이 존재하는지 확인해 주세요.")
    exit()

# 1. 부정적인 텍스트만 필터링 (74,954건)
df_negative = df_final[df_final['Predicted_Sentiment'] == 'Negative'].copy()
negative_texts = df_negative['text'].astype(str).str.lower()

# 2. 분석할 토픽 핵심 키워드 정의 (NMF 결과 기반)
# 토픽 1, 2, 4, 5의 핵심 키워드 및 논란 키워드
CORE_TOPIC_KEYWORDS = [
    'mask', 'masks', 'wear', 'wearing', 'right', 'work', 'virus', 'feel', 'long', 'side effect',
    'adverse', 'money', 'shit', 'youtube', 'com', 'reddit', 'message'
]

# 3. 불용어 목록 (이전 단계와 동일하게 설정)
STOP_WORDS_LIST = [
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself',
    'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
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


# 4. 언급 횟수 계산 (전체 텍스트에서 각 키워드의 등장 횟수)
keyword_counts = Counter()
total_texts = len(df_negative)

for keyword in CORE_TOPIC_KEYWORDS:
    # 텍스트 내에서 키워드의 등장 횟수를 직접 카운트
    count = df_negative['text'].str.lower().str.count(r'\b' + re.escape(keyword) + r'\b').sum()
    keyword_counts[keyword] = count

# 5. 결과 DataFrame 생성 및 출력 (논의 규모 시각화)
df_counts = pd.DataFrame(keyword_counts.items(), columns=['Keyword', 'Total Mentions'])
df_counts['Mentions per 1000 texts'] = (df_counts['Total Mentions'] / total_texts) * 1000

# 언급 횟수가 높은 순으로 정렬
df_counts = df_counts.sort_values(by='Total Mentions', ascending=False)

print("\n## 📊 핵심 논란 키워드 총 언급 빈도")
print("---")
print(f"**분석 대상 데이터:** {total_texts}건 (Negative Sentiment)")
print("\n--- 논란 키워드 총 언급 횟수 ---")
print(df_counts.to_markdown(index=False))

print("\n")

# 6. 보고서용 이미지 시각화 (막대 그래프)
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
# 횟수가 높은 순으로 정렬하여 플로팅
sns.barplot(x='Total Mentions', y='Keyword', data=df_counts, color='#E34A33')

plt.title('핵심 논란 키워드 총 언급 횟수 (Negative Sentiment)', fontsize=14)
plt.xlabel('총 언급 횟수', fontsize=12)
plt.ylabel('키워드', fontsize=12)
plt.tight_layout()

plt.savefig("controversy_keyword_mentions.png")
plt.close()