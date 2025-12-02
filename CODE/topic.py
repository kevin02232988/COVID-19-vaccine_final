import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import NMF
import numpy as np

# 파일 로드 (BERT 예측 결과 파일)
OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"
try:
    # 파일을 다시 로드하여 부정적 데이터만 필터링합니다.
    df_final = pd.read_csv(OUTPUT_PREDICTED_FILE)
except FileNotFoundError:
    print(f"[ERROR] 최종 예측 파일 ('{OUTPUT_PREDICTED_FILE}')을 찾을 수 없습니다. 분석 파일이 존재하는지 확인해 주세요.")
    exit()

# 1. 부정적인 텍스트만 필터링 (논란의 핵심)
df_negative = df_final[df_final['Predicted_Sentiment'] == 'Negative'].copy()
docs = df_negative['text'].astype(str).tolist()

# 2. 불용어 및 키워드 설정 (이전 단계에서 사용된 목록 재사용)
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

    # 프로젝트 관련 불필요한 공통 키워드 및 잡음 추가
    'covid', 'vaccine', 'get', 'would', 'could', 'one', 'take', 'need', 'people', 'us', 'say',
    'make', 'go', 'know', 'see', 'many', 'like', 'think', 'dont', 'im', 'ive', 'said', 'thats',
    'really', 'back', 'much', 'still', 'even', 'want', 'time', 'also', 'something', 'going',
    'look', 'lot', 'way', 'got', 'didnt', 'anyone', 'new', 'ever', 'may', 'tell', 'last',
    'week', 'every', 'things', 'using', 'way', 'since', 'first', 'getting', 'without'
]

# 3. CountVectorizer 설정 및 적용 (단어 빈도 행렬 생성)
# min_df를 50으로 설정하여 노이즈 및 희귀 단어 제거
vectorizer = CountVectorizer(
    stop_words=STOP_WORDS_LIST,
    min_df=50,
    ngram_range=(1, 2)  # 단어 1개 또는 2개 조합(빅그램) 사용
)
dtm = vectorizer.fit_transform(docs)

# 4. NMF 모델 학습 (5개 토픽 추출)
num_topics = 5
# NMF는 토픽 모델링에 효과적이며, max_iter를 300으로 설정하여 안정적인 결과를 유도
nmf = NMF(n_components=num_topics, random_state=1, max_iter=300)
nmf.fit(dtm)

feature_names = vectorizer.get_feature_names_out()
topic_results = []
top_words_count = 10

# 5. 토픽별 상위 단어 추출 및 저장
for topic_idx, topic in enumerate(nmf.components_):
    top_features_ind = topic.argsort()[:-top_words_count - 1:-1]
    top_features = [feature_names[i] for i in top_features_ind]
    topic_results.append({
        'Topic': f'토픽 {topic_idx + 1}',
        'Keywords': ', '.join(top_features)
    })

# 6. 최종 출력
df_topics = pd.DataFrame(topic_results)

print("\n## 🗺️ NMF 기반 부정 여론 핵심 토픽 추출")
print("---")
print(f"분석 대상 데이터: {len(df_negative)}건 (Negative Sentiment)")
print(f"추출된 토픽 수: {num_topics}개")
print("\n--- 토픽별 상위 키워드 ---")
print(df_topics.to_markdown(index=False))