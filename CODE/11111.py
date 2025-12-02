import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

# -------------------------------------------------------------------------------------
# 0. 환경 설정
# -------------------------------------------------------------------------------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

ABSOLUTE_PATH = r"C:\Users\user\PycharmProjects\PythonProject6\\"
FILE_SENTIMENT = ABSOLUTE_PATH + "DDDD.csv"

nltk.download('stopwords')
stop_words = stopwords.words('english')

# -------------------------------------------------------------------------------------
# 1. 데이터 로드
# -------------------------------------------------------------------------------------
df = pd.read_csv(FILE_SENTIMENT)
df = df.dropna(subset=['text'])

print("데이터 로드 완료:", df.shape)

# -------------------------------------------------------------------------------------
# 2. 긍정 / 부정 분리
# -------------------------------------------------------------------------------------
df_pos = df[df['sentiment_label'] == 1].copy()
df_neg = df[df['sentiment_label'] == 0].copy()

print(f"긍정 리뷰 수: {len(df_pos)}, 부정 리뷰 수: {len(df_neg)}")

# -------------------------------------------------------------------------------------
# 3. 공통 Vectorizer
# -------------------------------------------------------------------------------------
vectorizer_model = CountVectorizer(stop_words=stop_words)

# -------------------------------------------------------------------------------------
# 4. 부정 리뷰 토픽 모델링
# -------------------------------------------------------------------------------------
print("\n🔵 부정 리뷰 토픽 모델링 시작...")
topic_model_neg = BERTopic(
    vectorizer_model=vectorizer_model,
    language="multilingual",
    calculate_probabilities=True
)

topics_neg, probs_neg = topic_model_neg.fit_transform(df_neg["text"])
df_neg["topic"] = topics_neg
topic_model_neg.save("bertopic_negative")

print("부정 주요 토픽 예시:")
print(topic_model_neg.get_topic(0))

# -------------------------------------------------------------------------------------
# 5. 긍정 리뷰 토픽 모델링
# -------------------------------------------------------------------------------------
print("\n🟢 긍정 리뷰 토픽 모델링 시작...")
topic_model_pos = BERTopic(
    vectorizer_model=vectorizer_model,
    language="multilingual",
    calculate_probabilities=True
)

topics_pos, probs_pos = topic_model_pos.fit_transform(df_pos["text"])
df_pos["topic"] = topics_pos
topic_model_pos.save("bertopic_positive")

print("긍정 주요 토픽 예시:")
print(topic_model_pos.get_topic(0))

# -------------------------------------------------------------------------------------
# 6. 긍/부정 비율 그래프 (정적 막대 차트)
# -------------------------------------------------------------------------------------
pos_count = len(df_pos)
neg_count = len(df_neg)
total = pos_count + neg_count

plt.figure(figsize=(6, 5))
plt.bar(['Positive', 'Negative'], [pos_count / total, neg_count / total])
plt.title("긍정 / 부정 비율")
plt.ylabel("비율")
plt.tight_layout()
plt.savefig("sentiment_ratio.png")
plt.show()

print("\n📊 긍부정 비율 그래프 저장 완료 → sentiment_ratio.png")

# -------------------------------------------------------------------------------------
# 7. 토픽 결과 저장
# -------------------------------------------------------------------------------------
df_neg.to_csv("negative_topics_2.csv", index=False, encoding='utf-8-sig')
df_pos.to_csv("positive_topics_2.csv", index=False, encoding='utf-8-sig')

print("📁 토픽 결과 저장 완료: negative_topics.csv / positive_topics.csv")
