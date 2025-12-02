import pandas as pd
try:
    from textblob import TextBlob
except ImportError:
    print("TextBlob 라이브러리가 필요합니다. 설치: pip install textblob")
    raise

# -------------------------------
# 1️⃣ 데이터 로드
# -------------------------------
file_name = "20_per#_final.csv"
df = pd.read_csv(file_name)

# 'text' 결측값 처리
df['text'] = df['text'].fillna('').astype(str)

# -------------------------------
# 2️⃣ TextBlob 감성 점수 계산
# -------------------------------
def get_sentiment_polarity(text):
    return TextBlob(text).sentiment.polarity

df['polarity_score'] = df['text'].apply(get_sentiment_polarity)

# -------------------------------
# 3️⃣ 라벨링: 부정(0), 중립(1), 긍정(2)
# -------------------------------
# Threshold를 적용하여 중립 문장을 따로 라벨링
def label_sentiment(score, neg_th=-0.05, pos_th=0.05):
    if score < neg_th:
        return 0  # 부정
    elif score > pos_th:
        return 2  # 긍정
    else:
        return 1  # 중립

df['sentiment_label'] = df['polarity_score'].apply(label_sentiment)

# -------------------------------
# 4️⃣ 결과 확인
# -------------------------------
print("\n라벨 분포:")
print(df['sentiment_label'].value_counts())

print("\n상위 5개 데이터 확인:")
print(df[['text', 'polarity_score', 'sentiment_label']].head())

# -------------------------------
# 5️⃣ CSV 저장
# -------------------------------
output_file_name = "labeled_output#_balanced.csv"
df[['text', 'created_at', 'polarity_score', 'sentiment_label']].to_csv(
    output_file_name, index=False, encoding='utf-8-sig'
)

print(f"\n데이터가 성공적으로 라벨링되어 {output_file_name} 파일로 저장되었습니다.")
