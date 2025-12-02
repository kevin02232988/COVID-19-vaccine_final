import pandas as pd
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk
import os

# -------------------------------------------------------------------------------------
# 0. 환경 설정 및 정의
# -------------------------------------------------------------------------------------
# NLTK 및 불용어 설정
try:
    # stopwords가 다운로드되어 있는지 확인
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
stop_words = stopwords.words('english')
vectorizer_model = CountVectorizer(stop_words=stop_words)

# -------------------------------------------------------------------------------------
# 1. 모델 로드 및 데이터 준비
# -------------------------------------------------------------------------------------

topic_model_neg = None
topic_model_pos = None

# 1-1. 모델 로드 시도
try:
    # 모델 파일이 존재한다고 가정하고 로드
    topic_model_neg = BERTopic.load("bertopic_negative")
    topic_model_pos = BERTopic.load("bertopic_positive")
    print("✅ BERTopic 모델 로드 완료 (bertopic_negative, bertopic_positive).")

except Exception as e:
    print(f"⚠️ 경고: BERTopic 모델 로드 실패 ({e}). 토픽 결과를 보기 위해 모델을 다시 학습합니다.")

    # 1-2. 로드 실패 시, 데이터 로드 및 재학습을 통해 결과 생성
    try:
        # 사용자가 업로드한 CSV 파일 로드 및 병합 (재학습을 위한 텍스트 데이터 준비)
        df_neg_loaded = pd.read_csv("negative_topics.csv")
        df_pos_loaded = pd.read_csv("positive_topics.csv")
        df = pd.concat([df_neg_loaded, df_pos_loaded], ignore_index=True)

        df = df.dropna(subset=['text', 'sentiment_label'])
        df['sentiment_label'] = df['sentiment_label'].astype(int)

        df_neg = df[df['sentiment_label'] == 0].copy()
        df_pos = df[df['sentiment_label'] == 1].copy()

        # 부정 리뷰 재학습 (결과 출력이 목적이므로 확률 계산은 생략)
        print("\n🔵 부정 리뷰 토픽 모델 재학습 시작...")
        topic_model_neg = BERTopic(
            vectorizer_model=vectorizer_model,
            language="multilingual",
            calculate_probabilities=False
        ).fit(df_neg["text"])
        topic_model_neg.save("bertopic_negative")  # 다음 실행을 위해 저장

        # 긍정 리뷰 재학습
        print("\n🟢 긍정 리뷰 토픽 모델 재학습 시작...")
        topic_model_pos = BERTopic(
            vectorizer_model=vectorizer_model,
            language="multilingual",
            calculate_probabilities=False
        ).fit(df_pos["text"])
        topic_model_pos.save("bertopic_positive")  # 다음 실행을 위해 저장

        print("\n✅ BERTopic 모델 재학습 및 저장 완료.")

    except FileNotFoundError:
        print("\n❌ 오류: 모델 로드 및 재학습 모두 실패. 원본 CSV 파일(negative_topics.csv, positive_topics.csv)을 찾을 수 없습니다.")
        exit()
    except Exception as ee:
        print(f"\n❌ 심각한 오류: 재학습 중 오류 발생 ({ee})")
        exit()


# -------------------------------------------------------------------------------------
# 2. 토픽 정보 추출 및 출력 (핵심 결과)
# -------------------------------------------------------------------------------------

def print_topic_summary(model, title):
    """모델의 토픽 정보를 추출하고 포맷하여 출력합니다."""
    topic_info = model.get_topic_info()

    # 노이즈 토픽(-1) 제외하고 의미 있는 토픽만 선택
    meaningful_topics = topic_info[topic_info['Topic'] != -1].copy()

    print("\n" + "=" * 80)
    print(f"## {title} - 의미 있는 주요 토픽 (상위 10개)")
    print("=" * 80)

    # 필요한 컬럼만 선택
    display_cols = ['Topic', 'Count', 'Name', 'Representation']
    display_df = meaningful_topics[display_cols]

    # 리스트 형태의 키워드를 쉼표로 구분된 문자열로 변환하여 출력 가독성 높이기
    display_df['Representation'] = display_df['Representation'].apply(lambda x: ', '.join(x))

    # Markdown 형식으로 출력
    print(display_df.head(10).to_markdown(index=False))

    # 전체 문서 수 요약
    total_docs = topic_info['Count'].sum()
    noise_docs = topic_info[topic_info['Topic'] == -1]['Count'].iloc[0] if -1 in topic_info['Topic'].values else 0
    print("-" * 80)
    print(f"총 분석 문서 수: {total_docs} | 노이즈(-1) 토픽 문서 수: {noise_docs} | 의미 있는 토픽 문서 수: {total_docs - noise_docs}")
    print("=" * 80)


if topic_model_neg:
    print_topic_summary(topic_model_neg, "📉 부정 리뷰 토픽 모델 결과")

if topic_model_pos:
    print_topic_summary(topic_model_pos, "📈 긍정 리뷰 토픽 모델 결과")

print("\n\n🎉 코드를 돌린 결과(의미있는 토픽 키워드) 출력이 완료되었습니다. (추가 시각화 기능은 모두 제거되었습니다.)")