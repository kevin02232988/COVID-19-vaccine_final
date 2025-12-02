import pandas as pd
import numpy as np
# 클래스 불균형 문제를 해결하기 위한 Balanced ML 모델 사용
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# --- 파일 경로 설정 ---
TRAIN_FILENAME = "labeled_output#.csv"
NEW_DATA_FILENAME = "20_per#_final.csv"
OUTPUT_FILENAME = "labeled_output#_4.csv"

# 목표 비율 설정
TARGET_POSITIVE_RATIO = 0.30  # 30% 긍정

# 1. 훈련 데이터 로드 및 Balanced ML 모델 학습
try:
    df_train = pd.read_csv(TRAIN_FILENAME)
    X_train = df_train['text'].fillna('')
    y_train = df_train['sentiment']
except FileNotFoundError:
    print(f"❌ 오류: 훈련 파일 '{TRAIN_FILENAME}'을 찾을 수 없습니다.")
    exit()

# Class-Weighted Logistic Regression 파이프라인 구성
# class_weight='balanced' 옵션으로 긍정 클래스에 더 큰 가중치를 부여합니다.
pipeline_balanced = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words=None)),
    ('clf', LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42))
])
pipeline_balanced.fit(X_train, y_train)

# 2. 새로운 데이터에 예측 적용
try:
    df_new = pd.read_csv(NEW_DATA_FILENAME)
    df_new['text'] = df_new['text'].fillna('')
    X_new = df_new['text']
except FileNotFoundError:
    print(f"❌ 오류: 새 데이터 파일 '{NEW_DATA_FILENAME}'을 찾을 수 없습니다.")
    exit()

# 긍정 확률 예측 (prob_positive)
df_new['prob_positive'] = pipeline_balanced.predict_proba(X_new)[:, 1]

total_samples = len(df_new)
target_positive_count = int(total_samples * TARGET_POSITIVE_RATIO)  # 4186 * 0.30 = 1255

# 3. 목표 긍정 개수를 달성하는 확률 임계값 찾기
df_sorted = df_new.sort_values(by='prob_positive', ascending=False)
if target_positive_count > 0:
    # 긍정 확률 상위 1256번째 값(임계값)을 찾습니다.
    new_threshold = df_sorted['prob_positive'].iloc[target_positive_count - 1]
else:
    new_threshold = 1.0

# 4. 새로운 임계값을 적용하여 'sentiment' 레이블 재설정
df_new['sentiment'] = np.where(df_new['prob_positive'] >= new_threshold, 1, 0)

# 5. 결과 분석 및 출력
sentiment_counts = df_new['sentiment'].value_counts().sort_index()
sentiment_ratio = (sentiment_counts / total_samples) * 100
# ... (출력 생략)

# 6. 결과 저장
df_new.to_csv(OUTPUT_FILENAME, index=False)