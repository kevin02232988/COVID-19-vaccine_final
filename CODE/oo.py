import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter # 빈도 계산을 위해 Counter 라이브러리 추가
import sys

# ----------------------------------------------------------------------
# 1. 데이터 전처리 함수 (이전에 사용한 것과 동일)
# ----------------------------------------------------------------------
def preprocess_text(text):
    """텍스트를 전처리하는 함수: 소문자화, 불필요한 문자 제거, 토큰화, 불용어 제거, 표제어 추출."""
    if not isinstance(text, str):
        return []

    # 1-1. 소문자화 및 URL/숫자/특수문자 제거
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)

    # 1-2. 토큰화
    tokens = nltk.word_tokenize(text)

    # 1-3. 불용어 제거 및 표제어 추출
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # 1-4. 길이가 3자 이하인 단어와 불용어 제거, 그리고 표제어 추출 적용
    processed_tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in stop_words and len(token) > 3
    ]

    return processed_tokens

# ----------------------------------------------------------------------
# 메인 실행 블록 (단어 빈도 분석)
# ----------------------------------------------------------------------
if __name__ == '__main__':

    # NLTK 데이터는 이전 단계에서 다운로드되었으므로 주석 처리합니다.
    # check_and_download_nltk_data()

    file_path = "Real_Final.csv"
    try:
        df = pd.read_csv(file_path)
        print(f"'{file_path}' 파일 로드 성공. 총 {len(df)}개의 데이터.")
    except FileNotFoundError:
        print(f"오류: 파일 경로를 확인해주세요. '{file_path}'를 찾을 수 없습니다.")
        sys.exit(1)

    # 'text' 컬럼을 전처리
    print("텍스트 데이터 전처리 중...")
    df['processed_text'] = df['text'].apply(preprocess_text)

    # 1. 모든 전처리된 단어를 하나의 리스트로 통합
    all_words = []
    for doc in df['processed_text']:
        if isinstance(doc, list):
            all_words.extend(doc)

    # 2. 단어 빈도 계산
    print("단어 빈도 계산 중...")
    word_counts = Counter(all_words)

    # 3. 상위 N개 키워드 추출
    TOP_N = 50
    top_keywords = word_counts.most_common(TOP_N)

    print("-" * 50)
    print(f"🚨🚨🚨 총 데이터에서 빈도가 가장 높은 상위 {TOP_N}개 키워드 🚨🚨🚨")

    # 결과를 깔끔하게 출력
    for i, (word, count) in enumerate(top_keywords):
        # f-string 포맷팅을 사용하여 번호, 단어, 횟수를 정렬하여 출력
        print(f"{i+1:2d}. {word:15s} : {count:,d}회")

    print("-" * 50)
    print(f"전체 문서 수: {len(df):,d}개")
    print(f"전체 고유 단어 수: {len(word_counts):,d}개")
    print("✨ 참고: 이 목록은 단순히 빈도만을 기반으로 합니다. '의미 있는' 키워드를 찾으려면 TF-IDF와 같은 가중치 기법을 추가로 사용할 수 있습니다.")