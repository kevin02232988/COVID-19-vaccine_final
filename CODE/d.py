import pandas as pd
from transformers import pipeline
from tqdm import tqdm

# -------------------- 1. 파일 불러오기 --------------------
input_file = "Real_rabel.csv"  # 입력 파일 이름
output_file = "Real_rabel_labeled.csv"  # 출력 파일 이름

# CSV 읽기
df = pd.read_csv(input_file)
print(f"[INFO] 원본 데이터 로드 완료: {df.shape[0]}개 행")

# -------------------- 2. 감정 분석 모델 불러오기 --------------------
print("[INFO] 감정 분석 모델 로드 중...")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

# -------------------- 3. 감정 라벨링 수행 --------------------
results = []
for text in tqdm(df["text"].fillna(""), desc="감정 라벨링 중"):
    if len(text.strip()) == 0:
        results.append("neutral")  # 내용이 없을 때 중립 처리
        continue

    # 모델 예측
    result = sentiment_analyzer(text[:512])[0]  # 너무 긴 텍스트는 512토큰까지만
    label = result["label"].lower()  # positive / negative
    results.append(label)

df["sentiment"] = results

# -------------------- 4. 결과 저장 --------------------
df.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"[완료] 감정 라벨링 완료 → {output_file}")
print(df["sentiment"].value_counts())
