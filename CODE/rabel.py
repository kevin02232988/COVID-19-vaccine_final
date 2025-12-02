import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

# 1️⃣ 데이터 불러오기
df = pd.read_csv("sampled_for_#labeling.csv")

# 2️⃣ 모델 및 토크나이저 로드
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 3️⃣ 파이프라인 설정 (⭐ 긴 문장은 자르기)
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    truncation=True,        # <= 핵심 수정!
    max_length=512,         # 최대 길이 제한
)

# 4️⃣ 감정 예측
tqdm.pandas(desc="감정 예측 중...")
df["raw_label"] = df["text"].progress_apply(lambda x: sentiment_pipeline(str(x))[0]["label"])

# 5️⃣ 숫자형 별점 변환
df["stars"] = df["raw_label"].str.extract(r"(\d)").astype(int)

# 6️⃣ 이진 분류 (긍/부정)
df["sentiment_binary"] = df["stars"].apply(lambda x: "긍정" if x >= 4 else "부정")

# 7️⃣ 삼분류 (긍/부/중립)
def map_three_class(x):
    if x <= 2:
        return "부정"
    elif x == 3:
        return "중립"
    else:
        return "긍정"

df["sentiment_three"] = df["stars"].apply(map_three_class)

# 8️⃣ 결과 저장
df[["text", "created_at", "sentiment_binary"]].to_csv("BERT_labeled_binary.csv", index=False, encoding="utf-8-sig")
df[["text", "created_at", "sentiment_three"]].to_csv("BERT_labeled_three.csv", index=False, encoding="utf-8-sig")

print("✅ 라벨링 완료!")
print(" - BERT_labeled_binary.csv (긍/부정)")
print(" - BERT_labeled_three.csv (긍/부/중립)")

