# build_topics.py
import pandas as pd
from pathlib import Path

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

# =========================
# 0. 경로 설정
# =========================
BASE = Path(__file__).resolve().parent   # 이 파일 위치 기준
CSV_PATH = BASE / "DDDD.csv"             # DDDD.csv 위치 (루트에 있다고 가정)

print(f"[INFO] Load CSV: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# =========================
# 1. 기본 정리
# =========================
# 날짜 컬럼 변환 (필요에 따라 컬럼명 수정)
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.dropna(subset=["created_at", "text"]).copy()

# (선택) 부정 댓글만 사용 – 0=부정, 1=긍정이라고 가정
if "sentiment_label" in df.columns:
    df = df[df["sentiment_label"] == 0].copy()
    print(f"[INFO] Negative rows only: {len(df)}")
else:
    print("[WARN] 'sentiment_label' 컬럼이 없어 전체 데이터로 토픽을 추출합니다.")

# 텍스트 최소 길이 필터 (너무 짧은 걸 제거)
df["text_len"] = df["text"].astype(str).str.len()
df = df[df["text_len"] >= 30].copy()
print(f"[INFO] After length filter: {len(df)} rows")

docs = df["text"].astype(str).tolist()

# =========================
# 2. 임베딩 모델 + BERTopic 설정
# =========================
print("[INFO] Load sentence-transformer model...")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

topic_model = BERTopic(
    embedding_model=embedding_model,
    language="english",        # 영어 위주 데이터니까 english
    min_topic_size=50,         # 토픽 최소 크기 (필요하면 조정)
    nr_topics=None,            # 자동 결정
    low_memory=True,
    verbose=True,
)

# =========================
# 3. 토픽 학습
# =========================
print("[INFO] Fit BERTopic...")
topics, probs = topic_model.fit_transform(docs)

df["topic_id"] = topics

print("[INFO] Topic assignment example:")
print(df[["text", "topic_id"]].head(3))

# =========================
# 4. 결과 저장
# =========================
OUT_TOPIC_DF = BASE / "bertopic_doc_topics.csv"
OUT_TOPIC_INFO = BASE / "bertopic_topic_info.csv"
OUT_MODEL = BASE / "bertopic_model.pkl"

# 문서별 토픽 정보 저장 (id, created_at 같이 보관)
cols_to_keep = [c for c in ["id", "created_at", "text", "sentiment_label"] if c in df.columns]
cols_to_keep += ["topic_id"]
df[cols_to_keep].to_csv(OUT_TOPIC_DF, index=False)
print(f"[INFO] Saved doc-topic mapping to: {OUT_TOPIC_DF}")

# 토픽별 대표 단어/빈도 정보
topic_info = topic_model.get_topic_info()
topic_info.to_csv(OUT_TOPIC_INFO, index=False)
print(f"[INFO] Saved topic info to: {OUT_TOPIC_INFO}")

# 모델 자체도 나중에 다시 쓰고 싶으면
import pickle
with open(OUT_MODEL, "wb") as f:
    pickle.dump(topic_model, f)
print(f"[INFO] Saved BERTopic model to: {OUT_MODEL}")
