import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
from collections import Counter
import os

# =========================================
# 1️⃣ GPU 설정
# =========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# =========================================
# 2️⃣ Dataset 클래스 정의
# =========================================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # texts를 리스트로 확정하여 Pandas 인덱스 문제를 방지 (Safe conversion to list)
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # texts와 labels가 모두 순수 리스트임을 가정
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text, return_tensors="pt", max_length=self.max_len, padding="max_length", truncation=True
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# =========================================
# 3️⃣ 학습 + 평가 함수
# =========================================
def train_and_evaluate(train_texts, train_labels, val_texts, val_labels, num_labels, lr, epochs, weights_tensor):
    print(f"| LR: {lr}, Epochs: {epochs}")
    MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)

    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model = ElectraForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=num_labels, use_safetensors=True
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = AdamW(model.parameters(), lr=lr)

    # === 학습 ===
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            loss = loss_fn(outputs.logits, inputs["labels"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    # === 검증 ===
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds.cpu() == inputs["labels"].cpu()).sum().item()
            total += len(preds)
    acc = correct / total

    return model, acc, tokenizer


# =========================================
# 4️⃣ 예측 함수
# =========================================
def predict_data(model, tokenizer, df_target, output_path=None):
    # 예측 시에는 인덱스 문제가 없도록 리스트로 변환
    texts = df_target["text"].astype(str).tolist()
    dataset = TextDataset(texts, None, tokenizer)
    loader = DataLoader(dataset, batch_size=32)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

    df_target["predicted_label"] = preds

    if output_path:
        df_target.to_csv(output_path, index=False, encoding="utf-8")
        print(f"💾 결과 저장 완료: {output_path}")

    return df_target


# =========================================
# 5️⃣ Main Pipeline
# =========================================

THREE_FILE = "BERT_labeled_three.csv"
FINAL_FILE = "FINAL_DATA_FILTERED_#TRUE.csv"

# === Step 1. 데이터 로드 ===
df_three = pd.read_csv(THREE_FILE)
print(f"라벨링 데이터 로드 완료: {len(df_three)}개")

# === Step 2. 중립 vs 비중립 (1단계) ===
df_three["is_neutral"] = df_three["sentiment_three"].apply(lambda x: 1 if x == "중립" else 0)

# 인덱스 초기화 및 리스트 변환 (KeyError 방지)
df_temp = df_three[['text', 'is_neutral']].reset_index(drop=True)
texts = df_temp["text"].tolist()
labels = df_temp["is_neutral"].tolist()

train_t, val_t, train_l, val_l = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)

counts = Counter(train_l)
weights = {i: len(train_l) / (2 * c) for i, c in counts.items()}
weights_tensor = torch.tensor([weights[i] for i in sorted(weights.keys())], dtype=torch.float).to(device)

print(f"1단계 학습 라벨 분포: {counts}")

# 1단계 튜닝: LR을 안정화하고 Epoch를 10회로 증가
neutral_model, neutral_acc, neutral_tokenizer = train_and_evaluate(
    train_t, train_l, val_t, val_l, num_labels=2, lr=1e-5, epochs=10, weights_tensor=weights_tensor
)
print(f"✅ 1단계 (중립 필터링) Validation Accuracy: {neutral_acc:.4f}")

# === Step 3. 원본 데이터 로드 & 중립 제거 (2단계) ===
df_final = pd.read_csv(FINAL_FILE)
df_pred_neutral = predict_data(neutral_model, neutral_tokenizer, df_final, None)

# Soft filtering: 중립 확률이 높은 항목만 제거 (0 = 비중립만 남김)
df_filtered = df_pred_neutral[df_pred_neutral["predicted_label"] == 0].copy()
print(f"총 {len(df_final)} → 중립 제거 후 {len(df_filtered)}개 남음")

# === Step 4. 긍/부정 학습 데이터 준비 ===
df_pure = df_three[df_three["sentiment_three"] != "중립"].copy().reset_index(drop=True)
df_pure["label"] = df_pure["sentiment_three"].map({"부정": 0, "긍정": 1})

# 인덱스 리셋 후 리스트 변환
texts_p = df_pure["text"].tolist()
labels_p = df_pure["label"].tolist()

tp_t, vp_t, tp_l, vp_l = train_test_split(texts_p, labels_p, test_size=0.2, random_state=42, stratify=labels_p)

counts2 = Counter(tp_l)
weights2 = {i: len(tp_l) / (2 * c) for i, c in counts2.items()}
weights_tensor2 = torch.tensor([weights2[i] for i in sorted(weights2.keys())], dtype=torch.float).to(device)

print(f"2단계 학습 라벨 분포: {counts2}")

# 2단계 튜닝: LR=1e-5, Epochs=10 (학습 강화)
# 1단계의 성공적인 설정을 2단계에 그대로 적용하여 안정화와 성능 극대화를 목표로 합니다.
binary_model, binary_acc, binary_tokenizer = train_and_evaluate(
    tp_t, tp_l, vp_t, vp_l, num_labels=2, lr=1e-5, epochs=10, weights_tensor=weights_tensor2
)
print(f"✅ 2단계 (긍/부정) Validation Accuracy: {binary_acc:.4f}")

# === Step 5. 최종 예측 & 저장 ===
OUTPUT_FILE = "FINAL_no_neutral_binary_prediction_v3_fixed.csv"

# 최종 예측은 정제된 데이터프레임(df_filtered)에 대해 수행
df_final_pred = predict_data(binary_model, binary_tokenizer, df_filtered, OUTPUT_FILE)

# 🎯 보고 요약
print("\n" + "=" * 60)
print("🎯 최종 보고 요약")
print(f"1단계 (중립 필터링) Validation Accuracy: {neutral_acc:.4f}")
print(f"2단계 (긍/부정) Validation Accuracy: {binary_acc:.4f}")
print(f"최종 분석 데이터 수 (정제 후): {len(df_final_pred)}")
print("=" * 60)