import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW

# ============================================================
# 1. CONFIG
# ============================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"  # 공개 모델 사용
LR = 5e-5
EPOCHS_NEUTRAL = 4
EPOCHS_BINARY = 6
MAX_LEN = 384
BATCH_SIZE = 16

print(f"Using device: {DEVICE}")

# ============================================================
# 2. Data Loading
# ============================================================
df = pd.read_csv("BERT_labeled_three.csv")  # CSV 파일 이름 확인
df = df.dropna(subset=["text", "sentiment_three"])
print(f"라벨링 데이터 로드 완료: {len(df)}개")

# ============================================================
# 2-1. 문자열 라벨을 숫자로 매핑
# ============================================================
# CSV가 '긍정', '부정', '중립'으로 되어 있는 경우
label_map = {"중립": 0, "긍정": 1, "부정": 2}
df["sentiment_three"] = df["sentiment_three"].map(label_map)

# 매핑 후 결측치 제거 (예: 다른 값 있으면 제거)
df = df.dropna(subset=["sentiment_three"])
df["sentiment_three"] = df["sentiment_three"].astype(int)

# ============================================================
# 3. Neutral/Binary 분리
# ============================================================
df_neutral = df[df["sentiment_three"].isin([0, 1])].reset_index(drop=True)
df_binary = df[df["sentiment_three"].isin([1, 2])].copy().reset_index(drop=True)
df_binary["sentiment_three"] = df_binary["sentiment_three"].replace({1: 1, 2: 0})

print(f"Neutral 후보 수: {len(df_neutral)}")
print(f"Binary 후보 수: {len(df_binary)}")

# ============================================================
# 4. Text Augmentation
# ============================================================
def eda_augment(text, prob_del=0.12, prob_swap=0.10):
    words = text.split()
    new_words = [w for w in words if random.random() > prob_del]
    if len(new_words) > 1 and random.random() < prob_swap:
        idx1, idx2 = np.random.choice(len(new_words), 2, replace=False)
        new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
    return " ".join(new_words)

# ============================================================
# 5. Dataset Class
# ============================================================
class ReviewDataset(Dataset):
    def __init__(self, df, tokenizer, augment=False):
        self.df = df
        self.tokenizer = tokenizer
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        if self.augment:
            text = eda_augment(text)
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "label": torch.tensor(row["sentiment_three"], dtype=torch.long)
        }

# ============================================================
# 6. Train & Eval Functions
# ============================================================
def train_epoch(model, loader, optimizer, scheduler):
    model.train()
    losses = []
    for batch in tqdm(loader):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        mask = batch["attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        output = model(input_ids, attention_mask=mask, labels=labels)
        loss = output.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return np.mean(losses)

def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            logits = model(input_ids, attention_mask=mask).logits
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / total

# ============================================================
# 7. Predict Function
# ============================================================
def predict(model, df_raw, tokenizer):
    model.eval()
    preds = []
    loader = DataLoader(
        ReviewDataset(df_raw, tokenizer, augment=False),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            input_ids = batch["input_ids"].to(DEVICE)
            mask = batch["attention_mask"].to(DEVICE)

            logits = model(input_ids, attention_mask=mask).logits
            pred = logits.argmax(dim=1).cpu().numpy()
            preds.extend(pred.tolist())
    return preds

# ============================================================
# 8. Train Neutral Filter Model
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"1단계 학습 라벨 분포: {Counter(df_neutral['sentiment_three'])}")

train_neutral = df_neutral.sample(frac=0.85, random_state=42)
val_neutral = df_neutral.drop(train_neutral.index)

train_loader = DataLoader(
    ReviewDataset(train_neutral, tokenizer, augment=True),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    ReviewDataset(val_neutral, tokenizer),
    batch_size=BATCH_SIZE
)

model_neutral = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(DEVICE)

optimizer = AdamW(model_neutral.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS_NEUTRAL * len(train_loader)
)

best_acc = 0
best_path = "./saved_models/best_model_neutral.bin"
os.makedirs("saved_models", exist_ok=True)

for epoch in range(EPOCHS_NEUTRAL):
    train_loss = train_epoch(model_neutral, train_loader, optimizer, scheduler)
    val_acc = evaluate(model_neutral, val_loader)
    print(f"[neutral] Epoch {epoch+1}/{EPOCHS_NEUTRAL} validation accuracy: {val_acc:.4f}")
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model_neutral.state_dict(), best_path)
        print(f"Saved best model -> {best_path}")

print(f"✅ 1단계 (중립 필터링) 최고 Validation Accuracy: {best_acc:.4f}")

# ============================================================
# 9. Apply Neutral Filter
# ============================================================
df_raw = pd.read_csv("FINAL_cleaned_reviews_22939.csv")
df_raw["sentiment_three"] = df_raw["sentiment_three"].map(label_map).astype(int)
df_raw = df_raw.dropna(subset=["text", "sentiment_three"]).reset_index(drop=True)

model_neutral.load_state_dict(torch.load(best_path))
df_raw["neutral_pred"] = predict(model_neutral, df_raw, tokenizer)
df_no_neutral = df_raw[df_raw["neutral_pred"] == 1].reset_index(drop=True)
print(f"총 {len(df_raw)} → 중립 제거 후 {len(df_no_neutral)}개 남음")

# ============================================================
# 10. Train Binary Sentiment Model
# ============================================================
print(f"2단계 학습 라벨 분포: {Counter(df_binary['sentiment_three'])}")

train_bin = df_binary.sample(frac=0.85, random_state=42)
val_bin = df_binary.drop(train_bin.index)

train_loader = DataLoader(
    ReviewDataset(train_bin, tokenizer, augment=True),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    ReviewDataset(val_bin, tokenizer),
    batch_size=BATCH_SIZE
)

model_binary = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
).to(DEVICE)

optimizer = AdamW(model_binary.parameters(), lr=LR)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=EPOCHS_BINARY * len(train_loader)
)

best_acc_bin = 0
best_path_bin = "./saved_models/best_model_binary.bin"

for epoch in range(EPOCHS_BINARY):
    train_loss = train_epoch(model_binary, train_loader, optimizer, scheduler)
    val_acc = evaluate(model_binary, val_loader)
    print(f"[binary] Epoch {epoch+1}/{EPOCHS_BINARY} validation accuracy: {val_acc:.4f}")
    if val_acc > best_acc_bin:
        best_acc_bin = val_acc
        torch.save(model_binary.state_dict(), best_path_bin)
        print(f"Saved best model -> {best_path_bin}")

print(f"✅ 2단계 (긍/부정) 최고 Validation Accuracy: {best_acc_bin:.4f}")

# ============================================================
# 11. Predict final sentiment
# ============================================================
model_binary.load_state_dict(torch.load(best_path_bin))
df_no_neutral["sentiment"] = predict(model_binary, df_no_neutral, tokenizer)

# save to CSV
OUTPUT = "FINAL_no_neutral_binary_prediction_v3_large_recommended.csv"
df_no_neutral.to_csv(OUTPUT, index=False)
print(f"💾 결과 저장 완료: {OUTPUT}")

# ============================================================
# 12. Print final ratio
# ============================================================
pos = (df_no_neutral["sentiment"] == 1).sum()
neg = (df_no_neutral["sentiment"] == 0).sum()
total = len(df_no_neutral)

print("\n===== 최종 예측 결과 비율 =====")
print(f"전체 리뷰 수: {total}")
print(f"긍정(1): {pos}개 ({pos/total*100:.2f}%)")
print(f"부정(0): {neg}개 ({neg/total*100:.2f}%)")
print("================================")
