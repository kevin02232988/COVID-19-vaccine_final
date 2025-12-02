import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.optim import AdamW
from torch.nn.functional import softmax
from tqdm import tqdm
from collections import Counter
import os

# ======================================================
# 1️⃣ GPU 설정
# ======================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ======================================================
# 2️⃣ 데이터셋 클래스 정의
# ======================================================
class TextDataset(Dataset):
    """PyTorch 학습/검증용 데이터셋"""

    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer(
            text, return_tensors='pt', max_length=self.max_len, padding='max_length', truncation=True
        )
        item = {key: val.squeeze(0) for key, val in inputs.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ======================================================
# 3️⃣ 학습 및 평가 함수
# ======================================================
def train_and_evaluate(train_texts, train_labels, val_texts, val_labels, num_labels, custom_lr, epochs, weights_tensor):
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
    optimizer = AdamW(model.parameters(), lr=custom_lr)

    # === 학습 루프 ===
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = loss_fn(outputs.logits, inputs['labels'])
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
            correct += (preds.cpu() == inputs['labels'].cpu()).sum().item()
            total += len(preds)
    accuracy = correct / total

    return model, accuracy, tokenizer


# ======================================================
# 4️⃣ 예측 함수 (확신도 기반 필터링 추가)
# ======================================================
def predict_data(model, tokenizer, df_target, output_pred_path=None):
    final_texts = df_target['text'].astype(str).tolist()
    pred_dataset = TextDataset(final_texts, None, tokenizer)
    pred_loader = DataLoader(pred_dataset, batch_size=32)

    model.eval()
    preds_list, probs_list = [], []
    with torch.no_grad():
        for batch in tqdm(pred_loader, desc="Predicting dataset"):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            probs = softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            preds_list.extend(preds.cpu().numpy())
            probs_list.extend(probs[:, 1].cpu().numpy())  # 중립 확률

    df_target['predicted_label'] = preds_list
    df_target['neutral_prob'] = probs_list

    if output_pred_path:
        df_target.to_csv(output_pred_path, index=False, encoding='utf-8')
        print(f"💾 예측 결과 저장 완료: {output_pred_path}")

    return df_target


# ======================================================
# 5️⃣ 메인 파이프라인 실행
# ======================================================
THREE_CLASS_FILE = "BERT_labeled_three.csv"
FINAL_DATA_FILE = "FINAL_DATA_FILTERED_#TRUE.csv"

# === 1단계: 중립 vs 비중립 학습 ===
try:
    df_three = pd.read_csv(THREE_CLASS_FILE, encoding='utf-8')
except Exception:
    df_three = pd.read_csv(THREE_CLASS_FILE, encoding='cp949')

df_three['is_neutral'] = df_three['sentiment_three'].apply(lambda x: 1 if x == '중립' else 0)

neutral_texts = df_three['text'].tolist()
neutral_labels = df_three['is_neutral'].tolist()

tn_texts, vn_texts, tn_labels, vn_labels = train_test_split(
    neutral_texts, neutral_labels, test_size=0.2, random_state=42, stratify=neutral_labels
)

neutral_counts = Counter(tn_labels)
tn_total = len(tn_labels)
neutral_weights = {i: tn_total / (len(neutral_counts) * count) for i, count in neutral_counts.items()}
neutral_weights_tensor = torch.tensor([neutral_weights[i] for i in sorted(neutral_weights.keys())],
                                      dtype=torch.float).to(device)

print("\n" + "=" * 50)
print("1단계: 중립 분류기 학습 시작 (비중립=0, 중립=1)")
print(f"Original Training Distribution: {neutral_counts}")
print(f"Calculated Class Weights: {neutral_weights_tensor}")

neutral_model, neutral_acc, neutral_tokenizer = train_and_evaluate(
    tn_texts, tn_labels, vn_texts, vn_labels,
    num_labels=2, custom_lr=3e-5, epochs=6, weights_tensor=neutral_weights_tensor
)
print(f"✅ 1단계 검증 정확도 (중립 여부): {neutral_acc:.4f}")


# === 2단계: 원본 데이터 정제 ===
print("\n" + "=" * 50)
print("2단계: 중립 분류기로 원본 데이터 정제 시작")

try:
    df_final = pd.read_csv(FINAL_DATA_FILE, encoding='utf-8')
except Exception:
    df_final = pd.read_csv(FINAL_DATA_FILE, encoding='cp949')

df_predicted_neutral = predict_data(neutral_model, neutral_tokenizer, df_final)

# ✅ 확신 높은 중립(0.75 이상)만 제거
df_purified = df_predicted_neutral[df_predicted_neutral['neutral_prob'] < 0.75].copy()
df_purified.drop(columns=['predicted_label', 'neutral_prob'], inplace=True)

print(f"총 원본 데이터: {len(df_final)}개, 중립 확신>0.75 데이터 삭제 후: {len(df_purified)}개")


# === 3단계: 긍정/부정 분류 학습 ===
df_labeled = df_three[df_three['sentiment_three'] != '중립'].copy()
df_labeled.rename(columns={'sentiment_three': 'sentiment_purified'}, inplace=True)

purified_texts = df_labeled['text'].tolist()
purified_labels_text = df_labeled['sentiment_purified'].tolist()
purified_labels = [0 if l == '부정' else 1 for l in purified_labels_text]

tp_texts, vp_texts, tp_labels, vp_labels = train_test_split(
    purified_texts, purified_labels, test_size=0.2, random_state=42, stratify=purified_labels
)

purified_counts = Counter(tp_labels)
tp_total = len(tp_labels)
purified_weights = {i: tp_total / (len(purified_counts) * count) for i, count in purified_counts.items()}
purified_weights_tensor = torch.tensor([purified_weights[i] for i in sorted(purified_counts.keys())],
                                       dtype=torch.float).to(device)

print("\n" + "=" * 50)
print("3단계: 순수 Binary 분류기 학습 시작")
print(f"Original Training Distribution: {purified_counts}")
print(f"Calculated Class Weights: {purified_weights_tensor}")

final_binary_model, final_binary_acc, final_binary_tokenizer = train_and_evaluate(
    tp_texts, tp_labels, vp_texts, vp_labels,
    num_labels=2, custom_lr=2e-5, epochs=5, weights_tensor=purified_weights_tensor
)
print(f"✅ 3단계 최종 검증 정확도: {final_binary_acc:.4f}")


# === 4단계: 최종 예측 및 저장 ===
print("\n" + "=" * 50)
print("4단계: 최종 정제된 데이터셋 예측 및 저장")

df_final_predicted = predict_data(
    final_binary_model,
    final_binary_tokenizer,
    df_purified,
    "predicted_final_purified_binary_v2.csv"
)

print(f"✅ 최종 결과 파일 저장 완료: predicted_final_purified_binary_v2.csv")
print(f"총 최종 분석 댓글 수: {len(df_final_predicted)}")

print("\n🎯 최종 보고용 결과")
print(f"1단계 (중립 분류) 정확도: {neutral_acc:.4f}")
print(f"3단계 (Binary) 정확도: {final_binary_acc:.4f}")
