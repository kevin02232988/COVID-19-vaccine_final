import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
# === 모델 변경: Electra 대신 Bert 사용 ===
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
from collections import Counter

# 1️⃣ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# 2️⃣ 데이터셋 클래스 정의 (생략, 이전과 동일)
class TextDataset(Dataset):
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
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# 3️⃣ 학습 및 평가 함수
def train_and_evaluate(csv_path, num_labels, output_pred_path):
    # === BertForSequenceClassification 함수에 맞게 모델 이름 및 클래스 수정 ===
    MODEL_NAME = "bert-base-multilingual-cased"

    print(f"\n===== {csv_path} 모델 학습 시작 (모델: BERT, 클래스 수: {num_labels}) =====")

    # 데이터 로드 (이전과 동일)
    df = pd.read_csv(csv_path)

    if "binary" in csv_path.lower():
        label_col = "sentiment_binary";
        label_map = {"부정": 0, "긍정": 1}
        custom_lr = 2e-5;
        adjust_weights_manually = False

    elif "three" in csv_path.lower():
        label_col = "sentiment_three";
        label_map = {"부정": 0, "중립": 1, "긍정": 2}
        custom_lr = 1e-5;
        adjust_weights_manually = True

    else:
        raise ValueError("CSV 파일 이름에 'binary' 또는 'three'가 포함되어야 합니다.")

    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' 컬럼이 CSV에 없습니다. 현재 컬럼: {df.columns.tolist()}")

    texts = df['text'].astype(str).tolist()
    labels = [label_map[l] for l in df[label_col].tolist()]

    # 3.1. 학습/검증 데이터 분리 (생략)
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    print(f"Original Training Distribution: {Counter(train_labels)}")

    # 3.2. 클래스 가중치 계산 및 적용
    class_counts = Counter(train_labels)
    total_samples = len(train_labels)

    if adjust_weights_manually:
        # 🧪 [Three-Class] 수동 가중치 조정 전략: '중립'에 최대 페널티 부여
        weights = [
            0.50,  # 부정
            3.50,  # 중립 (최대 페널티)
            1.50  # 긍정
        ]
        weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
        print(f"Calculated Custom Class Weights (Order 0, 1, 2): {weights_tensor}")

    else:
        # [Binary] 자동 역빈도 가중치 사용
        class_weights = {i: total_samples / (num_labels * count) for i, count in class_counts.items()}
        weights_tensor = torch.tensor([class_weights[i] for i in sorted(class_weights.keys())], dtype=torch.float).to(
            device)
        print(f"Calculated Auto Class Weights (Order 0, 1): {weights_tensor}")

    # 3.3. 토크나이저 및 데이터셋
    # === BERT 토크나이저로 변경 ===
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 3.4. 모델 및 손실 함수
    # === BertForSequenceClassification으로 변경 ===
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels
        # BertForSequenceClassification은 'use_safetensors' 인수를 기본적으로 사용하지 않으므로 제거
    ).to(device)

    # === 클래스 가중치를 Loss Function에 적용 ===
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights_tensor)

    optimizer = AdamW(model.parameters(), lr=custom_lr)
    epochs = 8

    # 3.5. 학습, 3.6. 검증, 3.7. 예측 (이전과 동일)
    # ... (학습, 검증, 예측 루프는 이전과 동일) ...
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            # outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            # outputs의 키가 Bert 모델과 Electra 모델 간에 다를 수 있으므로, 인자를 명시
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = loss_fn(outputs.logits, inputs['labels'])

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1} | Train Loss: {total_loss / len(train_loader):.4f}")

    # 3.6. 검증 (생략)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            preds = torch.argmax(outputs.logits, dim=1)
            correct += (preds.cpu() == inputs['labels'].cpu()).sum().item()
            total += len(preds)
    accuracy = correct / total
    print(f"✅ Validation Accuracy: {accuracy:.4f}")

    # 3.7. 원본 데이터 예측 (생략)
    final_df = pd.read_csv("FINAL_DATA_FILTERED_#TRUE.csv")
    final_texts = final_df['text'].astype(str).tolist()
    final_dataset = TextDataset(final_texts, [0] * len(final_texts), tokenizer)
    final_loader = DataLoader(final_dataset, batch_size=32)

    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch in tqdm(final_loader, desc="Predicting FINAL dataset"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])
            preds = torch.argmax(outputs.logits, dim=1)
            preds_list.extend(preds.cpu().numpy())

    # 결과 저장 (파일명에 BERT 명시)
    final_df['predicted_label'] = preds_list
    final_df.to_csv(output_pred_path.replace("koelectra", "bert"), index=False)
    print(f"💾 예측 결과 저장 완료: {output_pred_path.replace('koelectra', 'bert')}")
    print("========================================\n")

    return accuracy


# 4️⃣ 이진 분류 (binary)
binary_acc = train_and_evaluate(
    "BERT_labeled_binary.csv",
    num_labels=2,
    output_pred_path="predicted_binary_lr_epochs_tuned_final_bert.csv"
)

# 5️⃣ 삼분류 (three-class)
three_acc = train_and_evaluate(
    "BERT_labeled_three.csv",
    num_labels=3,
    output_pred_path="predicted_three_lr_epochs_tuned_final_bert.csv"
)

print("🎯 최종 결과")
print(f"Binary Validation Accuracy : {binary_acc:.4f}")
print(f"Three-class Validation Accuracy : {three_acc:.4f}")