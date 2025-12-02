import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, logging
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "사용중")

logging.set_verbosity_error()

# -------------------------------
# 1️⃣ 데이터 불러오기
# -------------------------------
train_data_path = "labeled_output#_4.csv"
dataset = pd.read_csv(train_data_path)[['text', 'sentiment']].dropna(axis=0)

texts = list(dataset['text'].values)
labels = list(dataset['sentiment'].values)  # 0=부정, 1=긍정

print("데이터 확인")
for i in range(3):
    print(f"코로나 백신 레딧: {texts[i][:20]}, \t긍부정 라벨:{labels[i]}")
print(f"학습 데이터 수: {len(texts)}")
print(f"긍정 데이터 수: {labels.count(1)}")
print(f"부정 데이터 수: {labels.count(0)}")

# -------------------------------
# 2️⃣ Dataset 정의
# -------------------------------
class RedditDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# 3️⃣ 학습/검증 데이터 분리
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
texts_train, texts_val, labels_train, labels_val = train_test_split(
    texts, labels, test_size=0.2, random_state=2025, stratify=labels
)

train_dataset = RedditDataset(texts_train, labels_train, tokenizer)
val_dataset = RedditDataset(texts_val, labels_val, tokenizer)

# -------------------------------
# 4️⃣ 클래스 불균형 대응 (Weighted + Oversampling)
# -------------------------------
num_pos = labels_train.count(1)
num_neg = labels_train.count(0)

weight_pos = len(labels_train) / (2 * num_pos)
weight_neg = len(labels_train) / (2 * num_neg)
class_weights = torch.tensor([weight_neg, weight_pos]).to(device)

# Focal Loss 정의
class FocalLoss(nn.Module):
    def __init__(self, gamma=2., alpha=None, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = nn.CrossEntropyLoss(weight=self.alpha, reduction='none')(logits, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

criterion = FocalLoss(gamma=2., alpha=class_weights)

# Oversampling 준비
weights_per_sample = [0] * len(labels_train)
for i, label in enumerate(labels_train):
    weights_per_sample[i] = 1. / (num_pos if label == 1 else num_neg)
sampler = WeightedRandomSampler(weights_per_sample, num_samples=len(weights_per_sample), replacement=True)

# -------------------------------
# 5️⃣ DataLoader
# -------------------------------
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# 6️⃣ 모델 & 옵티마이저
# -------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/deberta-v3-base",
    num_labels=2,
    use_safetensors=True,
    trust_remote_code=True
)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
epochs = 10
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)*epochs
)

# -------------------------------
# 7️⃣ 학습 루프 + EarlyStopping
# -------------------------------
best_val_acc = 0
patience = 3
trigger_times = 0

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels_batch = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1)
        correct += (preds == labels_batch).sum().item()

    train_acc = correct / len(train_dataset)
    train_loss = total_loss / len(train_loader)

    # Validation
    model.eval()
    correct_val = 0
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct_val += (preds == labels_batch).sum().item()

    val_acc = correct_val / len(val_dataset)
    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}")

    # EarlyStopping 체크
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        trigger_times = 0
        torch.save(model.state_dict(), "best_model_focal.pt")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

# -------------------------------
# 8️⃣ 최종 모델 저장
# -------------------------------
save_path = "deberta_v3_focal_best"
model.cpu()
model.save_pretrained(save_path)
print("모델 저장 완료")
