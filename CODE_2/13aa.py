import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm
import numpy as np
# 🌟 CHANGE 1: imbalanced-learn 라이브러리 추가
from imblearn.over_sampling import RandomOverSampler


# ======================
# Dataset class
# ======================
class KoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])

        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "label": torch.tensor(label, dtype=torch.long)
        }


# ======================
# Train (오버샘플링 적용으로 클래스 가중치 사용 중단)
# ======================
# 오버샘플링 후에는 데이터 자체가 균형을 이루므로,
# train_epoch과 eval_epoch 함수에서 class_weights=None인 경우를 사용하도록 유지합니다.
def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0

    # 손실 함수 정의 (CrossEntropyLoss 사용)
    # 🌟 CHANGE 2: 오버샘플링 후에는 class_weights를 전달하지 않거나, 전달해도 None으로 처리됨
    if class_weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    for batch in tqdm(loader, desc="Training", ncols=80):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )

        loss = loss_fn(outputs.logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ======================
# Eval (Val Loss, Acc, F1-Score 계산)
# ======================
def eval_epoch(model, loader, device, class_weights=None):
    model.eval()
    preds, trues = [], []
    total_loss = 0

    # 손실 함수 정의
    if class_weights is not None:
        loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    else:
        loss_fn = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", ncols=80):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )

            logit = outputs.logits

            # Val Loss 계산
            loss = loss_fn(logit, labels)
            total_loss += loss.item()

            pred = torch.argmax(logit, dim=1).cpu().numpy()

            preds.extend(pred)
            trues.extend(batch["label"].numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(trues, preds)
    # F1-Score 계산 (불균형 데이터에서 핵심 지표)
    f1 = f1_score(trues, preds, average='binary', zero_division=0)

    return acc, avg_loss, f1


# ======================
# Main (오버샘플링 적용)
# ======================
def main():
    # ★ CSV 파일명 자동 고정 ★
    csv_file = "labeled_output#.csv"

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, csv_file)

    print(f"📄 Loading CSV File → {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {csv_path}")

    df = pd.read_csv(csv_path)

    # ===== 읽는 컬럼 수정됨 =====
    texts = df["text"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()

    # ===== Split =====
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 🌟 CHANGE 3: RandomOverSampler 적용
    print("\n🔄 Applying Random Over Sampling to Training Data...")
    ros = RandomOverSampler(random_state=42)

    # train_texts와 train_labels을 배열로 변환
    train_texts_array = np.array(train_texts).reshape(-1, 1)
    train_labels_array = np.array(train_labels)

    # RandomOverSampler는 X(특징)와 y(레이블)를 모두 받음.
    # 텍스트는 임베딩이 아니므로, 텍스트 배열 자체를 X로 전달하고, 레이블을 y로 전달
    combined_train_data = np.hstack((train_texts_array, train_labels_array.reshape(-1, 1)))

    # 오버샘플링 수행: X와 y를 함께 샘플링하고 분리 (레이블만 fit_resample의 두 번째 인자로 사용)
    resampled_data, _ = ros.fit_resample(combined_train_data, train_labels_array)

    # 오버샘플링된 데이터 분리
    train_texts_resampled = resampled_data[:, 0].tolist()
    train_labels_resampled = resampled_data[:, 1].astype(int).tolist()

    print(f"✅ Training Samples Before OverSampling: {len(train_texts)}")
    print(f"✅ Training Samples After OverSampling: {len(train_texts_resampled)}")

    # 🌟 CHANGE 4: 오버샘플링 후에는 클래스 가중치가 거의 1:1이 되므로, 가중치 사용을 중단합니다.
    # 이전 코드에서 사용하던 가중치 계산 및 출력 코드는 제거합니다.
    class_weights = None

    print(f"📊 Class Counts Before Sampling: {np.bincount(train_labels_array)}")
    print(f"📊 Class Counts After Sampling: {np.bincount(np.array(train_labels_resampled))}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\nDevice:", device)

    # Model/Tokenizer 로딩
    model_name = "monologg/koelectra-base-v3-discriminator"

    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        use_safetensors=True
    ).to(device)

    # Dataset/Loader 정의
    # 🌟 CHANGE 5: 오버샘플링된 데이터셋 사용
    train_dataset = KoDataset(train_texts_resampled, train_labels_resampled, tokenizer)
    val_dataset = KoDataset(val_texts, val_labels, tokenizer)

    # 배치 사이즈는 훈련 데이터가 늘어났으므로 16으로 유지하거나 32로 늘릴 수 있습니다.
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Optimizer (안정적인 학습률 2e-5 적용)
    optimizer = AdamW(model.parameters(), lr=2e-5)  # 💡 학습률 2e-5 적용

    # Scheduler
    num_epochs = 3  # 💡 에포크 3회로 유지 (오버샘플링 효과 확인)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training
    print(f"\n🚀 Start Training for {num_epochs} Epochs...")
    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1} / {num_epochs} =====")

        # 🌟 CHANGE 6: class_weights=None으로 함수 호출
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights=None)
        val_acc, val_loss, val_f1 = eval_epoch(model, val_loader, device, class_weights=None)  # F1-Score 반환

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    print("\n🎉 Training Completed!")


if __name__ == "__main__":
    main()