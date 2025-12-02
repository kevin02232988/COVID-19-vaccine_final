import os
import pandas as pd
import torch
import torch.nn.functional as F
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
# Dataset class (동일)
# ======================
class KoDataset(Dataset):
    # ... (생략: 기존 코드와 동일) ...
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


# 🌟 CHANGE 2: Focal Loss 클래스 유지 (단, Main에서 weight=None으로 호출)
class FocalLoss(torch.nn.Module):
    """Focal Loss 구현. 오버샘플링 후에는 weight(alpha) 없이 gamma만 사용."""

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)

        logpt_ce = -logpt

        pt_for_target = pt.gather(1, target.view(-1, 1)).squeeze()
        logpt_for_target = logpt_ce.gather(1, target.view(-1, 1)).squeeze()

        focal_term = (1.0 - pt_for_target) ** self.gamma
        loss = focal_term * logpt_for_target

        # 오버샘플링으로 데이터 균형을 맞췄기 때문에, weight는 보통 None으로 전달됨
        if self.weight is not None:
            weight = self.weight.gather(0, target)
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# ======================
# Train (Focal Loss 적용)
# ======================
# 🌟 CHANGE 3: FocalLoss 사용, class_weights는 None으로 전달됨
def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0

    # 오버샘플링 후에도 학습의 '질'적 개선을 위해 FocalLoss 사용
    loss_fn = FocalLoss(weight=None, gamma=2.0)  # weight=None

    for batch in tqdm(loader, desc="Training", ncols=80):
        optimizer.zero_grad()
        # ... (이하 동일)
        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask
        )

        loss = loss_fn(outputs.logits, labels)  # Focal Loss 적용

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ======================
# Eval (Focal Loss 적용)
# ======================
# 🌟 CHANGE 4: FocalLoss 사용, class_weights는 None으로 전달됨
def eval_epoch(model, loader, device, class_weights=None):
    model.eval()
    preds, trues = [], []
    total_loss = 0

    loss_fn = FocalLoss(weight=None, gamma=2.0)  # weight=None

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", ncols=80):
            # ... (이하 동일)
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )

            logit = outputs.logits

            loss = loss_fn(logit, labels)
            total_loss += loss.item()
            # ... (이하 동일)
            pred = torch.argmax(logit, dim=1).cpu().numpy()

            preds.extend(pred)
            trues.extend(batch["label"].numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='binary', zero_division=0)

    return acc, avg_loss, f1


# ======================
# Main (오버샘플링 + Focal Loss 적용)
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

    # ===== 데이터 로딩 및 Split =====
    texts = df["text"].astype(str).tolist()
    labels = df["sentiment"].astype(int).tolist()

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )

    # 🌟 CHANGE 5: RandomOverSampler 적용
    print("\n🔄 Applying Random Over Sampling to Training Data...")
    ros = RandomOverSampler(random_state=42)

    train_texts_array = np.array(train_texts).reshape(-1, 1)
    train_labels_array = np.array(train_labels)

    combined_train_data = np.hstack((train_texts_array, train_labels_array.reshape(-1, 1)))

    resampled_data, _ = ros.fit_resample(combined_train_data, train_labels_array)

    train_texts_resampled = resampled_data[:, 0].tolist()
    train_labels_resampled = resampled_data[:, 1].astype(int).tolist()

    print(f"✅ Training Samples Before OverSampling: {len(train_texts)}")
    print(f"✅ Training Samples After OverSampling: {len(train_texts_resampled)}")

    # 🌟 CHANGE 6: Focal Loss를 사용하더라도 오버샘플링을 했기 때문에 class_weights는 None으로 설정
    class_weights = None

    print(f"📊 Class Counts Before Sampling: {np.bincount(train_labels_array)}")
    print(f"📊 Class Counts After Sampling: {np.bincount(np.array(train_labels_resampled))}")
    print("🔥 Focal Loss (Gamma=2.0)가 적용됩니다.")

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
    # 🌟 CHANGE 7: 오버샘플링된 데이터셋 사용
    train_dataset = KoDataset(train_texts_resampled, train_labels_resampled, tokenizer)
    val_dataset = KoDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Optimizer 및 Scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    num_epochs = 3
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

        # 🌟 CHANGE 8: class_weights=None으로 함수 호출 (Focal Loss는 내부적으로 gamma만 사용)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights=None)
        val_acc, val_loss, val_f1 = eval_epoch(model, val_loader, device, class_weights=None)

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")

    print("\n🎉 Training Completed!")


if __name__ == "__main__":
    main()