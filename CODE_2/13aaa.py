import os
import pandas as pd
import torch
import torch.nn.functional as F  # 🌟 CHANGE 1: F.log_softmax 등을 위해 추가
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


# ======================
# Dataset class (동일)
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


# 🌟 CHANGE 2: Focal Loss 클래스 추가
class FocalLoss(torch.nn.Module):
    """
    Focal Loss 구현 (Cross-Entropy Loss 개선 버전)
    """

    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight  # 기존 클래스 가중치(alpha)가 여기에 전달됨
        self.reduction = reduction

    def forward(self, input, target):
        # input: 모델의 로짓(logits) (N, C)
        # target: 레이블 (N)

        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)  # 예측 확률

        # Cross Entropy Loss
        logpt_ce = -logpt

        # 정답 클래스에 해당하는 확률값 추출
        pt_for_target = pt.gather(1, target.view(-1, 1)).squeeze()
        logpt_for_target = logpt_ce.gather(1, target.view(-1, 1)).squeeze()

        # 변조 계수 (Modulating Factor): (1 - pt)^gamma
        focal_term = (1.0 - pt_for_target) ** self.gamma

        # Focal Loss: (1 - pt)^gamma * CE
        loss = focal_term * logpt_for_target

        if self.weight is not None:
            # 클래스 가중치 (alpha) 적용
            weight = self.weight.gather(0, target)
            loss = loss * weight

        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


# ======================
# Train (Focal Loss 적용)
# ======================
def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0

    # 🌟 CHANGE 3: FocalLoss 사용 (기존 class_weights를 FocalLoss의 weight로 전달)
    loss_fn = FocalLoss(weight=class_weights.to(device) if class_weights is not None else None, gamma=2.0)

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
# Eval (Focal Loss 적용)
# ======================
def eval_epoch(model, loader, device, class_weights=None):
    model.eval()
    preds, trues = [], []
    total_loss = 0

    # 🌟 CHANGE 4: FocalLoss 사용
    loss_fn = FocalLoss(weight=class_weights.to(device) if class_weights is not None else None, gamma=2.0)

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

            # Val Loss 계산 (Focal Loss 적용)
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
# Main (Focal Loss 적용)
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

    # 🌟 CHANGE 5: 클래스 가중치는 Focal Loss의 alpha 역할로 그대로 사용
    train_labels_array = np.array(train_labels)
    class_counts = np.bincount(train_labels_array)
    num_classes = len(class_counts)

    # 가중치 계산
    total_samples = len(train_labels_array)
    small_epsilon = 1e-6
    class_weights = total_samples / (num_classes * (class_counts + small_epsilon))
    class_weights = torch.tensor(class_weights, dtype=torch.float)

    print(f"\n📊 Class Counts: {class_counts}")
    print(f"⚖️ Calculated Class Weights (for Focal Loss's alpha): {class_weights.tolist()}")

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Model/Tokenizer 로딩
    model_name = "monologg/koelectra-base-v3-discriminator"

    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        use_safetensors=True
    ).to(device)

    # Dataset/Loader 정의
    train_dataset = KoDataset(train_texts, train_labels, tokenizer)
    val_dataset = KoDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Optimizer (안정적인 학습률 2e-5 적용)
    optimizer = AdamW(model.parameters(), lr=2e-5)  # 💡 학습률 2e-5 적용

    # Scheduler
    num_epochs = 3  # 💡 Epoch 3회로 유지 (Focal Loss 효과만 확인)
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

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, class_weights)
        val_acc, val_loss, val_f1 = eval_epoch(model, val_loader, device, class_weights)  # F1-Score 반환

        print(
            f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")  # F1-Score 출력

    print("\n🎉 Training Completed!")


if __name__ == "__main__":
    main()