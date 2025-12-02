import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from tqdm import tqdm


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
# Train
# ======================
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0

    for batch in tqdm(loader, desc="Training", ncols=80):
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(device)
        attn_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attn_mask,
            labels=labels
        )

        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ======================
# Eval
# ======================
def eval_epoch(model, loader, device):
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", ncols=80):
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attn_mask
            )

            logit = outputs.logits
            pred = torch.argmax(logit, dim=1).cpu().numpy()

            preds.extend(pred)
            trues.extend(batch["label"].numpy())

    acc = accuracy_score(trues, preds)
    return acc


# ======================
# Main
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

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # Model
    model_name = "monologg/koelectra-base-v3-discriminator"

    tokenizer = ElectraTokenizer.from_pretrained(model_name)
    model = ElectraForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        # 💡 이 부분을 추가합니다.
        use_safetensors=True
    ).to(device)

    # Dataset/Loader
    train_dataset = KoDataset(train_texts, train_labels, tokenizer)
    val_dataset = KoDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Scheduler
    total_steps = len(train_loader) * 3  # 3 epoch
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    # Training
    for epoch in range(3):
        print(f"\n===== Epoch {epoch+1} / 3 =====")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_acc = eval_epoch(model, val_loader, device)

        print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

    print("\n🎉 Training Completed!")


if __name__ == "__main__":
    main()
