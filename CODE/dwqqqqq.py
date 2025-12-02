# optimized_electra_pipeline.py
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import (
    ElectraTokenizer,
    ElectraForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from tqdm import tqdm

# ==========================
# 설정 (필요시 수정)
# ==========================
MODEL_NAME = "monologg/koelectra-base-v3-discriminator"
THREE_FILE = "BERT_labeled_three.csv"
FINAL_FILE = "FINAL_DATA_FILTERED_#TRUE.csv"
OUTPUT_FILE = "FINAL_no_neutral_binary_prediction_v3_optimized.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 학습 하이퍼파라미터
MAX_LEN = 256               # 128 -> 256으로 늘림
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 8                  # epochs 적절히 설정 (너 상황에 맞게)
ACCUMULATION_STEPS = 2      # effective batch size 증가
WARMUP_RATIO = 0.06         # scheduler warmup
WEIGHT_DECAY = 0.01
SEED = 42
SAVE_DIR = "./saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# 고정시드
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# ==========================
# 간단한 EDA 증강 (데이터 적을 때 사용)
# 매우 보수적으로 설계 (의미 훼손 최소화)
# ==========================
def random_deletion(words, p=0.1):
    if len(words) == 1:
        return words
    new_words = [w for w in words if random.random() > p]
    if len(new_words) == 0:
        return [random.choice(words)]
    return new_words

def random_swap(words, n_swaps=1):
    words = words.copy()
    for _ in range(n_swaps):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words

def eda_augment(text, prob_del=0.08, prob_swap=0.06):
    # 매우 보수적: 토큰을 공백 기준으로 나눔 (한국어에서 완벽하진 않음)
    words = text.split()
    if len(words) <= 2:
        return text
    if random.random() < 0.5:
        words = random_deletion(words, p=prob_del)
    if random.random() < 0.5:
        words = random_swap(words, n_swaps=1)
    return " ".join(words)

# ==========================
# Dataset
# ==========================
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LEN, augment=False):
        self.texts = list(texts)
        self.labels = None if labels is None else list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        if self.augment and self.labels is not None:
            # 증강은 학습 데이터에서만 적용
            text = eda_augment(text)

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
        )
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# ==========================
# LLRD (Layer-wise LR decay) 함수
# ==========================
def get_optimizer_grouped_parameters(model, base_lr, layer_decay=0.95):
    # model.electra.encoder.layer는 bottom -> top 순서
    # embeddings, encoder.layer[0] ... encoder.layer[-1], classifier
    lr = base_lr
    no_decay = ["bias", "LayerNorm.weight"]

    # collect layers
    layers = []
    layers.append((model.electra.embeddings, "embeddings"))
    # encoder layers
    for i, layer in enumerate(model.electra.encoder.layer):
        layers.append((layer, f"encoder.layer.{i}"))
    # pooler/other is absent for ELECTRA; classifier last
    grouped_parameters = []

    # assign lr with decay: lower layers smaller lr
    n_layers = len(layers)
    for i, (layer_module, name) in enumerate(layers):
        # decay factor increases for lower layers (i small => lower lr)
        scale = layer_decay ** (n_layers - i - 1)
        layer_lr = lr * scale
        grouped_parameters.append(
            {
                "params": [p for n, p in layer_module.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
                "lr": layer_lr,
            }
        )
        grouped_parameters.append(
            {
                "params": [p for n, p in layer_module.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }
        )

    # classifier params - top lr
    if hasattr(model, "classifier"):
        grouped_parameters.append(
            {
                "params": [p for n, p in model.classifier.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": WEIGHT_DECAY,
                "lr": lr,
            }
        )
        grouped_parameters.append(
            {
                "params": [p for n, p in model.classifier.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": lr,
            }
        )

    return grouped_parameters

# ==========================
# 학습/평가 루틴
# - model(..., labels=...)로 내부 loss 사용 (HuggingFace 권장)
# - WeightedRandomSampler로 클래스 불균형 보정
# - gradient accumulation 적용
# - scheduler 적용
# ==========================
def train_and_evaluate(train_texts, train_labels, val_texts, val_labels, num_labels, config):
    print(f"Training config: LR={config['lr']}, epochs={config['epochs']}, max_len={config['max_len']}")
    tokenizer = ElectraTokenizer.from_pretrained(MODEL_NAME)
    model = ElectraForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels, use_safetensors=True)
    model.to(DEVICE)

    # Dataset & Sampler (WeightedRandomSampler for imbalance)
    train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len=config["max_len"], augment=True)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len=config["max_len"], augment=False)

    # Weighted sampler
    class_counts = Counter(train_labels)
    sample_weights = [1.0 / class_counts[l] for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config["train_bs"], sampler=sampler, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=config["valid_bs"], shuffle=False)

    # Optimizer with LLRD parameter groups
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(model, base_lr=config["lr"], layer_decay=0.95)
    optimizer = AdamW(optimizer_grouped_parameters, lr=config["lr"], weight_decay=WEIGHT_DECAY)

    # Scheduler
    total_steps = (len(train_loader) // config["accum_steps"] + (1 if len(train_loader) % config["accum_steps"] else 0)) * config["epochs"]
    warmup_steps = max(1, int(total_steps * config["warmup_ratio"]))
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_val_acc = 0.0
    best_model_path = None

    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{config['epochs']}")
        for step, batch in pbar:
            # move to device
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)  # labels 포함되어 있으므로 outputs.loss 사용
            loss = outputs.loss / config["accum_steps"]
            loss.backward()
            running_loss += loss.item() * config["accum_steps"]

            if (step + 1) % config["accum_steps"] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                pbar.set_postfix({"loss": f"{running_loss / ((step+1)):.4f}"})

        # --- validation ---
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items()}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds.cpu() == inputs["labels"].cpu()).sum().item()
                total += len(preds)
        val_acc = correct / total if total > 0 else 0.0
        print(f"Epoch {epoch+1} validation accuracy: {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(SAVE_DIR, f"best_model_epoch{epoch+1}_acc{val_acc:.4f}.safetensors")
            # save using save_pretrained (will create pytorch_model.bin by default) - prefer save_pretrained
            model.save_pretrained(os.path.dirname(best_model_path), safe_serialization=True)
            print(f"Saved best model -> {os.path.dirname(best_model_path)}")

    return model, tokenizer, best_val_acc, best_model_path

# ==========================
# 예측 함수 (데이터프레임 기반)
# ==========================
def predict_data(model, tokenizer, df_target, output_path=None, max_len=MAX_LEN, batch_size=VALID_BATCH_SIZE):
    texts = df_target["text"].astype(str).tolist()
    dataset = TextDataset(texts, None, tokenizer, max_len=max_len, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting"):
            inputs = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**inputs)
            preds.extend(torch.argmax(outputs.logits, dim=1).cpu().numpy())

    df_target = df_target.copy()
    df_target["predicted_label"] = preds

    if output_path:
        df_target.to_csv(output_path, index=False, encoding="utf-8")
        print(f"💾 결과 저장 완료: {output_path}")

    return df_target

# ==========================
# 메인 파이프라인
# ==========================
def main():
    print("Using device:", DEVICE)
    # Step 1. 라벨링 데이터 로드
    df_three = pd.read_csv(THREE_FILE)
    print(f"라벨링 데이터 로드 완료: {len(df_three)}개")

    # Step 2. 중립 vs 비중립 (1단계)
    df_three["is_neutral"] = df_three["sentiment_three"].apply(lambda x: 1 if x == "중립" else 0)
    df_temp = df_three[["text", "is_neutral"]].reset_index(drop=True)
    texts = df_temp["text"].tolist()
    labels = df_temp["is_neutral"].tolist()

    train_t, val_t, train_l, val_l = train_test_split(texts, labels, test_size=0.2, random_state=SEED, stratify=labels)

    print("1단계 학습 라벨 분포:", Counter(train_l))

    config_stage1 = {
        "lr": LR,
        "epochs": max(3, EPOCHS//2),  # 중립필터는 epoch 작게도 시도 가능
        "train_bs": TRAIN_BATCH_SIZE,
        "valid_bs": VALID_BATCH_SIZE,
        "accum_steps": ACCUMULATION_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "max_len": MAX_LEN,
    }

    neutral_model, neutral_tokenizer, neutral_acc, _ = train_and_evaluate(train_t, train_l, val_t, val_l, num_labels=2, config=config_stage1)
    print(f"✅ 1단계 (중립 필터링) 최고 Validation Accuracy: {neutral_acc:.4f}")

    # Step 3. 원본 데이터 로드 & 중립 제거
    df_final = pd.read_csv(FINAL_FILE)
    df_pred_neutral = predict_data(neutral_model, neutral_tokenizer, df_final, output_path=None)
    df_filtered = df_pred_neutral[df_pred_neutral["predicted_label"] == 0].copy()
    print(f"총 {len(df_final)} → 중립 제거 후 {len(df_filtered)}개 남음")

    # Step 4. 긍/부정 학습 데이터 준비 (2단계)
    df_pure = df_three[df_three["sentiment_three"] != "중립"].copy().reset_index(drop=True)
    df_pure["label"] = df_pure["sentiment_three"].map({"부정": 0, "긍정": 1})
    texts_p = df_pure["text"].tolist()
    labels_p = df_pure["label"].tolist()

    tp_t, vp_t, tp_l, vp_l = train_test_split(texts_p, labels_p, test_size=0.2, random_state=SEED, stratify=labels_p)
    print("2단계 학습 라벨 분포:", Counter(tp_l))

    config_stage2 = {
        "lr": LR,
        "epochs": EPOCHS,
        "train_bs": TRAIN_BATCH_SIZE,
        "valid_bs": VALID_BATCH_SIZE,
        "accum_steps": ACCUMULATION_STEPS,
        "warmup_ratio": WARMUP_RATIO,
        "max_len": MAX_LEN,
    }

    binary_model, binary_tokenizer, binary_acc, best_path = train_and_evaluate(tp_t, tp_l, vp_t, vp_l, num_labels=2, config=config_stage2)
    print(f"✅ 2단계 (긍/부정) 최고 Validation Accuracy: {binary_acc:.4f}")

    # Step 5. 최종 예측 & 저장
    df_final_pred = predict_data(binary_model, binary_tokenizer, df_filtered, output_path=OUTPUT_FILE)
    print("\n" + "=" * 60)
    print("🎯 최종 보고 요약")
    print(f"1단계 (중립 필터링) Best Validation Accuracy: {neutral_acc:.4f}")
    print(f"2단계 (긍/부정) Best Validation Accuracy: {binary_acc:.4f}")
    print(f"최종 분석 데이터 수 (정제 후): {len(df_final_pred)}")
    print("=" * 60)

if __name__ == "__main__":
    main()
