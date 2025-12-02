import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm

# 1️⃣ GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 2️⃣ 데이터셋 클래스 정의
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
    print(f"\n===== {csv_path} 모델 학습 시작 =====")

    # 데이터 로드
    df = pd.read_csv(csv_path)

    # CSV 이름에 따라 라벨 컬럼과 매핑 선택
    if "binary" in csv_path.lower():
        label_col = "sentiment_binary"
        label_map = {"부정": 0, "긍정": 1}
    elif "three" in csv_path.lower():
        label_col = "sentiment_three"
        label_map = {"부정": 0, "중립": 1, "긍정": 2}
    else:
        raise ValueError("CSV 파일 이름에 'binary' 또는 'three'가 포함되어야 합니다.")

    if label_col not in df.columns:
        raise ValueError(f"'{label_col}' 컬럼이 CSV에 없습니다. 현재 컬럼: {df.columns.tolist()}")

    texts = df['text'].astype(str).tolist()
    # 문자열 라벨을 숫자로 변환
    labels = [label_map[l] for l in df[label_col].tolist()]

    # 데이터 분리
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 토크나이저 및 데이터셋
    tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    train_dataset = TextDataset(train_texts, train_labels, tokenizer)
    val_dataset = TextDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 모델
    model = ElectraForSequenceClassification.from_pretrained(
        "monologg/koelectra-base-v3-discriminator",
        use_safetensors=True
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    epochs = 2

    # 학습
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Train Loss: {total_loss / len(train_loader):.4f}")

    # 검증
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
    print(f"✅ Validation Accuracy: {accuracy:.4f}")

    # 원본 데이터 예측
    final_df = pd.read_csv("FINAL_DATA_FILTERED_#TRUE.csv")
    final_texts = final_df['text'].astype(str).tolist()

    final_dataset = TextDataset(final_texts, [0]*len(final_texts), tokenizer)
    final_loader = DataLoader(final_dataset, batch_size=32)

    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch in tqdm(final_loader, desc="Predicting FINAL dataset"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1)
            preds_list.extend(preds.cpu().numpy())

    # 결과 저장
    final_df['predicted_label'] = preds_list
    final_df.to_csv(output_pred_path, index=False)
    print(f"💾 예측 결과 저장 완료: {output_pred_path}")
    print("========================================\n")

    return accuracy

# 4️⃣ 이진 분류 (binary)
binary_acc = train_and_evaluate(
    "BERT_labeled_binary.csv",
    num_labels=2,
    output_pred_path="predicted_binary_0.csv"
)

# 5️⃣ 삼분류 (three-class)
three_acc = train_and_evaluate(
    "BERT_labeled_three.csv",
    num_labels=3,
    output_pred_path="predicted_three_0.csv"
)

print("🎯 최종 결과")
print(f"Binary Validation Accuracy : {binary_acc:.4f}")
print(f"Three-class Validation Accuracy : {three_acc:.4f}")
