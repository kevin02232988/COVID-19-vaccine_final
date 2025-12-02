import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ------------------- 파일 및 설정 정의 -------------------
# 학습 데이터 경로: 복구된 파일로 변경
FILE_USER_LABELED = "Real_rabel_labeled_CLEANED.csv"
FILE_FULL_DATA = "Real_Final.csv" # 전체 98,277건 데이터
OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"

# ------------------- 1. 데이터 로드 및 전처리 -------------------

# 1-1. 학습 데이터 (9,827건) 로드
df_train = pd.read_csv(FILE_USER_LABELED).fillna('')
df_train = df_train[df_train['sentiment'].isin(['positive', 'negative'])].copy() # 소문자로 정규화된 라벨 사용

# 1-2. 라벨 매핑 (Negative: 0, Positive: 1)
label_map = {'negative': 0, 'positive': 1}
df_train['labels'] = df_train['sentiment'].map(label_map)

# 1-3. 전체 예측 대상 데이터 (98,277건) 로드
df_predict = pd.read_csv(FILE_FULL_DATA).fillna('')

print(f"[INFO] 학습 데이터셋 크기: {len(df_train)}건")
print(f"[INFO] 전체 예측 대상 데이터 크기: {len(df_predict)}건")

# ------------------- 2. Dataset 및 Tokenizer 준비 -------------------
MODEL_NAME = 'bert-base-multilingual-cased'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings.input_ids)

# 데이터 분리 (학습: 80%, 검증: 20%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train['text'].tolist(),
    df_train['labels'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df_train['labels']
)

# 인코딩
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=128)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=128)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

# ------------------- 3. 모델 학습 (Fine-tuning) -------------------
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    # pos_label=1은 Positive (1)을 긍정 클래스로 간주함을 의미
    precision, recall, f1, _ = precision_recall_scores = precision_recall_fscore_support(p.label_ids, preds, average='binary', pos_label=1)
    acc = accuracy_score(p.label_ids, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# 학습 인자 설정 (수정된 부분: evaluation_strategy 제거 및 eval_strategy 사용)
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    # evaluation_strategy 대신 eval_strategy 사용
    eval_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n--- 3. BERT 모델 학습 시작 (약 9,827건) ---")
trainer.train()

# ------------------- 4. 모델 예측 및 최종 저장 -------------------

# 4-1. 예측 대상 데이터셋 준비
df_predict['text'] = df_predict['text'].astype(str).tolist()
predict_texts = df_predict['text'].tolist()
predict_encodings = tokenizer(predict_texts, truncation=True, padding=True, max_length=128)
predict_dataset = SentimentDataset(predict_encodings)

# 4-2. 예측 실행
print("\n--- 4. 전체 데이터셋 감정 예측 시작 (98,277건) ---")
predictions = trainer.predict(predict_dataset)
predicted_labels = np.argmax(predictions.predictions, axis=1)

# 4-3. 라벨 디코딩 및 저장
# 0 -> Negative, 1 -> Positive
label_decode = {0: 'Negative', 1: 'Positive'}
df_predict['Predicted_Sentiment'] = [label_decode[label] for label in predicted_labels]

# 4-4. 최종 CSV 저장
df_predict.to_csv(OUTPUT_PREDICTED_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ 최종 분석 데이터셋 생성 완료! 총 {len(df_predict)}건")
print(f"💾 파일 저장 완료: '{OUTPUT_PREDICTED_FILE}'")