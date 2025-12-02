import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# 1️⃣ 환경 설정
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "사용 중")

# -------------------------------
# 2️⃣ 원본 데이터 불러오기
# -------------------------------
file_path = "FINAL_DATA_ROWS_#DELETED.csv"
df = pd.read_csv(file_path)

# text, created_at 컬럼 존재 확인
df['text'] = df['text'].fillna('').astype(str)
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')

# -------------------------------
# 3️⃣ Tokenizer & Dataset
# -------------------------------
class SentimentDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len=256):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }

tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base", use_fast=False)
dataset = SentimentDataset(df['text'].tolist(), tokenizer)

dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

# -------------------------------
# 4️⃣ 모델 불러오기
# -------------------------------
model_path = "deberta_v3_weighted_oversample"  # val acc 0.87 모델
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    use_safetensors=True,
    trust_remote_code=True
)
model.to(device)
model.eval()

# -------------------------------
# 5️⃣ 예측
# -------------------------------
preds = []

with torch.no_grad():
    for batch in tqdm(dataloader, desc="Predicting"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        batch_preds = torch.argmax(outputs.logits, dim=1)
        preds.extend(batch_preds.cpu().numpy())

df['sentiment_label'] = preds

# -------------------------------
# 6️⃣ 비율 확인
# -------------------------------
label_counts = df['sentiment_label'].value_counts()
print("\n라벨 분포:\n", label_counts)

plt.figure(figsize=(6,4))
label_counts.plot(kind='bar', color=['red','green'])
plt.xticks([0,1], ['Negative','Positive'])
plt.ylabel('Count')
plt.title('Sentiment Distribution')
plt.show()

# -------------------------------
# 7️⃣ 시계열 부정 비율 확인
# -------------------------------
# 일별 부정 비율 계산
daily_counts = df.groupby(df['created_at'].dt.date)['sentiment_label'].value_counts().unstack(fill_value=0)
daily_counts['negative_ratio'] = daily_counts.get(0,0) / (daily_counts.get(0,0) + daily_counts.get(1,0))

plt.figure(figsize=(10,5))
daily_counts['negative_ratio'].plot()
plt.ylabel('Negative Ratio')
plt.xlabel('Date')
plt.title('Daily Negative Sentiment Ratio')
plt.show()

# -------------------------------
# 8️⃣ 라벨링된 데이터 저장
# -------------------------------
output_file = "FINAL_DATA_ROWS_labeled.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n라벨링 완료, {output_file}로 저장됨")
