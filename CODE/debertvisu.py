import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# 1️⃣ 환경 설정
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "사용중")

sns.set(style="whitegrid")

# -------------------------------
# 2️⃣ 원본 데이터 로드
# -------------------------------
file_path = "FINAL_DATA_ROWS_#DELETED.csv"
df = pd.read_csv(file_path)

# text와 created_at 처리
df['text'] = df['text'].fillna('').astype(str)
df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')  # datetime 변환

# -------------------------------
# 3️⃣ 모델 & 토크나이저 로드
# -------------------------------
model_path = "deberta_v3_weighted_oversample"   # fine-tuned model folder

# ❗ tokenizer는 pretrained deberta에서 가져와야 함
tokenizer = AutoTokenizer.from_pretrained(
    "microsoft/deberta-v3-base",
    use_fast=False
)

model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    num_labels=2,
    use_safetensors=True,
    trust_remote_code=True
)
model.to(device)


# -------------------------------
# 4️⃣ 모델로 라벨링
# -------------------------------
batch_size = 32
labels_pred = []

for i in range(0, len(df), batch_size):
    batch_texts = df['text'][i:i+batch_size].tolist()
    encodings = tokenizer(batch_texts, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        labels_pred.extend(preds.cpu().numpy())

df['sentiment_label'] = labels_pred
print("✅ 라벨링 완료")
print(df['sentiment_label'].value_counts())

# -------------------------------
# 5️⃣ 라벨 분포 시각화
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x='sentiment_label', data=df)
plt.title("Sentiment Label Distribution")
plt.xlabel("Label (0:Negative, 1:Positive)")
plt.ylabel("Count")
plt.show()

# -------------------------------
# 6️⃣ 시계열 부정 비율 분석
# -------------------------------
df['month'] = df['created_at'].dt.to_period('M')
monthly_neg_ratio = df.groupby('month')['sentiment_label'].apply(lambda x: (x==0).mean())

plt.figure(figsize=(10,4))
monthly_neg_ratio.plot(marker='o')
plt.title("Monthly Negative Sentiment Ratio")
plt.xlabel("Month")
plt.ylabel("Negative Ratio")
plt.grid(True)
plt.show()

# -------------------------------
# 7️⃣ 라벨링 결과 저장
# -------------------------------
output_file = "FINAL_DATA_ROWS_#DELETED_labeled.csv"
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"✅ 라벨링된 데이터 저장 완료: {output_file}")
