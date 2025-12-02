import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import ElectraTokenizer, ElectraForSequenceClassification
from transformers import get_linear_schedule_with_warmup, logging
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device, "사용중")

# 경고제거
logging.set_verbosity_error()

train_data_path = "data/ratings_train2.txt"

dataset = pd.read_csv(train_data_path, sep='\t').dropna(axis=0)
text = list(dataset['document'].values)
label = dataset['label'].values

num_to_print = 3
print("데이터 확인")
for i in range(num_to_print):
    print(f"영화 리뷰: {text[i][:20]}, \t긍부정 라벨:{label[i]}")
print(f"학습 데이터 수: {len(text)}")
print(f"긍정 데이터 수: {list(label).count(0)}")
print(f"부정 데이터 수: {list(label).count(1)}")

tokenizer = ElectraTokenizer.from_pretrained('koelectra-base-v3-discriminator')
inputs = tokenizer(text, truncation=True, max_length=256, add_special_tokens=True, padding="max_length")
input_ids = inputs['input_ids']
attention_mask = inputs['attention_mask']
print("토큰화")
for i in range(num_to_print):
    print(f"\n{i+1}번째 데이터")
    print("토큰")
    print(f"{input_ids[i]}")
    print("어텐션 마스크")
    print(f"{attention_mask[i]}")

train, validation, train_y, validation_y = train_test_split(input_ids, label, test_size=0.2, random_state=2025)
train_masks, validation_masks, _, _ = train_test_split(attention_mask, label, test_size=0.2, random_state=2025)

batch_size = 32
train_inputs = torch.tensor(train)
train_labels = torch.tensor(train_y)
train_masks = torch.tensor(train_masks)
train_data = TensorDataset(train_inputs, train_masks, train_labels)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

validation_inputs = torch.tensor(validation)
validation_labels = torch.tensor(validation_y)
validation_masks = torch.tensor(validation_masks)
validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
validation_sampler = SequentialSampler(validation_data)
validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

model = ElectraForSequenceClassification.from_pretrained('koelectra-small-v3-discriminator', num_labels=2)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-04, eps=1e-06, betas=(0.9, 0.999))

# optimizer = torch.optim.Adam(model.parameters(), lr=3e-04, eps=1e-06, betas=(0.9, 0.999))

epoch = 4
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader)*epoch)

epoch_results = []

for e in range(0, epoch):
    model.train()
    total_train_loss = 0

    progress_bar = tqdm(train_dataloader, desc=f"training epoch {e+1}", leave=False)

    for batch in progress_bar:
        batch_ids, batch_mask, batch_label = tuple(t.to(device)for t in batch)
        model.zero_grad()

        outputs = model(batch_ids, attention_mask=batch_mask, labels= batch_label)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_train_loss = total_train_loss / len(train_dataloader)

    model.eval()
    train_preds = []
    train_true = []
    for batch in tqdm(train_dataloader, desc=f"evaluating train epoch {e+1}", leave=False):
        batch_ids, batch_mask, batch_label = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        train_preds.extend(preds.cpu().numpy())
        train_true.extend(batch_label.cpu().numpy())

    train_accuracy = np.sum(np.array(train_preds) == np.array(train_true)) / len(train_preds)

    val_preds = []
    val_true = []
    for batch in tqdm(validation_dataloader, desc=f"evaluating train epoch {e + 1}", leave=False):
        batch_ids, batch_mask, batch_label = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            outputs = model(batch_ids, attention_mask=batch_mask)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)
        val_preds.extend(preds.cpu().numpy())
        val_true.extend(batch_label.cpu().numpy())

    val_accuracy = np.sum(np.array(val_preds) == np.array(val_true)) / len(val_preds)

    epoch_results.append((avg_train_loss, train_accuracy, val_accuracy))

print("학습 요약")
for idx, (loss, train_acc, val_acc) in enumerate(epoch_results, start=1):
    print(f"epoch:{idx}, train loss:{loss:.4f}, train accuracy:{train_acc:.4f}, validation accuracy:{val_acc:.4f}")

print("모델 저장")
save_path = "koelectra_small_movie"
model.cpu()
for param in model.parameters():
   if not param.is_contiguous():
        param.data = param.data.contiguous()
model.save_pretrained(save_path, '.pt')
