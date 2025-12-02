import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import argparse


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict(texts, tokenizer, model):
    inputs = tokenizer(
        texts,
        max_length=256,
        truncation=True,
        padding=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    # preds: 0 = negative, 1 = positive
    return preds.numpy(), probs[:, 1].numpy()


def main(csv_path, text_col, output_path):
    print("📌 Loading English sentiment model...")
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer, model = load_model(model_name)

    print(f"📌 Loading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Auto-detect text column if not provided
    if text_col is None:
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                print(f"➡ Automatically detected text column: {text_col}")
                break

    texts = df[text_col].astype(str).tolist()

    print("📌 Running sentiment prediction (0=negative, 1=positive)...")
    preds, probs = predict(texts, tokenizer, model)

    df["sentiment"] = preds          # 0 or 1
    df["prob_positive"] = probs      # probability of being positive

    df.to_csv(output_path, index=False)
    print(f"✅ Done! Results saved to {output_path}")

    # ----- 계산 요청한 비율 출력 ------
    total = len(preds)
    neg_ratio = (preds == 0).sum() / total * 100
    pos_ratio = (preds == 1).sum() / total * 100

    print("\n===== Sentiment Ratio =====")
    print(f"Negative (0): {neg_ratio:.2f}%")
    print(f"Positive (1): {pos_ratio:.2f}%")
    print("===========================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default="10_per#_final.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--output_path", type=str, default="labeled_output.csv")

    args = parser.parse_args()
    main(args.csv_path, args.text_col, args.output_path)
