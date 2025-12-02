import pandas as pd
import numpy as np
import re
import os

# 1. 파일 로드
FILE_PATH = "FINAL_Final_label.csv"
try:
    df = pd.read_csv(FILE_PATH, on_bad_lines='skip')
    print(f"[INFO] 파일 로드 완료. 총 {len(df)}건.")

    if 'text' not in df.columns:
        print("[ERROR] 'text' 컬럼을 찾을 수 없습니다. 라벨링을 진행할 수 없습니다.")
        exit()

    df['text'] = df['text'].fillna('').astype(str)

except Exception as e:
    print(f"[ERROR] 파일 로드 중 오류 발생: {e}")
    exit()


# 2. AI 라벨링 기준 정의 (Sentiment Analysis Simulation)
# 1: Negative (부정), 2: Neutral (중립), 3: Positive (긍정)
def generate_ai_label(text):
    """키워드 기반으로 감정을 분류하는 AI 라벨링 시뮬레이션"""
    text_lower = text.lower()

    # 부정적 키워드 (부작용, 불안, 비난)
    negative_kws = ['side effect', 'adverse', 'scary', 'afraid', 'died', 'regret', 'unnecessary', 'mandate', 'forced',
                    'problem', 'harmful', 'pain', 'worst']
    # 긍정적 키워드 (효과, 만족, 독려)
    positive_kws = ['effective', 'thankful', 'relief', 'safe', 'great', 'good job', 'recommend', 'glad', 'proud',
                    'finally', 'happy']

    # 카운트
    neg_count = sum(text_lower.count(kw) for kw in negative_kws)
    pos_count = sum(text_lower.count(kw) for kw in positive_kws)

    # 논리 판단
    if neg_count > pos_count and neg_count >= 1:  # 부정 키워드가 더 많고, 최소 1개 이상이면 부정
        return 1
    elif pos_count > neg_count and pos_count >= 1:  # 긍정 키워드가 더 많고, 최소 1개 이상이면 긍정
        return 3
    else:
        return 2  # 나머지, 또는 중립/사실 전달


# 3. 라벨링 적용
# 기존의 sentiment_label 컬럼이 있다면 덮어쓰고, 없다면 새로 생성합니다.
df['sentiment_label_AI_draft'] = df['text'].apply(generate_ai_label)

# 4. 최종 라벨 분포 확인
label_counts = df['sentiment_label_AI_draft'].value_counts(normalize=True).sort_index() * 100
print("\n--- AI 라벨링 결과 분포 ---")
print(f"총 라벨링된 행: {len(df)}건")
print("감정별 비율 (1:부정, 2:중립, 3:긍정):")
print(label_counts.to_markdown(floatfmt=".2f"))

# 5. 최종 CSV 저장
OUTPUT_LABELED_FILE = "FINAL_AI_labeled_9k_draft.csv"
df.to_csv(OUTPUT_LABELED_FILE, index=False, encoding="utf-8-sig")

print(f"\n[NEXT STEP] AI 라벨링 초안 완료. '{OUTPUT_LABELED_FILE}' 파일을 다운로드해 주세요.")
print("           이 파일은 최종 제출 전 사용자님의 검수 및 수정이 필수입니다.")