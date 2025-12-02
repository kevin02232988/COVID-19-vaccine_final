import pandas as pd
import numpy as np
import os

# --- 파일 경로 설정 ---
INPUT_FILE = "Real_Final.csv" # 마지막으로 정제된 파일
LABELING_FILE = "FINAL_AI_labeling_2.csv"

# --- 1. 데이터 로드 ---
try:
    df_master = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    df_master = df_master.dropna(subset=['text'])
    final_count = len(df_master)
    print(f"[INFO] 최종 데이터 총 {final_count}건 로드 완료.")
except FileNotFoundError:
    print(f"[ERROR] '{INPUT_FILE}' 파일을 찾을 수 없습니다. 정제 단계를 다시 확인해 주세요.")
    exit()

# --- 2. 샘플 개수 설정 (10% 계산) ---
# 데이터 총 개수 (110,273건)의 10%를 계산하여 샘플 크기로 설정합니다.
target_sample_size = int(final_count * 0.10)
target_sample_size = max(1, target_sample_size) # 최소 1개는 추출하도록 설정

print(f"[INFO] 라벨링 샘플 개수 (10%): {target_sample_size}건.")

# --- 3. 샘플 추출 ---
df_sample = df_master.sample(n=target_sample_size, random_state=42).copy()

# --- 4. 수동 라벨링 컬럼 추가 (AI 라벨링 준비) ---
df_sample['sentiment_label'] = np.nan

# --- 5. 라벨링 파일로 저장 ---
df_sample.to_csv(LABELING_FILE, index=False, encoding="utf-8-sig")

print(f"\n[NEXT STEP] AI 라벨링 샘플 추출 완료: '{LABELING_FILE}' ({len(df_sample)}건)")
print("           이 파일을 다운로드하여 AI 라벨링을 요청해 주세요.")