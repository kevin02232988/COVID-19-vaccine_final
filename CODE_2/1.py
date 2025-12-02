import pandas as pd
import numpy as np
import re
import os

# --- 파일 경로 설정 ---
INPUT_FILE = "FINAL_data_CLEANED_CHECK_2"
CLEANED_FILE = "FINAL_data_CLEANEDver2.csv"

# --- 1. 데이터 로드 ---
try:
    df_master = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    df_master = df_master.dropna(subset=['text'])  # 텍스트가 NaN인 행 제거
    df_master['text'] = df_master['text'].astype(str)
    initial_count = len(df_master)
    print(f"[INFO] 초기 데이터 총 {initial_count}건 로드 완료.")
except FileNotFoundError:
    print(f"[ERROR] '{INPUT_FILE}' 파일을 찾을 수 없습니다.")
    exit()


# --- 2. 빈 내용/플레이스홀더 제거 (핵심 수정) ---
def is_invalid_text(text):
    """텍스트가 유효하지 않은 내용(No Content, 매우 짧음)인지 확인"""
    text_cleaned = text.strip().lower()

    # 1) 텍스트 길이가 10자 미만인 경우
    if len(text_cleaned.split()) < 5:  # 단어 5개 미만인 경우
        return True

    # 2) [No Content] 또는 [deleted]와 같은 플레이스홀더인 경우
    if text_cleaned in ['[no content]', '[deleted]', '[removed]', '[image]', '[video]']:
        return True

    return False


# 유효하지 않은 텍스트를 가진 행 제거
df_master = df_master[~df_master['text'].apply(is_invalid_text)].copy()


# --- 3. 한국어 데이터 필터링 (기존 로직 유지) ---
def is_korean_heavy(text):
    korean_chars = re.findall(r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]', text)
    if len(text) == 0: return False
    return len(korean_chars) / len(text) > 0.10


df_master = df_master[~df_master['text'].apply(is_korean_heavy)].copy()


# --- 4. 특수문자 과다 필터링 (기존 로직 유지) ---
def is_special_char_heavy(text):
    alphanum_space_len = len(re.sub(r'[^a-zA-Z0-9\s]', '', text))
    total_len = len(text)
    if total_len == 0: return True
    return (total_len - alphanum_space_len) / total_len > 0.40


df_master = df_master[~df_master['text'].apply(is_special_char_heavy)].copy()

final_count = len(df_master)
print(f"[INFO] 필터링 후 최종 데이터 총 {final_count}건 남음 (제거된 행: {initial_count - final_count}건).")

# --- 5. 최종 데이터 저장 ---
df_master.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")
print(f"[INFO] 최종 정제 데이터 '{CLEANED_FILE}' 저장 완료.")

# --- 6. AI 라벨링 샘플 추출 ---
# 라벨링 샘플 개수 설정 (최대 1,500건)
target_sample_size = min(1500, final_count)
if target_sample_size < 1000 and final_count >= 1000:
    target_sample_size = 1000
elif target_sample_size == 0 and final_count > 0:
    target_sample_size = final_count

# 샘플 추출
if target_sample_size > 0:
    df_sample = df_master.sample(n=target_sample_size, random_state=42).copy()
    df_sample['sentiment_label'] = np.nan
    LABELING_FILE = "FINAL_AI_labeling_1500.csv"
    df_sample.to_csv(LABELING_FILE, index=False, encoding="utf-8-sig")

    print(f"\n[NEXT STEP] AI 라벨링 샘플 추출 완료: '{LABELING_FILE}' ({len(df_sample)}건)")
    print("           이제 이 파일을 다운로드하여 AI 라벨링을 요청해주세요.")
else:
    print("\n[WARNING] 유효 데이터가 부족하여 라벨링 샘플 추출을 건너뛰고, 프로젝트를 완료할 수 없습니다.")