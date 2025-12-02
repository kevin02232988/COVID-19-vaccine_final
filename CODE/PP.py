import pandas as pd
import numpy as np
import re
import os

# --- 파일 경로 설정 (사용자 지정) ---
INPUT_FILE = "FINAL_data_CLEANED_CHECK_2.csv"  # 정제할 원본 데이터 파일명
CLEANED_FILE = "FINAL_DATA_CLEANED_READY.csv"  # 최종 정제 데이터 저장 파일
LABELING_FILE = "FINAL_AI_labeling_10_percent.csv"  # 라벨링 샘플 저장 파일

# ------------------- 1. 데이터 로드 및 초기 설정 -------------------
try:
    df_master = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    df_master = df_master.dropna(subset=['text'])  # 텍스트가 NaN인 행 제거
    df_master['text'] = df_master['text'].astype(str)
    initial_count = len(df_master)
    print(f"[INFO] 초기 데이터 총 {initial_count}건 로드 완료.")
except FileNotFoundError:
    print(f"[ERROR] '{INPUT_FILE}' 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    exit()


# ------------------- 2. 함수 정의: 가비지 필터링 로직 -------------------

def is_invalid_text(text):
    """빈 내용/플레이스홀더 및 짧은 텍스트 필터링"""
    text_cleaned = text.strip().lower()

    # 1) 플레이스홀더 제거
    if text_cleaned in ['[no content]', '[deleted]', '[removed]', '[image]', '[video]', '[link]']:
        return True

    # 2) 텍스트 길이가 5단어 미만인 경우
    if len(text_cleaned.split()) < 5:
        return True

    return False


def is_korean_heavy(text):
    """한국어 비율 10% 초과 필터링"""
    korean_chars = re.findall(r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]', text)
    total_len = len(text)
    if total_len == 0: return False
    return len(korean_chars) / total_len > 0.10


def is_special_char_heavy(text):
    """특수문자 비율 40% 초과 필터링"""
    # 알파벳, 숫자, 공백만 남기고 제거한 문자열 길이
    alphanum_space_len = len(re.sub(r'[^a-zA-Z0-9\s]', '', text))
    total_len = len(text)
    if total_len == 0: return True
    # 비-알파벳/숫자/공백 문자 비율이 40%를 초과하면 제거
    return (total_len - alphanum_space_len) / total_len > 0.40


# ------------------- 3. 필터링 파이프라인 실행 -------------------

df_master = df_master[~df_master['text'].apply(is_invalid_text)].copy()
df_master = df_master[~df_master['text'].apply(is_korean_heavy)].copy()
df_master = df_master[~df_master['text'].apply(is_special_char_heavy)].copy()

final_count = len(df_master)
print(f"[INFO] 필터링 후 최종 데이터 총 {final_count}건 남음 (제거된 총 행 수: {initial_count - final_count}건).")

# ------------------- 4. 최종 데이터 저장 및 샘플 추출 -------------------

# 최종 정제 데이터 저장 (전체 데이터)
df_master.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 최종 정제 데이터 '{CLEANED_FILE}' 저장 완료.")

# 라벨링 샘플 추출 (10% 계산)
target_sample_size = int(final_count * 0.10)
target_sample_size = max(1000, target_sample_size)  # 최소 1000개는 확보하도록 설정
target_sample_size = min(final_count, target_sample_size)

if target_sample_size > 0:
    df_sample = df_master.sample(n=target_sample_size, random_state=42).copy()
    df_sample['sentiment_label'] = np.nan  # AI 라벨링을 위한 빈 컬럼 추가

    # 샘플 파일 저장
    df_sample.to_csv(LABELING_FILE, index=False, encoding="utf-8-sig")

    print(f"\n[NEXT STEP] AI 라벨링 샘플 추출 완료: '{LABELING_FILE}' ({len(df_sample)}건)")
    print("           이 파일을 다운로드하여 AI 라벨링을 요청해 주세요.")
else:
    print("\n[WARNING] 유효 데이터가 부족하여 프로젝트 진행이 어렵습니다.")