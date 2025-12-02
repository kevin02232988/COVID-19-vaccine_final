import pandas as pd
import numpy as np
import re
import os

# --- 파일 경로 설정 ---
INPUT_FILE = "FINAL_data_1.csv"
CLEANED_FILE = "FINAL_data_CLEANED_CHECK_2.csv" # 확인용 파일명 변경

# --- 1. 데이터 로드 ---
try:
    df_master = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    df_master = df_master.dropna(subset=['text']) # 텍스트가 없는 행 제거
    df_master['text'] = df_master['text'].astype(str)
    initial_count = len(df_master)
    print(f"[INFO] 초기 데이터 총 {initial_count}건 로드 완료.")
except FileNotFoundError:
    print(f"[ERROR] '{INPUT_FILE}' 파일을 찾을 수 없습니다.")
    exit()

# --- 2. 한국어 데이터 필터링 ---
def is_korean_heavy(text):
    """텍스트 내 한국어 문자 비율이 10%를 초과하면 True 반환 (Naver 데이터 제거 목적)"""
    korean_chars = re.findall(r'[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]', text)
    if len(text) == 0:
        return False
    # 한국어 비율 10% 초과 시 제거
    return len(korean_chars) / len(text) > 0.20

df_master = df_master[~df_master['text'].apply(is_korean_heavy)].copy()

# --- 3. 특수문자 과다 필터링 ---
def is_special_char_heavy(text):
    """알파벳, 숫자, 공백 외 문자 비율이 40%를 초과하면 True 반환 (노이즈 제거)"""
    # 알파벳, 숫자, 공백만 남기고 제거한 문자열 길이
    alphanum_space_len = len(re.sub(r'[^a-zA-Z0-9\s]', '', text))
    total_len = len(text)
    if total_len == 0:
        return True # 빈 텍스트 제거
    # 비-알파벳/숫자/공백 문자 비율이 40%를 초과하면 제거
    return (total_len - alphanum_space_len) / total_len > 0.40

df_master = df_master[~df_master['text'].apply(is_special_char_heavy)].copy()

final_count = len(df_master)
print(f"[INFO] 필터링 후 최종 데이터 총 {final_count}건 남음.")
print(f"[INFO] 제거된 행: {initial_count - final_count}건.")

# --- 4. 최종 데이터 저장 (확인용) ---
df_master.to_csv(CLEANED_FILE, index=False, encoding="utf-8-sig")
print(f"[INFO] 최종 정제 데이터 '{CLEANED_FILE}' 저장 완료. 필터링 결과 확인 후 다음 단계를 진행합니다.")