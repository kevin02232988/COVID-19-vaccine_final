import pandas as pd
import numpy as np
import re
import os

# --- 파일 경로 설정 ---
INPUT_FILE = "Real_Final.csv"
OUTPUT_FILE = "FINAL_DATA_CLEANED_CLASSIFIED_V2.csv"  # 최종 분류 데이터 저장 파일

# ------------------- 1. 데이터 로드 -------------------
try:
    df_master = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    df_master = df_master.dropna(subset=['text'])
    df_master['text'] = df_master['text'].astype(str)
    initial_count = len(df_master)
    print(f"[INFO] 초기 데이터 총 {initial_count}건 로드 완료.")
except FileNotFoundError:
    print(f"[ERROR] '{INPUT_FILE}' 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    exit()

# ------------------- 2. 주제 관련성 정의 (핵심 키워드 확장) -------------------
# 사용자님의 요청에 따라 키워드를 대폭 확장했습니다.
CORE_RELATED_KEYWORDS = [
    'vaccine', 'covid', 'coronavirus', 'side effect', 'adverse', 'pfizer', 'moderna',
    'booster', 'jab', 'shot', 'vax', 'myocarditis', 'astrazeneca', 'janssen',
    'symptoms', 'mandate', 'mask', 'masked', 'unvaccinated', 'vaxxed', 'unvaxxed',
    'hospital', 'death', 'long covid', 'long-covid', 'spike protein', 'mrna'
]


# ------------------- 3. True/False 분류 로직 적용 -------------------
def classify_relevance(text):
    """텍스트가 COVID/Vaccine 주제와 직접적으로 관련 있는지 True/False로 분류"""
    text_lower = text.lower()

    # 확장된 핵심 키워드 중 하나라도 포함되어 있으면 True (관련 있음)
    if any(keyword in text_lower for keyword in CORE_RELATED_KEYWORDS):
        return True

    return False


# 새로운 컬럼 'is_related_topic'에 분류 결과 저장
df_master['is_related_topic'] = df_master['text'].apply(classify_relevance)

# ------------------- 4. 결과 분석 및 저장 -------------------

related_count = df_master['is_related_topic'].sum()
unrelated_count = initial_count - related_count

print("\n--- 주제 관련성 분류 결과 ---")
print(f"✅ 1 (관련 있음) 데이터: {related_count}건")
print(f"❌ 2 (관련 없음) 데이터: {unrelated_count}건")
print(f"   (True 비율: {related_count / initial_count * 100:.2f}%)")

# 최종 CSV 저장
df_master.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 최종 분류 데이터 '{OUTPUT_FILE}' 저장 완료.")

print("\n[NEXT STEP] 이제 분류된 데이터를 기반으로 논란 분석을 진행할 수 있습니다.")