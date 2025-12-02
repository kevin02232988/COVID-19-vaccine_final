import pandas as pd
import os

# --- 파일 경로 설정 ---
INPUT_FILE = "FINAL_DATA_CLEANED_CLASSIFIED_V2.csv"  # 이전 단계에서 분류된 파일
OUTPUT_FILE = "FINAL_DATA_FILTERED_TRUE.csv"  # True만 남긴 최종 필터링 파일

# ------------------- 1. 데이터 로드 -------------------
try:
    df_master = pd.read_csv(INPUT_FILE, on_bad_lines='skip')
    initial_count = len(df_master)

    if 'is_related_topic' not in df_master.columns:
        print("[ERROR] 'is_related_topic' 컬럼을 찾을 수 없습니다. 분류 코드를 먼저 실행해 주세요.")
        exit()

    print(f"[INFO] 초기 데이터 총 {initial_count}건 로드 완료.")
except FileNotFoundError:
    print(f"[ERROR] '{INPUT_FILE}' 파일을 찾을 수 없습니다. 파일명을 확인해 주세요.")
    exit()

# ------------------- 2. True 값만 필터링 (False 제거) -------------------

# is_related_topic 값이 True인 행만 남깁니다.
df_filtered = df_master[df_master['is_related_topic'] == True].copy()

final_count = len(df_filtered)

print("\n--- 필터링 결과 ---")
print(f"✅ True (관련 있음) 데이터만 남김: {final_count}건")
print(f"❌ False (무관함) 데이터 제거됨: {initial_count - final_count}건")

# ------------------- 3. 최종 데이터 저장 -------------------
df_filtered.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✅ 최종 필터링 데이터 '{OUTPUT_FILE}' 저장 완료.")

# ------------------- 4. 다음 단계 안내: AI 라벨링 샘플 추출 -------------------

# AI 라벨링 샘플 추출 (10% 계산)
target_sample_size = int(final_count * 0.10)
# 최소 1000개는 확보 (만약 데이터가 1000개 미만이면 전체 사용)
target_sample_size = max(1000, target_sample_size)
target_sample_size = min(final_count, target_sample_size)

if target_sample_size > 0:
    df_sample = df_filtered.sample(n=target_sample_size, random_state=42).copy()
    df_sample['sentiment_label'] = np.nan
    LABELING_FILE = "FINAL_AI_labeling_TRUE_SAMPLE.csv"
    df_sample.to_csv(LABELING_FILE, index=False, encoding="utf-8-sig")

    print(f"\n[NEXT STEP] AI 라벨링 샘플 추출 완료: '{LABELING_FILE}' ({len(df_sample)}건)")
    print("           이 파일을 다운로드하여 AI 라벨링을 요청해 주세요.")
else:
    print("\n[WARNING] 필터링 후 유효 데이터가 부족하여 프로젝트 진행이 어렵습니다.")