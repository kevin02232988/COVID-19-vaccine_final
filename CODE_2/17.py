import pandas as pd
import glob
import os
import hashlib

# ------------------- 설정 및 파일 목록 -------------------
# 업로드해주신 모든 CSV 파일을 포함하는 패턴
file_pattern = "*.csv"
output_file = "FINAL_INTEGRATED_REAL_DATA.csv"

# 최종 컬럼 정의 (모든 소스 데이터를 통합할 기준 컬럼)
master_columns = ['id', 'title', 'text', 'source', 'created_at']

# ------------------- 데이터 통합 및 정제 -------------------
df_master = pd.DataFrame()
seen_hashes = set()

# 폴더 내의 모든 CSV 파일을 순회
all_files = glob.glob(file_pattern)

print(f"--- 총 {len(all_files)}개의 CSV 파일을 통합합니다 ---")

for file in all_files:
    if file == output_file:
        continue

    try:
        df = pd.read_csv(file, on_bad_lines='skip')
        df = df.fillna('')  # NaN 값을 빈 문자열로 처리하여 오류 방지

        # 1. 컬럼 매핑 및 정규화
        if 'comment' in df.columns:
            # Naver 댓글 데이터
            df_temp = pd.DataFrame({
                'id': df['url'].astype(str) + df['comment'].astype(str).str[:50],  # URL + 댓글 50자로 ID 생성
                'title': df.get('title', 'Naver Comment'),
                'text': df['comment'],
                'source': 'Naver',
                'created_at': df.get('time', pd.NA),
            })

        elif 'selftext' in df.columns and 'id' in df.columns:
            # Reddit 게시글 데이터
            df_temp = pd.DataFrame({
                'id': df['id'],
                'title': df['title'],
                'text': df['selftext'],
                'source': 'Reddit',
                'created_at': df['created_at']
            })

        elif 'content' in df.columns and 'title' in df.columns:
            # DC Inside 데이터
            df_temp = pd.DataFrame({
                'id': df.get('url', 'DC_Post') + df['title'].astype(str).str[:50],
                'title': df['title'],
                'text': df['content'],
                'source': 'DC_Inside',
                'created_at': pd.NA,
            })

        elif 'url' in df.columns and len(df.columns) == 1:
            # URL 목록만 있는 파일 (예: naver_vaccine_urls_test.csv)
            print(f"[SKIP] {file}: URL 목록만 있는 파일은 건너뜁니다.")
            continue

        else:
            print(f"[SKIP] {file}: 알 수 없는 형식의 파일입니다.")
            continue

        # 2. 텍스트가 없는 행 제거
        df_temp = df_temp[df_temp['text'].str.len() > 10]

        # 3. 마스터 데이터에 합치기
        df_master = pd.concat([df_master, df_temp[master_columns]], ignore_index=True)
        print(f"[LOAD] {file} 통합 완료. 현재 총 {len(df_master)}건.")

    except Exception as e:
        print(f"[ERROR] 파일 처리 중 오류 발생 ({file}): {e}")
        continue

# 4. 최종 정제 및 중복 제거
initial_count = len(df_master)
# ID를 기준으로 중복 제거
df_master.drop_duplicates(subset=['id'], keep='first', inplace=True)
final_count = len(df_master)

print("\n--- 데이터 통합 및 정제 완료 ---")
print(f"최초 통합 데이터 수: {initial_count}")
print(f"최종 유효 데이터 수: {final_count}건 (중복 및 노이즈 제거)")

# 5. 최종 CSV 파일 저장
df_master.to_csv(output_file, index=False, encoding="utf-8-sig")
print(f"\n✅ 데이터 통합 성공. 최종 파일 저장 완료: '{output_file}'")

# 6. 최종 데이터 개수 확인 및 분석 시작
if final_count > 0:
    print(f"\n[NEXT STEP] 총 {final_count}건의 고품질 데이터를 확보했습니다.")
    print("이 데이터를 기반으로 분석을 시작하겠습니다.")
else:
    print("⚠️ 유효 데이터가 0건입니다. 프로젝트 진행을 위해 공개 데이터셋 활용을 권장합니다.")