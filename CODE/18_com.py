import pandas as pd
import glob
import os
import hashlib

# ------------------- 설정 -------------------
OUTPUT_FILE = "FINAL_combi.csv"
file_pattern = "*.csv"

# 최종 분석에 필요한 마스터 컬럼 정의
master_columns = ['id', 'text', 'source', 'created_at']

# ------------------- 데이터 통합 및 정제 -------------------
df_master = pd.DataFrame()
seen_ids = set()

# 폴더 내의 모든 CSV 파일을 순회
all_files = glob.glob(file_pattern)
print(f"--- 총 {len(all_files)}개의 CSV 파일 통합 시작 ---")

for file in all_files:
    if file == OUTPUT_FILE:
        continue

    try:
        # 파일 로드 및 오류 처리
        df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        df = df.fillna('')  # NaN 값을 빈 문자열로 처리

        # 1. 컬럼 매핑 및 통일 (감정 분석에 필요한 'text'와 'id', 'created_at'에 집중)
        if 'body' in df.columns:
            # Reddit 댓글 데이터 (body와 comment_id, date)
            df_temp = pd.DataFrame({
                'id': df['comment_id'].astype(str),
                'text': df['body'],
                'source': df['subreddit'],  # subreddit을 source로 사용 (더 구체적임)
                'created_at': df['date'],
            })

        elif 'comment' in df.columns and 'url' in df.columns:
            # Naver 댓글 데이터 (ID가 없으므로 URL + 댓글 일부로 생성)
            df_temp = pd.DataFrame({
                'id': df['url'].astype(str) + df['comment'].astype(str).str[:50],
                'text': df['comment'],
                'source': 'Naver_Comment',
                'created_at': df.get('time', pd.NA),
            })

        elif 'selftext' in df.columns and 'id' in df.columns:
            # Reddit 게시글 데이터 (selftext와 id)
            df_temp = pd.DataFrame({
                'id': df['id'],
                'text': df['selftext'],
                'source': df.get('topic_type', 'Reddit_Post'),
                'created_at': df['created_at'],
            })

        elif 'text' in df.columns:
            # 기타 정제된 텍스트 데이터 (예: HealthBoards)
            df_temp = pd.DataFrame({
                'id': df['url'].astype(str) + df['text'].astype(str).str[:50],  # URL + 텍스트 일부로 ID 생성
                'text': df['text'],
                'source': df.get('source', 'Unknown_Forum'),
                'created_at': df.get('date', pd.NA),
            })

        else:
            print(f"[SKIP] 파일 형식 불일치 또는 불필요: {file}")
            continue

        # 2. 텍스트 정제 및 유효성 검사
        df_temp = df_temp[df_temp['text'].str.len() > 20]  # 20자 미만 텍스트 제거 (노이즈 필터링)

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
df_master.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ 데이터 통합 성공. 최종 파일 저장 완료: '{OUTPUT_FILE}'")
print("\n[NEXT STEP] 최종 확보된 데이터를 기반으로 라벨링 및 모델 학습을 진행할 수 있습니다.")