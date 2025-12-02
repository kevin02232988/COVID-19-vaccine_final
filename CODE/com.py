import pandas as pd
import glob
import os
import hashlib
import numpy as np

# ------------------- 설정 및 파일 목록 -------------------
# 업로드해주신 모든 CSV 파일의 목록입니다.
file_list = [
    "backup_covid_vaccine_reviews.csv",
    "covid_vaccine_comment.csv",
    "covid_vaccine_comments_negative.csv",
    "covid_vaccine_reviews.csv",
    "covid_vaccine_reviews_2.csv",
    "covid_vaccine_reviews_3.csv",
    "covid_vaccine_reviews_fixed.csv",
    "drugs_com_reviews_cache_final.csv",
    "FINAL_COMBI.csv",
    "FINAL_INTEGRATED_REAL_DATA.csv",
    "FINAL_INTEGRATED_VACCINE_DATA.csv",
    "hb_covid.csv",
    "healthboards_comments.csv",
    "healthboards_covid_vaccine.csv",
    "merged_covid_vaccine_reviews.csv",
    "naver_vaccine_comments.csv",
    "naver_vaccine_urls_test.csv",
    "Reddit_comments_final_bulk.csv",
    "Reddit_COVID.csv",
    "Reddit_COVID_Filtered_2.csv",
    "reddit_covid_vaccine_combined.csv",
    "reddit_covid_vaccine_combined_ver2.csv",
    "reddit_covid_vaccine_posts.csv",
    "reddit_covid_vaccine_pushshift.csv",
    "reddit_final_controversy_posts_new.csv",
    "reddit_final_controversy_posts_segmented.csv",
    "reddit_final_controversy_posts_review.csv",
    "reddit_final_controversy_posts_raw.csv",
    "reddit_final_15k_target.csv",
    "webmd_covid_posts_cache_final.csv"
]
OUTPUT_FILE = "FINAL_data_1.csv"

# 최종 분석에 필요한 마스터 컬럼 정의 (데이터를 이 컬럼에 맞춥니다)
master_columns = ['id', 'text', 'source', 'created_at']

# ------------------- 데이터 통합 및 정제 -------------------
df_master = pd.DataFrame()

print("--- 모든 CSV 파일 통합 및 정제 시작 ---")

for file in file_list:
    try:
        # 파일 로드 및 오류 처리
        # low_memory=False로 설정하여 대용량 파일을 처리하고, on_bad_lines='skip'으로 깨진 행을 건너뜁니다.
        df = pd.read_csv(file, on_bad_lines='skip', low_memory=False)
        df = df.fillna('')  # NaN 값을 빈 문자열로 처리

        # 1. 컬럼 매핑 및 통일 (감정 분석에 필요한 'text'와 'id', 'created_at'에 집중)

        # Reddit 댓글 데이터 (body, comment_id)
        if 'body' in df.columns and 'comment_id' in df.columns:
            df_temp = pd.DataFrame({
                'id': df['comment_id'].astype(str),
                'text': df['body'],
                'source': df.get('subreddit', 'Reddit_Comment'),
                'created_at': df.get('date', df.get('created_at', np.nan)),  # date 컬럼 우선 사용
            })

        # Naver 댓글 데이터 (comment, time)
        elif 'comment' in df.columns and 'url' in df.columns:
            df_temp = pd.DataFrame({
                'id': df['url'].astype(str) + df['comment'].astype(str).str[:50],  # ID 생성
                'text': df['comment'],
                'source': 'Naver_Comment',
                'created_at': df.get('time', np.nan),
            })

        # Reddit 게시글 데이터 (selftext, id)
        elif 'selftext' in df.columns and 'id' in df.columns:
            df_temp = pd.DataFrame({
                'id': df['id'],
                'text': df['selftext'],
                'source': df.get('topic_type', df.get('subreddit', 'Reddit_Post')),
                'created_at': df['created_at'],
            })

        # DC Inside 데이터 (content, title)
        elif 'content' in df.columns and 'title' in df.columns:
            df_temp = pd.DataFrame({
                'id': df.get('url', 'DC_Post') + df['title'].astype(str).str[:50],  # ID 생성
                'text': df['content'],
                'source': 'DC_Inside',
                'created_at': np.nan,
            })

        else:
            # URL 목록만 있는 파일 등 분석에 불필요한 파일 건너뛰기
            print(f"[SKIP] 파일 형식 불일치 또는 불필요: {file}")
            continue

        # 2. 텍스트 정제 및 유효성 검사
        df_temp = df_temp[df_temp['text'].str.len() > 3]  # 20자 미만 텍스트 제거 (노이즈 필터링)

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
print(f"총 {final_count}건의 데이터를 확보했습니다. 이제 분석을 시작합니다.")