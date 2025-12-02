import pandas as pd
import glob
import os

# 1. 통합할 파일 목록 설정 (업로드해주신 모든 CSV 파일)
file_list = [
    "naver_vaccine_urls_test.csv",
    "covid_vaccine_comments_negative.csv",
    "dc_inside_covid_vaccine_posts.csv",
    "naver_vaccine_comments.csv",
    "reddit_vaccine_posts_praw.csv",
    "reddit_covid_vaccine_combined.csv",
    "reddit_covid_vaccine_combined_ver2.csv",
    "reddit_covid_vaccine_posts.csv",
    "reddit_covid_vaccine_pushshift.csv",
    "reddit_final_controversy_posts_new.csv"
]

# 2. 통합 및 정제 (Master DataFrame)
df_master = pd.DataFrame()
master_columns = ['id', 'title', 'text', 'source', 'type', 'created_at']

print("--- 모든 CSV 파일 통합 및 정제 시작 ---")

for file in file_list:
    try:
        df = pd.read_csv(file, encoding='utf-8')
        print(f"[LOAD] 파일 로딩: {file} ({len(df)}건)")

        # 데이터셋 유형에 따라 컬럼 통일 (핵심 로직)
        if 'comment' in df.columns and 'url' in df.columns:
            # Naver 댓글 데이터
            df_temp = pd.DataFrame({
                'id': df['url'].astype(str) + df['comment'].astype(str).str[:30],  # ID가 없으므로 URL+댓글 일부로 생성
                'title': df.get('title', 'Naver Comment'),
                'text': df['comment'],
                'source': 'Naver',
                'type': 'Comment',
                'created_at': df.get('time', pd.NA)  # 시간 정보가 있으면 사용
            })

        elif 'selftext' in df.columns:
            # Reddit 게시글 데이터
            df_temp = pd.DataFrame({
                'id': df['id'],
                'title': df['title'],
                'text': df['selftext'],
                'source': 'Reddit',
                'type': 'Post',
                'created_at': df['created_at']
            })

        elif 'content' in df.columns and 'title' in df.columns:
            # DC Inside 데이터
            df_temp = pd.DataFrame({
                'id': df.get('url', 'DC_Post') + df['title'].astype(str).str[:20],
                'title': df['title'],
                'text': df['content'],
                'source': 'DC_Inside',
                'type': 'Post',
                'created_at': pd.NA
            })

        else:
            print(f"[SKIP] 알 수 없는 형식 또는 불필요한 파일: {file}")
            continue

        # 최종 컬럼만 남기고 통합
        df_master = pd.concat([df_master, df_temp[master_columns]], ignore_index=True)

    except Exception as e:
        print(f"[ERROR] 파일 처리 중 오류 발생 ({file}): {e}")
        continue

# 3. 최종 정제 및 중복 제거
# ID를 기준으로 중복 제거 (가장 중요한 단계)
initial_count = len(df_master)
df_master.dropna(subset=['text'], inplace=True)  # 텍스트가 없는 행 제거
df_master.drop_duplicates(subset=['id'], keep='first', inplace=True)

final_count = len(df_master)

print("\n--- 통합 및 정제 완료 ---")
print(f"초기 통합 데이터 수: {initial_count}")
print(f"최종 유효 데이터 수: {final_count} (중복 및 텍스트 없는 행 {initial_count - final_count}개 제거)")

# 4. 최종 CSV 파일 저장
output_file = "FINAL_INTEGRATED_VACCINE_DATA.csv"
df_master.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n✅ 데이터 통합 성공. 총 {final_count}건 확보.")
print(f"💾 최종 파일 저장 완료: '{output_file}'")