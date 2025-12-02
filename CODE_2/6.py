import praw
import pandas as pd
import datetime
import os
import time

# --- 1. Reddit API 인증 ---
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "covid_vaccine_analyzer_stable by /u/YOUR_USERNAME"

try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent=USER_AGENT
    )
    reddit.read_only = True
    print("[INFO] Reddit API 인증 성공.")
except Exception as e:
    print(f"[ERROR] Reddit API 인증 실패: {e}")
    exit()

# --- 2. 크롤링 설정 ---
SUBREDDITS = ["CovidVaccinated", "Coronavirus", "vaccine", "COVID19", "COVID19Vaccine"]
TARGET_TOTAL = 20000
SAVE_INTERVAL = 5000  # 중간 저장 단위
MAX_RETRIES = 3       # 오류 재시도 횟수
output_file = "reddit_covid_vaccine_combined_ver2.csv"

# --- 3. 기존 데이터 불러오기 ---
if os.path.exists(output_file):
    df_combined = pd.read_csv(output_file)
    existing_ids = set(df_combined['id'].tolist())
    print(f"[INFO] 기존 데이터 {len(df_combined)}건 불러옴.")
else:
    df_combined = pd.DataFrame()
    existing_ids = set()
    print("[INFO] 기존 파일 없음, 새로 수집 시작.")

counter = len(existing_ids)

# --- 4. 날짜 범위 생성 (1개월 단위) ---
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2023, 12, 31)
delta = datetime.timedelta(days=30)
date_ranges = []
current_start = start_date
while current_start < end_date:
    current_end = current_start + delta
    date_ranges.append((int(current_start.timestamp()), int(current_end.timestamp())))
    current_start = current_end

# --- 5. 수집 루프 ---
for sub_name in SUBREDDITS:
    print(f"\n--- r/{sub_name} 수집 시작 ---")
    for after_ts, before_ts in date_ranges:
        posts_buffer = []
        retries = 0
        while retries < MAX_RETRIES:
            try:
                for submission in reddit.subreddit(sub_name).new(limit=None):
                    post_time = int(submission.created_utc)
                    if post_time < after_ts or post_time >= before_ts:
                        continue
                    if submission.id in existing_ids:
                        continue

                    timestamp = datetime.datetime.fromtimestamp(post_time).strftime('%Y-%m-%d %H:%M:%S')
                    data = {
                        'id': submission.id,
                        'title': submission.title,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_at': timestamp,
                        'url': submission.url,
                        'selftext': submission.selftext if submission.selftext else '[No Content]',
                        'topic_type': 'Vaccine'
                    }
                    posts_buffer.append(data)
                    existing_ids.add(submission.id)
                    counter += 1

                    # 진행 표시
                    if counter % 500 == 0:
                        print(f"[INFO] 현재 {counter}건 수집 완료...")

                    # 목표 도달
                    if counter >= TARGET_TOTAL:
                        raise StopIteration

                break  # 정상 완료 시 반복 종료

            except Exception as e:
                retries += 1
                wait_time = retries * 5
                print(f"[WARN] 오류 발생: {e}. {wait_time}s 후 재시도 {retries}/{MAX_RETRIES}")
                time.sleep(wait_time)
                continue

        # 기간별 중간 저장
        if posts_buffer:
            df_new = pd.DataFrame(posts_buffer)
            df_combined = pd.concat([df_combined, df_new], ignore_index=True)
            df_combined.drop_duplicates(subset=['id'], inplace=True)
            df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"[INFO] 기간 {datetime.datetime.fromtimestamp(after_ts).strftime('%Y-%m')} ~ "
                  f"{datetime.datetime.fromtimestamp(before_ts).strftime('%Y-%m')} 중간 저장 완료 ({len(df_combined)}건)")

# --- 6. 최종 저장 ---
df_combined.drop_duplicates(subset=['id'], inplace=True)
df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[INFO] 수집 완료. 총 {len(df_combined)}건 저장됨.")
