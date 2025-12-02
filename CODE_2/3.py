import praw
import pandas as pd
import datetime
import time
import os

# --- 1. Reddit API 인증 ---
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "covid_vaccine_analyzer_v1 by /u/YOUR_USERNAME"

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
SUBREDDITS = ["CovidVaccinated", "Coronavirus", "vaccine"]  # 여러 서브레딧 조합
TARGET_TOTAL = 20000
SAVE_INTERVAL = 1000  # 중간 저장 단위
output_file = "reddit_covid_vaccine_combined.csv"

# --- 3. 기존 데이터 불러오기 ---
if os.path.exists(output_file):
    df_combined = pd.read_csv(output_file)
    print(f"[INFO] 기존 데이터 {len(df_combined)}건 불러옴.")
else:
    df_combined = pd.DataFrame()
    print("[INFO] 기존 파일 없음, 새로 수집 시작.")

counter = 0  # 총 수집 개수

# --- 4. 수집 루프 ---
for sub_name in SUBREDDITS:
    print(f"\n--- r/{sub_name} 수집 시작 ---")
    try:
        for submission in reddit.subreddit(sub_name).new(limit=None):  # 최신 글부터
            try:
                timestamp = datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')
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
                df_combined = pd.concat([df_combined, pd.DataFrame([data])], ignore_index=True)
                counter += 1

                # 진행 상황
                if counter % 500 == 0:
                    print(f"[INFO] 현재 {counter}건 수집 완료...")

                # 중간 저장
                if counter % SAVE_INTERVAL == 0:
                    df_combined.drop_duplicates(subset=['id'], inplace=True)
                    df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f"[INFO] 중간 저장 완료 ({len(df_combined)}건)")

                # 목표 도달
                if counter >= TARGET_TOTAL:
                    raise StopIteration

            except Exception as e:
                print(f"[WARN] 게시글 처리 실패: {e}, 넘어감")
                continue
    except StopIteration:
        print(f"[INFO] 목표 {TARGET_TOTAL}건 달성, 수집 종료.")
        break

# --- 5. 최종 저장 ---
df_combined.drop_duplicates(subset=['id'], inplace=True)
df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[INFO] 수집 완료. 총 {len(df_combined)}건 저장됨.")
