import praw
import pandas as pd
import datetime
import os
import time

# --- 1. Reddit API 인증 ---
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"
USER_AGENT = "covid_vaccine_analyzer_v3 by /u/YOUR_USERNAME"

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
SAVE_INTERVAL = 5000  # 중간 저장 간격
SUB_LIMIT = 5000      # 서브레딧별 최대 수집
output_file = "reddit_covid_vaccine_combined.csv"

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
all_posts = []

# --- 4. 수집 루프 ---
for sub_name in SUBREDDITS:
    print(f"\n--- r/{sub_name} 수집 시작 (최대 {SUB_LIMIT}건) ---")
    try:
        for submission in reddit.subreddit(sub_name).new(limit=SUB_LIMIT):
            try:
                if submission.id in existing_ids:
                    continue  # 중복 방지

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
                all_posts.append(data)
                existing_ids.add(submission.id)
                counter += 1

                # 진행 상황
                if counter % 500 == 0:
                    print(f"[INFO] 현재 {counter}건 수집 완료...")

                # 중간 저장
                if len(all_posts) >= SAVE_INTERVAL:
                    df_new = pd.DataFrame(all_posts)
                    df_combined = pd.concat([df_combined, df_new], ignore_index=True)
                    df_combined.drop_duplicates(subset=['id'], inplace=True)
                    df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
                    print(f"[INFO] 중간 저장 완료 ({len(df_combined)}건)")
                    all_posts = []  # 리스트 초기화

                # 목표 도달
                if counter >= TARGET_TOTAL:
                    raise StopIteration

            except Exception:
                continue  # 오류 무시하고 넘어감

    except StopIteration:
        print(f"[INFO] 목표 {TARGET_TOTAL}건 달성, 수집 종료.")
        break
    except Exception as e:
        print(f"[WARN] 서브레딧 처리 실패: {e}, 넘어갑니다.")
        continue

# --- 5. 최종 저장 ---
if all_posts:  # 남은 데이터 저장
    df_new = pd.DataFrame(all_posts)
    df_combined = pd.concat([df_combined, df_new], ignore_index=True)

df_combined.drop_duplicates(subset=['id'], inplace=True)
df_combined.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n[INFO] 수집 완료. 총 {len(df_combined)}건 저장됨.")
