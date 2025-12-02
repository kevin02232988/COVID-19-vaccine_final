import praw
import pandas as pd
import datetime as dt
import time
import os
import csv
from prawcore.exceptions import ResponseException

# ------------------- 설정 및 인증 정보 -------------------
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "Final_Controversy_Collector_v3 by /u/Delicious_Tough_2446"

# 수집 대상 서브레딧
TARGET_SUBREDDITS = ["Coronavirus", "vaccine", "COVID19", "CovidVaccinated", "news"]
LIMIT_PER_SUB = 10000  # 서브레딧 당 최대 1만 건 요청
OUTPUT_FILE = "reddit_final_15k_target.csv"

# ------------------- PRAW 클라이언트 생성 -------------------
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

# ------------------- 유틸리티 함수 -------------------
def utc_to_datestr(utc_ts):
    try:
        # 날짜 변환 함수
        return dt.datetime.fromtimestamp(utc_ts).strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ""


# ------------------- 데이터 수집 루프 -------------------
df_combined = pd.DataFrame()
existing_ids = set()

# 기존 파일 로드하여 중복 방지 (체크포인트)
if os.path.exists(OUTPUT_FILE):
    try:
        df_existing = pd.read_csv(OUTPUT_FILE)
        existing_ids = set(df_existing['id'].astype(str))
        df_combined = df_existing
        print(f"[INFO] 기존 파일에서 {len(existing_ids)}건의 데이터를 불러왔습니다.")
    except Exception:
        pass

total_collected = len(existing_ids)
result = []

print(f"\n--- Reddit 데이터 최종 수집 시작 (15,000건 목표) ---")

for sub_name in TARGET_SUBREDDITS:
    print(f"\n[SUBREDDIT] r/{sub_name} 데이터 수집 시작...")

    try:
        # .top() 메소드를 사용하여 가장 논란이 많았던 인기 게시글부터 수집
        submissions = reddit.subreddit(sub_name).top(time_filter="all", limit=LIMIT_PER_SUB)

        count_this_sub = 0
        for submission in submissions:
            if submission.id in existing_ids:
                continue

            # 'news' 서브레딧에서는 코로나 관련 키워드를 반드시 확인 (노이즈 방지)
            if sub_name == 'news' or sub_name == 'worldnews':
                title_lower = submission.title.lower()
                if 'vaccine' not in title_lower and 'covid' not in title_lower:
                    continue

            timestamp = utc_to_datestr(submission.created_utc)

            result.append({
                'id': submission.id,
                'title': submission.title,
                'score': submission.score,
                'num_comments': submission.num_comments,
                'created_at': timestamp,
                'url': submission.url,
                'selftext': submission.selftext if submission.selftext else '[No Content]',
                'topic_type': sub_name
            })
            existing_ids.add(submission.id)
            count_this_sub += 1
            total_collected += 1

        print(f"[INFO] r/{sub_name} 수집 완료: {count_this_sub}건 추가. 누적: {total_collected}건.")

    except Exception as e:
        print(f"[ERROR] r/{sub_name} 데이터 수집 중 오류 발생: {e}. 이 서브레딧은 건너뜁니다.")

    time.sleep(3)  # 서브레딧 변경 시 3초 대기

# ------------------- 최종 저장 -------------------
df_new = pd.DataFrame(result)
df_combined = pd.concat([df_combined, df_new]).drop_duplicates(subset=['id']).reset_index(drop=True)

df_combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 최종 완료. 총 {len(df_combined)}건의 게시글을 수집했습니다.")
print(f"파일 저장 완료: '{OUTPUT_FILE}'")