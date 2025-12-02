import praw
import pandas as pd
import datetime as dt
import time

# 1. Reddit API 인증 정보 (반드시 본인 정보로 교체)
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USER_AGENT = "covid_vaccine_crawler by /u/YOUR_USERNAME"

# 2. Reddit API 초기화
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        user_agent=USER_AGENT
    )
    reddit.read_only = True
    print("[INFO] Reddit API 인증 성공.")
except Exception as e:
    print(f"[ERROR] Reddit API 인증 실패: {e}")
    exit()

# 3. 크롤링 설정
TARGET_SUBREDDITS = ["CovidVaccinated", "Coronavirus", "vaccine", "COVID19", "COVID19Vaccine"]
START_YEAR = 2020
END_YEAR = 2024
OUTPUT_FILE = "reddit_vaccine_posts_praw.csv"

# 기존 데이터 불러오기
try:
    df_existing = pd.read_csv(OUTPUT_FILE)
    existing_ids = set(df_existing["id"])
    print(f"[INFO] 기존 파일에서 {len(existing_ids)}건 불러옴.")
except FileNotFoundError:
    df_existing = pd.DataFrame()
    existing_ids = set()
    print("[INFO] 기존 파일이 없어 새로 생성합니다.")

result = []

# 4. 월별 수집 루프
for sub_name in TARGET_SUBREDDITS:
    print(f"\n--- r/{sub_name} 수집 시작 ---")

    for year in range(START_YEAR, END_YEAR + 1):
        start_month = 3 if year == 2020 else 1

        for month in range(start_month, 13):
            start_date = dt.datetime(year, month, 1)
            if month == 12:
                end_date = dt.datetime(year + 1, 1, 1)
            else:
                end_date = dt.datetime(year, month + 1, 1)

            print(f"[INFO] {year}-{month:02d} 기간 데이터 수집 중...")

            query = "(covid OR vaccine)"
            collected_count = 0

            try:
                # search()는 Reddit API 내부에서 자동 페이지네이션 처리됨
                submissions = reddit.subreddit(sub_name).search(
                    query=query,
                    sort="new",
                    time_filter="all",
                    limit=300  # 한 달당 최대 300개 (필요시 늘릴 수 있음)
                )

                for submission in submissions:
                    created_time = dt.datetime.fromtimestamp(submission.created_utc)
                    if not (start_date <= created_time < end_date):
                        continue  # 월 범위 벗어난 글 제외

                    if submission.id not in existing_ids:
                        result.append({
                            "id": submission.id,
                            "title": submission.title,
                            "score": submission.score,
                            "created_at": created_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "selftext": submission.selftext if submission.selftext else "[No Content]",
                            "subreddit": sub_name,
                            "url": submission.url
                        })
                        existing_ids.add(submission.id)
                        collected_count += 1

                print(f"   → {collected_count}건 수집 완료 (누적 {len(existing_ids)}건)")
                time.sleep(3)  # Rate limit 완화용 대기

            except Exception as e:
                print(f"[ERROR] {year}-{month:02d} 수집 중 오류: {e}")
                time.sleep(5)

# 5. 저장
df_new = pd.DataFrame(result)
df_combined = pd.concat([df_existing, df_new], ignore_index=True).drop_duplicates(subset=["id"])

df_combined.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n✅ [완료] 총 {len(df_combined)}건 수집 완료. 저장 파일: {OUTPUT_FILE}")
