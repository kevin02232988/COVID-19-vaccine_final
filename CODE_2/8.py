import praw
import pandas as pd
import datetime as dt
import time

# 1. Reddit API 인증 정보 (여기에 발급받은 정보로 교체 필요)
# 이 네 변수를 사용자님의 실제 정보로 반드시 교체해야 합니다.
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "controversy_analyzer_final by /u/YOUR_USERNAME"

# 2. PRAW 인스턴스 초기화
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

# 3. 크롤링 설정
TARGET_SUBREDDITS = ["CovidVaccinated", "Coronavirus", "vaccine", "COVID19", "COVID19Vaccine"]
START_YEAR = 2020
END_YEAR = 2024

output_file = "reddit_final_controversy_posts_segmented_praw.csv"

# 기존 데이터 불러오기 및 중복 방지
try:
    df_combined = pd.read_csv(output_file)
    existing_ids = set(df_combined['id'])
    print(f"[INFO] 기존 파      일에서 {len(existing_ids)}건의 데이터를 불러왔습니다.")
except FileNotFoundError:
    df_combined = pd.DataFrame()
    existing_ids = set()
    print("[INFO] 기존 파일이 없어 빈 데이터프레임으로 시작합니다.")

total_collected_new = 0
result = []

# 4. 시간대 분할 및 수집 루프 (월별)
for sub_name in TARGET_SUBREDDITS:
    print(f"\n--- 서브레딧 r/{sub_name} 데이터 수집 시작 (월별 분할) ---")

    for year in range(START_YEAR, END_YEAR + 1):
        # 2020년은 3월부터 시작 (코로나 초기)
        start_month = 3 if year == 2020 else 1

        for month in range(start_month, 13):
            # 다음 달의 시작 시간 = 현재 달의 끝 시간
            try:
                # 현재 달의 시작 시간 (after)
                start_of_month = int(dt.datetime(year, month, 1).timestamp())

                # 다음 달의 시작 시간 (before)
                if month == 12:
                    end_of_month = int(dt.datetime(year + 1, 1, 1).timestamp()) - 1
                else:
                    end_of_month = int(dt.datetime(year, month + 1, 1).timestamp()) - 1
            except ValueError:
                continue

            # PRAW는 t:all 필터와 함께 after/before 파라미터를 사용하지 못하므로,
            # search() 메소드와 custom time filter를 사용해야 합니다.
            time_filter = f"timestamp:{start_of_month}..{end_of_month}"

            print(f"  [TIME] {year}-{month:02d} 데이터 수집 시도...")

            try:
                # search() 메소드를 사용하여 시간대 필터링
                submissions = reddit.subreddit(sub_name).search(
                    query='vaccine OR covid',  # 관련 키워드 포함
                    sort='new',
                    time_filter='all',
                    limit=None,  # 최대한 많은 데이터를 가져오기 위해 limit=None (자동 페이징)
                    params={'before': end_of_month, 'after': start_of_month}
                )

                collected_count = 0
                for submission in submissions:
                    if submission.id not in existing_ids:  # 중복 방지
                        timestamp = dt.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

                        result.append({
                            'id': submission.id,
                            'title': submission.title,
                            'score': submission.score,
                            'created_at': timestamp,
                            'selftext': submission.selftext if submission.selftext else '[No Content]',
                            'topic_type': sub_name
                        })
                        existing_ids.add(submission.id)
                        collected_count += 1

                print(f"    → {collected_count}건 수집 완료 (누적 {len(existing_ids)}건)")

            except Exception as e:
                # API 요청 실패 (Rate Limit 등) 시
                print(f"    [ERROR] API 요청 실패 ({e}). 이 구간은 건너뜁니다.")

            time.sleep(5)  # API 요청 간 5초 대기 (서버 부하 및 Rate Limit 방지)

# 5. 최종 저장
df_new = pd.DataFrame(result)
df_combined = pd.concat([df_combined, df_new]).drop_duplicates(subset=['id'])

df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 최종 완료. 총 {len(df_combined)}건의 게시글을 수집했습니다.")
print(f"파일 저장 완료: '{output_file}'")