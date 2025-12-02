import requests
import pandas as pd
import time
import datetime as dt

# 1. 수집 설정
SUBREDDIT = 'Coronavirus'
# 2020년 3월 1일 00:00:00 UTC
START_TIME = int(dt.datetime(2020, 3, 1).timestamp())
# 2022년 12월 31일 23:59:59 UTC
END_TIME = int(dt.datetime(2022, 12, 31).timestamp())

LIMIT = 1000  # API 한 번 요청 시 최대 수집 개수
total_collected = 0
result = []

print(f"--- Pushshift API를 사용한 r/{SUBREDDIT} 데이터 수집 시작 (2020.03 ~ 2022.12) ---")

current_time = END_TIME

while current_time > START_TIME and total_collected < 20000:
    url = (
        f"https://api.pushshift.io/reddit/submission/search/?subreddit={SUBREDDIT}"
        f"&after={START_TIME}&before={current_time}&size={LIMIT}&sort=desc&sort_type=created_utc"
    )

    try:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()
        posts = data['data']

        if not posts:
            print("[STOP] 더 이상 게시글이 없습니다. 수집 종료.")
            break

        for post in posts:
            timestamp = dt.datetime.fromtimestamp(post['created_utc']).strftime('%Y-%m-%d %H:%M:%S')

            # selftext: 게시글 본문. 제거되었을 경우 [No Content] 처리
            selftext = post.get('selftext', '').replace('\n', ' ')

            result.append({
                'id': post['id'],
                'title': post['title'],
                'score': post['score'],
                'num_comments': post['num_comments'],
                'created_at': timestamp,
                'selftext': selftext,
                'topic_type': 'Coronavirus'
            })

        # 다음 요청을 위해 before 시간을 가장 최근 수집된 게시글 시간으로 업데이트
        current_time = posts[-1]['created_utc']
        total_collected = len(result)

        print(f"[INFO] 현재 {total_collected}건 수집 완료 (마지막 시간: {dt.datetime.fromtimestamp(current_time)})")

    except Exception as e:
        print(f"[ERROR] 요청 중 오류 발생: {e}. 잠시 대기 후 재시도.")
        time.sleep(10)

    time.sleep(2)  # API 부하 방지

# 2. 기존 데이터와 합치기 및 저장
df_new = pd.DataFrame(result)
output_file = "reddit_covid_vaccine_combined.csv"

try:
    df_old = pd.read_csv(output_file)
except FileNotFoundError:
    df_old = pd.DataFrame()

df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['id'])

df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 최종 완료. 총 {len(df_combined)}건의 게시글을 수집했습니다.")
print(f"파일 저장 완료: '{output_file}'")