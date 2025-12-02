import requests
import pandas as pd
import datetime as dt
import time

SUBREDDITS = ["CovidVaccinated", "Coronavirus", "vaccine", "COVID19", "COVID19Vaccine"]
START_DATE = int(dt.datetime(2020, 3, 1).timestamp())
END_DATE = int(dt.datetime(2024, 12, 31).timestamp())
OUTFILE = "reddit_covid_vaccine_pushshift.csv"

API_BASE = "https://reddit-data-api.quantumbyte.dev/search/submission/"  # 새 엔드포인트

all_posts = []

for sub in SUBREDDITS:
    print(f"[INFO] r/{sub} 수집 중...")
    after = START_DATE

    while after < END_DATE:
        url = f"{API_BASE}?subreddit={sub}&q=vaccine&after={after}&before={after+2592000}&size=500"
        try:
            response = requests.get(url, timeout=20)
            if not response.text.strip():
                print(f"  [WARNING] 빈 응답 - {dt.datetime.fromtimestamp(after).strftime('%Y-%m')}")
                after += 2592000
                time.sleep(2)
                continue

            data = response.json().get("data", [])
            if not data:
                print(f"  [INFO] 데이터 없음 ({dt.datetime.fromtimestamp(after).strftime('%Y-%m')})")
                after += 2592000
                continue

            for d in data:
                all_posts.append({
                    "id": d.get("id"),
                    "title": d.get("title"),
                    "selftext": d.get("selftext"),
                    "score": d.get("score"),
                    "created_utc": dt.datetime.fromtimestamp(d["created_utc"]).strftime("%Y-%m-%d %H:%M:%S"),
                    "subreddit": sub
                })

            print(f"  → {len(data)}건 수집 완료 ({dt.datetime.fromtimestamp(after).strftime('%Y-%m')})")
            after += 2592000  # 다음 달
            time.sleep(2)

        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(5)
            continue

df = pd.DataFrame(all_posts).drop_duplicates(subset=["id"])
df.to_csv(OUTFILE, index=False, encoding="utf-8-sig")
print(f"\n[완료] 총 {len(df)}건 수집 완료. 파일 저장: {OUTFILE}")
