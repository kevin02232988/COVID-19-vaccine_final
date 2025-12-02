import praw
import pandas as pd
import datetime

# 1. Reddit API 인증 정보 (여기에 발급받은 정보로 교체 필요)
# 이 네 변수를 사용자님의 실제 정보로 반드시 교체해야 합니다.
CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
USERNAME = "YOUR_USERNAME"
PASSWORD = "YOUR_PASSWORD"
USER_AGENT = "covid_vaccine_analyzer_v1 by /u/YOUR_USERNAME"

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
SUBREDDIT_NAME = "CovidVaccinated"
LIMIT = 20000  # <--- 수집 목표 20,000건으로 증가
result = []

print(f"--- Reddit 서브레딧 r/{SUBREDDIT_NAME} 최신 게시글 수집 시작 (20,000건 목표) ---")

# 4. 데이터 수집 루프: .new() 메소드 사용
for submission in reddit.subreddit(SUBREDDIT_NAME).new(limit=LIMIT):
    timestamp = datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

    submission_data = {
        'id': submission.id,
        'title': submission.title,
        'score': submission.score,
        'num_comments': submission.num_comments,
        'created_at': timestamp,
        'url': submission.url,
        'selftext': submission.selftext if submission.selftext else '[No Content]',
        'topic_type': 'Vaccine'
    }
    result.append(submission_data)

    if len(result) % 5000 == 0:
        print(f"[INFO] 현재 {len(result)}개의 게시글 수집 완료.")

# 5. 기존 데이터와 합치기 및 중복 제거
df_new = pd.DataFrame(result)

try:
    # 이전에 수집한 'top' 리스팅 데이터 불러오기
    df_old = pd.read_csv("reddit_covid_vaccine_posts.csv")
except FileNotFoundError:
    print("[WARNING] 기존 파일 (reddit_covid_vaccine_posts.csv)을 찾을 수 없습니다. 새로운 데이터만 저장합니다.")
    df_old = pd.DataFrame()  # 빈 데이터프레임으로 초기화

df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['id'])

output_file = "reddit_covid_vaccine_combined.csv"
df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 최종 완료. 총 {len(df_combined)}건의 게시글을 수집했습니다.")
print(f"파일 저장 완료: '{output_file}'")