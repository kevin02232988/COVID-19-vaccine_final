import praw
import pandas as pd
import datetime

# 1. Reddit API 인증 정보 (여기에 발급받은 정보로 교체 필요)
# 이 네 변수를 사용자님의 실제 정보로 반드시 교체해야 합니다.
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "controversy_analyzer_v2 by /u/YOUR_USERNAME"

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
SUBREDDIT_LIST = ["Coronavirus", "vaccine"]  # <--- 대형 서브레딧
LIMIT_PER_SUB = 10000  # 서브레딧 당 10,000건 목표
output_file = "reddit_final_controversy_posts_new.csv"

# 기존 데이터 불러오기 (이전 파일이 있다면 합치기 위함)
try:
    df_combined = pd.read_csv("reddit_final_controversy_posts_new.csv")
    print(f"[INFO] 기존 데이터 총 {len(df_combined)}건 불러옴. 새 데이터와 통합합니다.")
except FileNotFoundError:
    print("[INFO] 기존 파일이 없어 새 데이터프레임으로 시작합니다.")
    df_combined = pd.DataFrame()

# 4. 데이터 수집 루프 (서브레딧 순회)
for sub_name in SUBREDDIT_LIST:
    result = []
    print(f"\n--- 서브레딧 r/{sub_name} 데이터 수집 시작 (Top 10,000건 목표) ---")

    # .top() 메소드를 사용하여 가장 논란이 많았던 인기 게시글부터 수집
    for submission in reddit.subreddit(sub_name).top(time_filter="all", limit=LIMIT_PER_SUB):
        timestamp = datetime.datetime.fromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S')

        submission_data = {
            'id': submission.id,
            'title': submission.title,
            'score': submission.score,
            'num_comments': submission.num_comments,
            'created_at': timestamp,
            'url': submission.url,
            'selftext': submission.selftext if submission.selftext else '[No Content]',
            'topic_type': sub_name
        }
        result.append(submission_data)

        if len(result) % 1000 == 0:
            print(f"[INFO] r/{sub_name} 현재 {len(result)}개의 게시글 수집 완료.")

    df_new = pd.DataFrame(result)
    df_combined = pd.concat([df_combined, df_new]).drop_duplicates(subset=['id'])

# 5. DataFrame 저장 및 결과 출력
df_combined.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 최종 완료. 총 {len(df_combined)}건의 게시글을 수집했습니다.")
print(f"파일 저장 완료: '{output_file}'")