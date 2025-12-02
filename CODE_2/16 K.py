# reddit_collect_praw_full.py
import time
import csv
from datetime import datetime
import praw
import os

# -------------------
# 설정
# -------------------
TARGET = 20000                # 목표 댓글 수
SAVE_EVERY = 1000             # 몇 건마다 CSV로 중간저장
OUT_CSV = "reddit_comments_praw.csv"

# 서브레딧 리스트 (코로나/백신 관련)
SUBREDDITS = [
    "Coronavirus",
    "COVID19",
    "CovidVaccinated",
    "vaccines",
    "COVIDVaccine"
]

# 키워드 필터 (본문/댓글에 포함되면 수집대상)
KEYWORDS = ["vaccine", "covid vaccine", "covid-19 vaccine", "side effects", "side-effect", "adverse"]

# -------------------
# Reddit 인증 (환경변수 없이 직접 입력)
# -------------------
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "covid-vaccine-scraper/0.1 by Delicious_Tough"

# -------------------
# PRAW 클라이언트 생성
# -------------------
reddit = praw.Reddit(
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    username=USERNAME,
    password=PASSWORD,
    user_agent=USER_AGENT,
    check_for_async=False
)

# -------------------
# 유틸: 날짜 변환
# -------------------
def utc_to_datestr(utc_ts):
    try:
        return datetime.utcfromtimestamp(int(utc_ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ""

# -------------------
# CSV 저장 함수
# -------------------
def save_rows(rows, fname=OUT_CSV):
    header = ["source","subreddit","submission_id","submission_title","comment_id","author","body","created_utc","date"]
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for r in rows:
            writer.writerow([r.get(h,"") for h in header])

# -------------------
# 댓글 처리
# -------------------
collected = []
seen_comment_ids = set()

def process_submission(sub):
    try:
        sub.comments.replace_more(limit=0)
    except Exception:
        return []
    out = []
    for comment in sub.comments.list():
        if comment.id in seen_comment_ids:
            continue
        text = (comment.body or "").lower()
        if any(k.lower() in text for k in KEYWORDS):
            row = {
                "source": "reddit",
                "subreddit": sub.subreddit.display_name,
                "submission_id": sub.id,
                "submission_title": sub.title,
                "comment_id": comment.id,
                "author": str(comment.author) if comment.author else "",
                "body": comment.body.replace("\n"," ").strip(),
                "created_utc": int(comment.created_utc) if getattr(comment, "created_utc", None) else "",
                "date": utc_to_datestr(getattr(comment, "created_utc", ""))
            }
            out.append(row)
            seen_comment_ids.add(comment.id)
    return out

# -------------------
# 서브레딧 수집
# -------------------
def collect_from_subreddit(subreddit, limit_submissions=200):
    rows = []
    subreddit_obj = reddit.subreddit(subreddit)
    for submission in subreddit_obj.new(limit=limit_submissions):
        rows.extend(process_submission(submission))
        if len(rows) + len(collected) >= TARGET:
            break
    return rows

# -------------------
# 메인 루프
# -------------------
try:
    print("댓글 수집 시작...")
    while len(collected) < TARGET:
        any_found = False
        for sub in SUBREDDITS:
            rows = collect_from_subreddit(sub, limit_submissions=200)
            if rows:
                any_found = True
                collected.extend(rows)
                print(f"[{sub}] 수집된 댓글: {len(rows)}, 총 합계: {len(collected)}")
                if len(collected) % SAVE_EVERY == 0 or len(collected) >= TARGET:
                    save_rows(collected)
                    print(f"[SAVE] {len(collected)}개 저장됨 -> {OUT_CSV}")
                    collected = []
            time.sleep(1.0)
            if len(collected) >= TARGET:
                break
        if not any_found:
            print("새로운 댓글을 찾지 못했습니다. 30초 대기 후 재시도...")
            time.sleep(30)

    if collected:
        save_rows(collected)
        print(f"최종 저장 완료: {OUT_CSV}")

    print("수집 완료!")

except Exception as e:
    print("오류 발생:", e)
