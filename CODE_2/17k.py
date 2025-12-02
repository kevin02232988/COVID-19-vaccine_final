import time
import csv
from datetime import datetime
import praw
import os
import pandas as pd
from prawcore.exceptions import ResponseException

# -------------------
# 설정
# -------------------
TARGET = 20000  # 목표 댓글 수
SAVE_EVERY = 1000  # 몇 건마다 CSV로 중간저장
OUT_CSV = "Reddit_COVID.csv"  # <--- 파일 이름 변경 적용

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
# Reddit 인증 (사용자 정보로 교체 필요)
# -------------------
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "covid-vaccine-scraper/0.1 by Delicious_Tough"

# PRAW 클라이언트 생성
try:
    reddit = praw.Reddit(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        username=USERNAME,
        password=PASSWORD,
        user_agent=USER_AGENT,
        check_for_async=False
    )
    reddit.read_only = True
    print("[INFO] Reddit API 인증 성공.")
except Exception as e:
    print(f"[ERROR] Reddit API 인증 실패: {e}")
    exit()


# -------------------
# 유틸: 날짜 변환
# -------------------
def utc_to_datestr(utc_ts):
    try:
        return datetime.utcfromtimestamp(int(utc_ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ""


# -------------------
# 체크포인트 로드 및 초기화 (핵심 체크포인트 기능)
# -------------------
def load_checkpoint(fname=OUT_CSV):
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname, usecols=['comment_id'], on_bad_lines='skip')
            # 기존에 수집된 모든 comment_id를 set으로 로드하여 중복 방지에 사용
            return set(df['comment_id'].astype(str).tolist()), len(df)
        except Exception as e:
            print(f"[WARNING] 체크포인트 로드 오류 ({e}). 새 파일로 시작합니다.")
            return set(), 0
    return set(), 0


# 전역 변수로 관리 (시작 시 로드)
seen_comment_ids, initial_count = load_checkpoint()
collected = []


# -------------------
# CSV 저장 함수
# -------------------
def save_rows(rows, fname=OUT_CSV):
    header = ["source", "subreddit", "submission_id", "submission_title", "comment_id", "author", "body", "created_utc",
              "date"]
    write_header = not os.path.exists(fname)
    with open(fname, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        for r in rows:
            # CSV 저장 시 콤마 문제 방지를 위해 모든 항목을 문자열로 변환하여 저장
            writer.writerow([str(r.get(h, "")) for h in header])

        # -------------------


# 댓글 처리
# -------------------
def process_submission(sub):
    try:
        sub.comments.replace_more(limit=0)
    except Exception:
        return []

    out = []
    try:
        comment_list = sub.comments.list()
    except Exception:
        return []

    for comment in comment_list:
        if comment.id in seen_comment_ids:
            continue

        text = (comment.body or "").lower()
        # 키워드 필터링
        if any(k.lower() in text for k in KEYWORDS):
            row = {
                "source": "reddit",
                "subreddit": sub.subreddit.display_name,
                "submission_id": sub.id,
                "submission_title": sub.title,
                "comment_id": comment.id,
                "author": str(comment.author) if comment.author else "",
                "body": comment.body.replace("\n", " ").strip(),
                "created_utc": int(comment.created_utc) if getattr(comment, "created_utc", None) else "",
                "date": utc_to_datestr(getattr(comment, "created_utc", ""))
            }
            out.append(row)
            # 새로운 comment_id를 seen_comment_ids에 추가
            seen_comment_ids.add(comment.id)

    return out


# -------------------
# 서브레딧 수집
# -------------------
def collect_from_subreddit(subreddit, limit_submissions=200):
    rows = []
    subreddit_obj = reddit.subreddit(subreddit)

    # .new() 리스팅에서 최신 게시글 200개를 가져옴
    for submission in subreddit_obj.new(limit=limit_submissions):
        # 게시글 제목에 필터링 키워드가 포함되어 있으면 댓글을 파싱 (수집 효율 증대)
        if any(k.lower() in submission.title.lower() for k in KEYWORDS):
            rows.extend(process_submission(submission))

        # 목표 도달 시 루프 탈출
        if len(rows) + len(collected) + initial_count >= TARGET:
            break

    return rows


# -------------------
# 메인 루프
# -------------------
try:
    current_total = initial_count
    print(f"댓글 수집 시작... 기존 {current_total}개 로드됨. 목표: {TARGET}개")

    while current_total < TARGET:
        any_found = False

        for sub in SUBREDDITS:
            rows = collect_from_subreddit(sub, limit_submissions=200)

            if rows:
                any_found = True
                collected.extend(rows)
                current_total = len(collected) + initial_count

                print(f"[{sub}] 수집된 댓글: {len(rows)}, 총 합계: {current_total}")

                # 중간 저장 (SAVE_EVERY 단위 또는 목표 달성 시)
                if (current_total >= TARGET) or (len(collected) >= SAVE_EVERY):
                    save_rows(collected)
                    print(f"[SAVE] {current_total}개 저장됨 -> {OUT_CSV}")
                    collected = []  # 저장 후 collected 리스트 비우기

            time.sleep(2.0)  # 서브레딧 간 2초 대기 (Rate Limit 방지)

            if current_total >= TARGET:
                break

        if not any_found and current_total < TARGET:
            print("새로운 댓글을 찾지 못했습니다. 30초 대기 후 재시도...")
            time.sleep(30)

        if current_total >= TARGET:
            break

    # 마지막 남은 댓글 저장
    if collected:
        save_rows(collected)
        print(f"최종 저장 완료: {OUT_CSV}")

    print("수집 완료!")

except KeyboardInterrupt:
    print("\n[INFO] 사용자 중단 감지. 현재 수집된 데이터 저장 시도...")
    if collected:
        save_rows(collected)
        print(f"[SAVE] 최종 중단 시점 데이터 저장 완료: {OUT_CSV}")
except Exception as e:
    print(f"\n[CRITICAL ERROR] 오류 발생: {e}. 현재 수집된 데이터 저장 시도...")
    if collected:
        save_rows(collected)
finally:
    current_total = len(collected) + initial_count
    print(f"\n프로그램 종료. 최종 데이터 개수 확인: {current_total}건.")