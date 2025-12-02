import praw
import pandas as pd
import datetime as dt
import time
import csv
import os  # <--- 누락된 os 모듈 추가
from datetime import datetime
from prawcore.exceptions import ResponseException

# ------------------- 설정 -------------------
TARGET_COMMENT_COUNT = 100000  # <--- 댓글 10만 건 목표
OUT_CSV = "reddit_comments_final_bulk.csv"

# ------------------- Reddit 인증 정보 (사용자 정보로 교체 필요) -------------------
CLIENT_ID = "wkiDDqFG7tgXo_1U5tyuZA"
CLIENT_SECRET = "69T2Z_QlECA4vZOddorn66jT0kk6vA"
USERNAME = "Delicious_Tough_2446"
PASSWORD = "kevin02233988@"
USER_AGENT = "bulk_comment_collector_final_v1 by /u/Delicious_Tough_2446"

# ------------------- PRAW 클라이언트 생성 -------------------
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


# ------------------- 유틸리티 함수 -------------------
def utc_to_datestr(utc_ts):
    try:
        return datetime.utcfromtimestamp(int(utc_ts)).strftime("%Y-%m-%d %H:%M:%S")
    except:
        return ""


# ------------------- 댓글 수집 함수 (핵심 로직) -------------------
def collect_comments_from_submission(submission):
    comments_list = []

    # 댓글 '더보기'를 모두 확장 (limit=None)
    try:
        submission.comments.replace_more(limit=None)
    except Exception:
        return []

    # 모든 댓글을 순회
    for comment in submission.comments.list():
        # 삭제되거나 자동으로 제거된 댓글은 제외
        if not comment.author or comment.body in ['[deleted]', '[removed]']:
            continue

        comments_list.append({
            "source": "Reddit",
            "submission_id": submission.id,
            "submission_title": submission.title,
            "comment_id": comment.id,
            "body": comment.body.replace("\n", " ").strip(),
            "date": utc_to_datestr(getattr(comment, "created_utc", ""))
        })
    return comments_list


# ------------------- CSV 저장 함수 (기존 파일 로드 및 저장) -------------------
def save_rows(rows, fname=OUT_CSV, fieldnames=None, mode='a', write_header=False):
    if fieldnames is None:
        fieldnames = ["source", "submission_id", "submission_title", "comment_id", "body", "date"]

    # 파일이 존재하지 않으면 헤더를 작성해야 합니다.
    write_header = not os.path.exists(fname)

    with open(fname, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ------------------- 메인 루프 -------------------
try:
    # 1. 기존 데이터 로드 (체크포인트)
    OUT_CSV = "reddit_comments_final_bulk.csv"  # 파일명 재확인
    existing_count = 0
    if os.path.exists(OUT_CSV):
        try:
            df_existing = pd.read_csv(OUT_CSV)
            existing_count = len(df_existing)
            print(f"[INFO] 기존 파일에서 {existing_count}개 댓글 로드됨. 이어서 수집.")
        except Exception:
            pass

    collected_rows = []
    current_total = existing_count

    # 2. r/Coronavirus 에서 Top 1000개의 논란 게시글 확보
    subreddit_name = "Coronavirus"
    print(f"\n--- 1단계: r/{subreddit_name} Top 1000 게시글 확보 ---")

    submissions = reddit.subreddit(subreddit_name).top(time_filter="all", limit=1000)

    print(f"--- 2단계: 댓글 대량 수집 시작 (10만 건 목표) ---")

    for sub in submissions:
        # 이미 10만 건을 달성했으면 중단
        if current_total >= TARGET_COMMENT_COUNT:
            break

        print(f"  [SUBMISSION] '{sub.title[:50]}...' 댓글 수집 중...")

        # 댓글 대량 수집 함수 호출
        comments = collect_comments_from_submission(sub)

        if comments:
            collected_rows.extend(comments)
            current_total = existing_count + len(collected_rows)
            print(f"    -> {len(comments)}개 댓글 수집. 총 합계: {current_total}")

        # 중간 저장 (5000개 단위)
        if len(collected_rows) >= 5000:
            save_rows(collected_rows)
            existing_count = current_total
            collected_rows = []
            print(f"[SAVE] 5000개 중간 저장 완료. 누적: {current_total}")
            time.sleep(10)  # 대량 요청 후 10초 대기

        time.sleep(1)  # 게시글 간 1초 대기

    # 3. 최종 저장
    if collected_rows:
        save_rows(collected_rows)

    final_count = existing_count + len(collected_rows)
    print(f"\n[INFO] 크롤링 최종 완료. 총 {final_count}건의 댓글을 수집했습니다.")
    print(f"파일 저장 완료: {OUT_CSV}")

except KeyboardInterrupt:
    print("\n[INFO] 사용자 중단 감지. 현재 수집된 데이터 저장 시도...")
    if collected_rows:
        save_rows(collected_rows)
        print(f"[SAVE] 최종 중단 시점 데이터 저장 완료: {OUT_CSV}")
except Exception as e:
    print(f"\n[CRITICAL ERROR] 오류 발생: {e}. 현재 수집된 데이터 저장 시도...")
    if collected_rows:
        save_rows(collected_rows)
finally:
    current_total = existing_count + len(collected_rows)
    print(f"\n프로그램 종료. 최종 데이터 개수 확인: {current_total}건.")