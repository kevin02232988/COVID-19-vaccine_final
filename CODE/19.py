import requests
from bs4 import BeautifulSoup
import time
import random
import csv
import hashlib
import os
import logging
import pandas as pd
from urllib.parse import urljoin
from datetime import datetime

# --------------------------- 설정 및 유틸리티 ---------------------------
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0',
    'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
]

COVID_KEYWORDS = [
    'vaccine', 'covid vaccine', 'side effects', 'booster', 'pfizer', 'moderna', 'antivax'
]

OUTPUT_FILE = "additional_forum_data.csv"
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
MIN_WORD_COUNT = 10  # 최소 단어 수 설정


def clean_text(text: str) -> str:
    txt = text.replace('\r', ' ').replace('\n', ' ').strip()
    return ' '.join(txt.split())


def contains_covid_keyword(text: str) -> bool:
    t = text.lower()
    for kw in COVID_KEYWORDS:
        if kw in t:
            return True
    return False


def http_get(url, timeout=10):
    headers = {'User-Agent': random.choice(USER_AGENTS), 'Accept-Language': 'en-US,en;q=0.9'}
    try:
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception:
        return None


def save_data(source, url, text, date_str, rows_list):
    if len(text.split()) < MIN_WORD_COUNT or not contains_covid_keyword(text):
        return 0

    # URL과 텍스트를 결합한 해시로 중복 체크 (간이 체크)
    h = hashlib.sha256((url + text).encode('utf-8')).hexdigest()
    if h in global_hashes:
        return 0

    global_hashes.add(h)

    rows_list.append({
        'source': source,
        'url': url,
        'text': text,
        'date': date_str
    })
    return 1


# --------------------------- 크롤러 함수 정의 ---------------------------

# 1. MedHelp.org 크롤러
def crawl_medhelp(rows_list, max_pages=100):
    logging.info(f"--- MedHelp.org 수집 시작 ---")
    base_search = "https://www.medhelp.org/posts/list?cid=604&search_text=covid%20vaccine&page="  # COVID-19 관련 포럼 검색

    for p in range(1, max_pages + 1):
        logging.info(f"[PAGE] MedHelp 페이지 {p} 수집 시도...")
        url = base_search + str(p)
        r = http_get(url)

        if not r:
            time.sleep(10)
            continue

        soup = BeautifulSoup(r.text, 'html.parser')

        post_links = soup.select("div.mh_post_title a")

        if not post_links:
            logging.info(f"MedHelp 페이지 {p}: 더 이상 링크가 없습니다.")
            break

        new_count = 0
        for link in post_links:
            post_url = "https://www.medhelp.org" + link['href']

            r_post = http_get(post_url)
            if not r_post:
                time.sleep(1)
                continue

            soup_post = BeautifulSoup(r_post.text, 'html.parser')

            # 본문 및 댓글 추출 (날짜 정보 추출 보강)
            posts = soup_post.select("div.post_text_container")  # 포스트 본문 및 댓글 텍스트
            dates = soup_post.select("span.post_date")  # 날짜 정보

            for post, date in zip(posts, dates):
                text = clean_text(post.get_text(" ", strip=True))
                # 날짜 문자열 정리 (예: 'Posted: Jul 03, 2021 11:22AM')
                date_str = clean_text(date.get_text(" ", strip=True)).replace("Posted:", "").strip()

                new_count += save_data('MedHelp', post_url, text, date_str, rows_list)

            time.sleep(random.uniform(0.5, 1.5))

        logging.info(f"MedHelp 페이지 {p}: {new_count}건 수집 완료. 누적: {len(rows_list)}건")
        time.sleep(random.uniform(2, 4))

    # 2. The Student Room 크롤러


def crawl_student_room(rows_list, max_pages=100):
    logging.info(f"--- The Student Room (TSR) 수집 시작 ---")
    # Health/Medicine 포럼에서 Vaccine/COVID 관련 스레드 검색
    base_search = "https://www.thestudentroom.co.uk/forum/search.php?do=process&query=covid+vaccine&forumchoice%5B%5D=16&page="

    for p in range(1, max_pages + 1):
        logging.info(f"[PAGE] TSR 검색 페이지 {p} 수집 시도...")
        url = base_search + str(p)
        r = http_get(url)

        if not r:
            time.sleep(10)
            continue

        soup = BeautifulSoup(r.text, 'html.parser')

        post_links = soup.select("a.title")

        if not post_links:
            logging.info(f"TSR 페이지 {p}: 더 이상 링크가 없습니다.")
            break

        new_count = 0
        for link in post_links:
            post_url = urljoin("https://www.thestudentroom.co.uk/", link['href'])

            r_post = http_get(post_url)
            if not r_post:
                time.sleep(1)
                continue

            soup_post = BeautifulSoup(r_post.text, 'html.parser')

            # 본문 및 댓글 추출 (TSR 구조에 맞춤)
            posts = soup_post.select("div.post__content")
            dates = soup_post.select("time.post__time")  # 날짜 정보 (가장 안정적)

            for post, date in zip(posts, dates):
                text = clean_text(post.get_text(" ", strip=True))
                # time 태그의 datetime 속성을 사용하여 정확한 날짜 추출
                date_str = date.get('datetime', clean_text(date.get_text(" ", strip=True)))

                new_count += save_data('TSR', post_url, text, date_str, rows_list)

            time.sleep(random.uniform(0.5, 1.5))

        logging.info(f"TSR 페이지 {p}: {new_count}건 수집 완료. 누적: {len(rows_list)}건")
        time.sleep(random.uniform(2, 4))

    # --------------------------- 메인 함수 ---------------------------


if __name__ == '__main__':

    global_rows_list = []
    global_hashes = set()

    # 1. 크롤링 시작
    crawl_medhelp(global_rows_list)
    crawl_student_room(global_rows_list)

    # 2. DataFrame 생성 및 저장
    if global_rows_list:
        df = pd.DataFrame(global_rows_list)

        # 날짜 컬럼 후처리 (필요하다면)
        # df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # 최종 파일 저장 (이전에 확보한 Reddit 데이터와 합치기 위해 임시 저장)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

        logging.info(f"\n--- 최종 수집 요약 ---")
        logging.info(f"총 수집된 유효 데이터: {len(df)}건")
        logging.info(f"CSV 저장 완료: {OUTPUT_FILE}")
    else:
        logging.warning("\n데이터를 수집하지 못했습니다. IP 차단이 강력한 것으로 보입니다.")

print("[DEBUG] 크롤링 종료")
print(f"[DEBUG] 총 수집된 데이터 수: {len(rows_list)}")
print(f"[DEBUG] 현재 작업 디렉토리: {os.getcwd()}")
