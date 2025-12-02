"""
WebMD + HealthBoards 통합 크롤러 (COVID-19 vaccine target)
목표: 날짜를 포함한 감정적 여론 데이터 2만 건 이상 수집 (URL Path 오류 수정)
"""
import requests
from bs4 import BeautifulSoup
import time
import random
import csv
import hashlib
import os
import json
import argparse
import logging
from urllib.parse import urljoin
from tqdm import tqdm

# --------------------------- 설정 및 유틸리티 ---------------------------
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.64 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:100.0) Gecko/20100101 Firefox/100.0',
    'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
]

DEFAULT_HEADERS = {
    'Accept-Language': 'en-US,en;q=0.9',
}

COVID_KEYWORDS = [
    'covid', 'covid-19', 'coronavirus', 'vaccine', 'pfizer', 'moderna', 'astrazeneca',
    'side effect', 'side-effect', 'side effects', 'booster', 'antivax', 'vax', 'myocarditis'
]

def clean_text(text: str) -> str:
    txt = text.replace('\r', ' ').replace('\n', ' ').strip()
    txt = ' '.join(txt.split())
    return txt

def contains_covid_keyword(text: str) -> bool:
    t = text.lower()
    for kw in COVID_KEYWORDS:
        if kw in t:
            return True
    return False

def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def http_get(url, headers=None, proxies=None, timeout=15):
    h = DEFAULT_HEADERS.copy()
    h['User-Agent'] = random.choice(USER_AGENTS)
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, headers=h, proxies=proxies, timeout=timeout)
        r.raise_for_status()
        return r
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP Error {e.response.status_code}: {url}")
        if e.response.status_code in [403, 429]:
            raise Exception("CRITICAL_BLOCK")
        return None
    except Exception as e:
        logging.debug(f"HTTP GET failed: {url} -> {e}")
        return None

def extract_candidate_texts(soup: BeautifulSoup):
    selectors = [
        'div.user-comment', 'div.user-review', 'div.review', 'div.post-content',
        'div.message-body', 'div.comment', 'div.entry'
    ]
    texts = []
    for sel in selectors:
        found = soup.select(sel)
        for f in found:
            txt = clean_text(f.get_text(" ", strip=True))
            if len(txt) > 30:
                texts.append(txt)
    return texts

def append_to_csv(path, rows, header=None):
    exists = os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not exists and header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoint(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --------------------------- 사이트별 크롤러 클래스 ---------------------------

class UnifiedCrawler:
    def __init__(self, out_csv, checkpoint_path, proxies=None, min_words=10, verbose=True):
        self.out_csv = out_csv
        self.checkpoint_path = checkpoint_path
        self.proxies = proxies
        self.min_words = min_words
        self.seen_hashes = set()
        self.checkpoint = load_checkpoint(checkpoint_path)
        self.verbose = verbose
        # CSV 헤더에 'date' 필드 포함
        self.csv_header = ['source', 'url', 'text', 'date']

        if os.path.exists(out_csv):
            try:
                with open(out_csv, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) >= 3:
                            h = hash_text(row[1] + row[2])
                            self.seen_hashes.add(h)
            except Exception as e:
                logging.warning(f"Failed to load existing hashes: {e}")

    def save_texts(self, source, url, text, date_str):
        if len(text.split()) < self.min_words:
            return 0
        if not contains_covid_keyword(text):
            return 0

        h = hash_text(url + text)
        if h in self.seen_hashes:
            return 0

        self.seen_hashes.add(h)
        row = [source, url, text, date_str]
        append_to_csv(self.out_csv, [row], header=self.csv_header)
        return 1

    # ------------------ WebMD 크롤러 (링크 수집) ------------------
    def crawl_webmd_forum_list(self, base_forum_url, max_pages=1000, delay=(2.0, 4.0), start_page=1):
        logging.info(f"Starting WebMD Forum crawl: {base_forum_url}")

        for page in tqdm(range(start_page, max_pages + 1), desc="WebMD pages"):
            # URL 포맷: .../forum/page/N
            url = base_forum_url.rstrip('/') + f'/page/{page}'

            r = http_get(url, proxies=self.proxies)
            if not r:
                if isinstance(r, Exception) and "CRITICAL_BLOCK" in str(r):
                    logging.error("WebMD IP 차단! 크롤링을 중단합니다.")
                    break
                time.sleep(random.uniform(*delay))
                continue

            soup = BeautifulSoup(r.text, 'html.parser')
            thread_links = set()

            for a in soup.select("a.topic-title"):
                href = a.get('href')
                if href:
                    thread_links.add(urljoin("https://exchanges.webmd.com", href))

            if not thread_links:
                logging.info(f"Page {page}: No more thread links found.")
                break

            new_items_page = 0
            for tlink in tqdm(thread_links, desc=f"WebMD links (page {page})", leave=False):
                try:
                    # WebMD는 구조가 복잡하므로 날짜 없이 일반 텍스트만 추출하는 로직 사용
                    new_items_page += self.process_page('webmd', tlink)
                except Exception as e:
                    logging.debug(f"Error processing WebMD link {tlink}: {e}")

            logging.info(f"Saved {new_items_page} new items on page {page}. Total seen: {len(self.seen_hashes)}")
            self.checkpoint['webmd_last_page'] = page
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

        logging.info("WebMD crawling finished.")

    # ------------------ HealthBoards 크롤러 (링크 순회) ------------------
    def crawl_healthboards_forum(self, forum_base_url, max_pages=1000, start_page=1, delay=(2.0,4.0)):
        logging.info(f"Starting HealthBoards crawl: {forum_base_url}")

        for p in tqdm(range(start_page, max_pages+1), desc="HealthBoards pages"):
            # URL 오류 수정 핵심: 1페이지는 기본 주소, 2페이지부터 page-N 형식으로 만듭니다.
            if p == 1:
                page_url = forum_base_url.rstrip('/') + '/'
            else:
                # HealthBoards의 포럼 페이지 URL은 '.../forum/page-N' 형태를 따릅니다.
                # forum_base_url이 'https://www.healthboards.com/boards/coronavirus/'라고 가정
                page_url = forum_base_url.rstrip('/') + f'/{p}'

            r = http_get(page_url, proxies=self.proxies)
            if not r:
                if isinstance(r, Exception) and "CRITICAL_BLOCK" in str(r):
                    logging.error("HealthBoards IP 차단! 크롤링을 중단합니다.")
                    break
                time.sleep(random.uniform(*delay))
                continue

            soup = BeautifulSoup(r.text, 'html.parser')
            thread_links = set()

            # 스레드 링크 추출 (showthread 패턴)
            for a in soup.select("a[href*='showthread']"):
                href = a['href']
                thread_links.add(urljoin(forum_base_url, href))

            if not thread_links:
                logging.info(f"Page {p}: No thread links found.")
                break

            new_items_page = 0
            for tlink in tqdm(thread_links, desc=f"HealthBoards threads (page {p})", leave=False):
                try:
                    # HealthBoards는 날짜 추출 로직을 process_page에서 사용합니다.
                    new_items_page += self.process_page('healthboards', tlink)
                except Exception as e:
                    logging.debug(f"Error processing HealthBoards thread {tlink}: {e}")

            logging.info(f"Saved {new_items_page} new items on page {p}. Total seen: {len(self.seen_hashes)}")
            self.checkpoint['healthboards_last_page'] = p
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

        logging.info("HealthBoards crawling finished.")


    # ------------------ 공용: 개별 스레드/페이지 처리 ------------------
    def process_page(self, source, url):
        r = http_get(url, proxies=self.proxies)
        if not r:
            return 0
        soup = BeautifulSoup(r.text, 'html.parser')

        new_count = 0

        # HealthBoards: 날짜와 본문 모두 추출
        if 'healthboards.com' in url.lower():
            messages = soup.select('article.message')

            if not messages:
                # 스레드 본문이 댓글 형식 외로 있는 경우 (날짜 없이)
                texts = extract_candidate_texts(soup)
                for t in texts:
                    new_count += self.save_texts(source, url, t, 'Unknown Date (Post)')
                return new_count

            for message in messages:
                # 1. 텍스트 추출
                content_element = message.select_one('.message-content')
                raw_text = content_element.get_text(" ", strip=True) if content_element else ''
                text = clean_text(raw_text)

                # 2. 날짜 추출 (HealthBoards의 날짜 추출 로직)
                date_element = message.select_one('time')
                if date_element and 'datetime' in date_element.attrs:
                    date_str = date_element['datetime']
                elif date_element:
                    date_str = date_element.get_text(strip=True)
                else:
                    date_str = 'Unknown Date'

                # 3. 키워드 필터링 및 저장 (날짜 포함)
                new_count += self.save_texts(source, url, text, date_str)

            return new_count

        # WebMD & 기타: 날짜 없이 일반 텍스트 추출
        else:
            texts = extract_candidate_texts(soup)
            for t in texts:
                new_count += self.save_texts(source, url, t, 'Unknown Date (WebMD)')
            return new_count


# --------------------------- 메인 스크립트 (완전본) ---------------------------

def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoint(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, obj, ensure_ascii=False, indent=2)

def main():
    parser = argparse.ArgumentParser(description='WebMD + HealthBoards unified crawler (COVID-19 target)')
    parser.add_argument('--out', type=str, default='covid_vaccine_reviews_final.csv', help='Output CSV path')
    parser.add_argument('--checkpoint', type=str, default='crawler_checkpoint.json', help='Checkpoint JSON path')
    parser.add_argument('--max-pages', type=int, default=1000, help='Max pages per site (heuristic)')
    parser.add_argument('--site', choices=['both','webmd','healthboards'], default='both', help='Which site to crawl')
    # WebMD 포럼 URL (페이지 형식: /forum/page/N)
    parser.add_argument('--webmd-base', type=str, default='https://exchanges.webmd.com/covid-19-exchange/forum', help='WebMD base forum URL')
    # HealthBoards 포럼 URL (페이지 형식: /boards/forum/N)
    parser.add_argument('--healthboards-base', type=str, default='https://www.healthboards.com/boards/coronavirus', help='HealthBoards forum base URL')
    parser.add_argument('--delay-min', type=float, default=2.0)
    parser.add_argument('--delay-max', type=float, default=4.0)
    parser.add_argument('--proxy', type=str, default=None, help='Optional proxy (http://user:pass@host:port)')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    # 로그 레벨 INFO로 설정 (verbose=True 기본)
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    proxies = {'http': args.proxy, 'https': args.proxy} if args.proxy else None

    # 최소 단어 수 10개로 설정 (노이즈 감소)
    crawler = UnifiedCrawler(out_csv=args.out, checkpoint_path=args.checkpoint, proxies=proxies, verbose=args.verbose, min_words=10)

    cp = load_checkpoint(args.checkpoint)

    # 1. WebMD 수집
    if args.site in ('both','webmd'):
        webmd_start = cp.get('webmd_last_page', 1)
        crawler.crawl_webmd_forum_list(base_forum_url=args.webmd_base, max_pages=args.max_pages, delay=(args.delay_min, args.delay_max), start_page=webmd_start)

    # 2. HealthBoards 수집
    if args.site in ('both','healthboards'):
        hb_start = cp.get('healthboards_last_page', 1)
        crawler.crawl_healthboards_forum(forum_base_url=args.healthboards_base, max_pages=args.max_pages, delay=(args.delay_min, args.delay_max), start_page=hb_start)

    logging.info('Crawling finished. Please check the output CSV file.')


if __name__ == '__main__':
    main()