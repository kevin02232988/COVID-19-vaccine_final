"""
WebMD + HealthBoards 통합 크롤러 (COVID-19 vaccine target)

사용법:
    python webmd_healthboards_crawler_covid_vaccine.py --out data/covid_vaccine_reviews.csv --max-pages 1000

주요 특징:
- WebMD 및 HealthBoards의 기사/리뷰/포럼 글을 수집 (COVID-19 백신 관련 텍스트 필터링)
- 페이지 단위 저장(중단/재개 가능), CSV 출력, 중복 제거
- 사용자 에이전트 로테이션, rate-limit(지연), 간단한 proxy 지원 옵션
- HTML 구조가 다른 사이트에 대해 "유연한 텍스트 추출" 휴리스틱 적용

주의사항:
- 사이트 구조 변경이나 로봇 배제 정책(robots.txt)에 주의하세요. 대량 수집 전 이용약관을 확인하세요.
- 일부 페이지는 JS로 렌더링되어 requests만으로는 콘텐츠가 나오지 않을 수 있습니다. 이 경우 Selenium / requests_html 사용을 권장합니다.

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
from urllib.parse import urljoin, urlparse
from tqdm import tqdm

# --------------------------- 설정 및 유틸리티 ---------------------------
USER_AGENTS = [
    # 간단한 User-Agent 샘플 목록 (원하면 더 추가)
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0 Safari/537.36',
]

DEFAULT_HEADERS = {
    'Accept-Language': 'en-US,en;q=0.9',
}

COVID_KEYWORDS = [
    'covid', 'covid-19', 'coronavirus', 'vaccine', 'pfizer', 'moderna', 'astrazeneca', 'johnson & johnson', 'janssen',
    'sputnik', 'sinovac', 'side effect', 'side-effect', 'side effects', 'injection', 'shot', 'booster', 'anti-vax', 'vax'
]

# 간단한 텍스트 정제
def clean_text(text: str) -> str:
    txt = text.replace('\r', ' ').replace('\n', ' ').strip()
    txt = ' '.join(txt.split())
    return txt

# 텍스트가 COVID 관련인지 체크
def contains_covid_keyword(text: str) -> bool:
    t = text.lower()
    for kw in COVID_KEYWORDS:
        if kw in t:
            return True
    return False

# 텍스트 해시 생성 (중복 체크용)
def hash_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# 간단한 요청 래퍼
def http_get(url, headers=None, proxies=None, timeout=15):
    h = DEFAULT_HEADERS.copy()
    h['User-Agent'] = random.choice(USER_AGENTS)
    if headers:
        h.update(headers)
    try:
        r = requests.get(url, headers=h, proxies=proxies, timeout=timeout)
        r.raise_for_status()
        return r
    except Exception as e:
        logging.debug(f"HTTP GET failed: {url} -> {e}")
        return None

# 페이지 내 텍스트 후보 추출 휴리스틱
def extract_candidate_texts(soup: BeautifulSoup):
    # 우선적으로 댓글/리뷰/게시글을 담을 만한 흔한 셀렉터 시도
    selectors = [
        'div.user-comment', 'div.user-review', 'div.review', 'div.post-content', 'div.post',
        'div.message-body', 'div.msg', 'article', 'div.comment', 'div.entry', 'div#content'
    ]

    texts = []
    for sel in selectors:
        found = soup.select(sel)
        for f in found:
            txt = clean_text(f.get_text(" ", strip=True))
            if len(txt) > 30:
                texts.append(txt)
    # fallback: 본문 내 <p> 태그 모으기
    if not texts:
        paragraphs = soup.find_all('p')
        for p in paragraphs:
            txt = clean_text(p.get_text(" ", strip=True))
            if len(txt) > 60:
                texts.append(txt)
    # 마지막으로 body 텍스트 (매우 큰 경우 제외)
    if not texts:
        body = soup.body
        if body:
            txt = clean_text(body.get_text(" ", strip=True))
            if 80 < len(txt) < 5000:
                texts.append(txt)
    return texts

# CSV 쓰기 헬퍼 (헤더가 없으면 생성)
def append_to_csv(path, rows, header=None):
    exists = os.path.exists(path)
    with open(path, 'a', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        if not exists and header:
            writer.writerow(header)
        for r in rows:
            writer.writerow(r)

# 체크포인트 로드/저장
def load_checkpoint(path):
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_checkpoint(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# --------------------------- 사이트별 크롤러 (유연한 접근) ---------------------------

class UnifiedCrawler:
    def __init__(self, out_csv, checkpoint_path, proxies=None, min_words=8, verbose=True):
        self.out_csv = out_csv
        self.checkpoint_path = checkpoint_path
        self.proxies = proxies
        self.min_words = min_words
        self.seen_hashes = set()
        self.checkpoint = load_checkpoint(checkpoint_path)
        self.verbose = verbose
        # CSV 헤더
        self.csv_header = ['source', 'url', 'text']
        # 기존 CSV에서 중복 해시 로드
        if os.path.exists(out_csv):
            try:
                with open(out_csv, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) >= 3:
                            h = hash_text(','.join(parts[1:3]))
                            self.seen_hashes.add(h)
            except Exception:
                pass

    def save_texts(self, source, url, texts):
        rows = []
        new_count = 0
        for t in texts:
            if len(t.split()) < self.min_words:
                continue
            if not contains_covid_keyword(t):
                continue
            h = hash_text(t)
            if h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)
            rows.append([source, url, t])
            new_count += 1
        if rows:
            append_to_csv(self.out_csv, rows, header=self.csv_header)
        if self.verbose:
            logging.info(f"Saved {new_count} new items from {url}")
        return new_count

    # ------------------ WebMD 크롤러 ------------------
    # WebMD는 리뷰/기사 페이지를 페이지 인덱스 방식으로 순회 가능 (사이트 구조에 따라 수정 필요)
    def crawl_webmd_search(self, query, max_pages=200, delay=(1.0, 2.0), start_page=1):
        """
        WebMD 검색 결과 페이지를 순회해서 개별 항목들을 방문, 텍스트 추출
        - query: 검색어 (예: "covid vaccine")
        - site 구조가 변경될 경우 CSS 선택자 조정 필요
        """
        logging.info(f"Starting WebMD crawl for query={query}, max_pages={max_pages}")
        base_search = f"https://www.webmd.com/search/search_results/default.aspx?query={requests.utils.quote(query)}"

        # 매우 간단한 페이징 루프 (WebMD의 실제 페이징 파라미터는 다를 수 있음)
        for page in range(start_page, max_pages + 1):
            url = base_search + f"&page={page}"
            r = http_get(url, proxies=self.proxies)
            if not r:
                logging.debug(f"Failed to fetch search page {page}")
                time.sleep(random.uniform(*delay))
                continue
            soup = BeautifulSoup(r.text, 'html.parser')
            # 검색 결과에서 결과 링크 추출 (다양한 구조를 고려하여 href가 있는 태그부터 수집)
            links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if 'webmd.com' in href:
                    links.add(href)
                elif href.startswith('/'):
                    links.add(urljoin('https://www.webmd.com', href))
            if not links:
                logging.debug(f"No links found on WebMD search page {page}; breaking")
                break

            for link in links:
                # 간단한 필터: 검색어 포함 URL 우선
                try:
                    self.process_page('webmd', link)
                except Exception as e:
                    logging.debug(f"Error processing WebMD link {link}: {e}")
            # checkpoint
            self.checkpoint['webmd_last_page'] = page
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

    def crawl_webmd_manual_paging(self, base_url, start_page=1, max_pages=200, page_param='pageIndex', delay=(1.0,2.0)):
        """
        특정 WebMD 리뷰/목록 URL에 대해 page 인덱스를 증가시키며 크롤링
        예: https://www.webmd.com/drugs/drugreview-4034-birth-control?pageIndex=1
        """
        for p in range(start_page, max_pages+1):
            url = f"{base_url}?{page_param}={p}"
            r = http_get(url, proxies=self.proxies)
            if not r:
                continue
            try:
                self.process_page('webmd', url)
            except Exception as e:
                logging.debug(f"Error in manual paging {url}: {e}")
            self.checkpoint['webmd_manual_last'] = p
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

    # ------------------ HealthBoards 크롤러 ------------------
    def crawl_healthboards_forum(self, forum_base_url, max_pages=500, start_page=1, delay=(1.0,2.0)):
        """
        HealthBoards 포럼(스레드 목록) 페이지를 순회하고 각 스레드를 방문하여 본문/댓글 추출
        forum_base_url 예: 'https://www.healthboards.com/boards/coronavirus/'
        """
        logging.info(f"Starting HealthBoards crawl: {forum_base_url}")
        for p in range(start_page, max_pages+1):
            page_url = forum_base_url.rstrip('/') + f'/page-{p}'
            r = http_get(page_url, proxies=self.proxies)
            if not r:
                logging.debug(f"Failed to fetch HealthBoards page {p}")
                continue
            soup = BeautifulSoup(r.text, 'html.parser')
            # 스레드 링크 추출 (다양한 구조 대응)
            thread_links = set()
            for a in soup.find_all('a', href=True):
                href = a['href']
                if '/threads/' in href or '/topic/' in href:
                    if href.startswith('http'):
                        thread_links.add(href)
                    else:
                        thread_links.add(urljoin(page_url, href))
            if not thread_links:
                # 대안: 페이지 내 모든 상대 링크 중 포럼 도메인 링크 수집
                for a in soup.find_all('a', href=True):
                    href = a['href']
                    if href.startswith('/'):
                        thread_links.add(urljoin(page_url, href))
            for tlink in thread_links:
                try:
                    self.process_page('healthboards', tlink)
                except Exception as e:
                    logging.debug(f"Error processing HealthBoards thread {tlink}: {e}")
            self.checkpoint['healthboards_last_page'] = p
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

    # ------------------ 공용: 페이지 처리 ------------------
    def process_page(self, source, url):
        r = http_get(url, proxies=self.proxies)
        if not r:
            return 0
        soup = BeautifulSoup(r.text, 'html.parser')
        texts = extract_candidate_texts(soup)
        saved = self.save_texts(source, url, texts)
        return saved

# --------------------------- 메인 스크립트 ---------------------------

def main():
    parser = argparse.ArgumentParser(description='WebMD + HealthBoards unified crawler (COVID-19 target)')
    parser.add_argument('--out', type=str, default='covid_vaccine_reviews.csv', help='Output CSV path')
    parser.add_argument('--checkpoint', type=str, default='crawler_checkpoint.json', help='Checkpoint JSON path')
    parser.add_argument('--max-pages', type=int, default=1000, help='Max pages per site (heuristic)')
    parser.add_argument('--site', choices=['both','webmd','healthboards'], default='both', help='Which site to crawl')
    parser.add_argument('--webmd-query', type=str, default='covid vaccine', help='WebMD search query')
    parser.add_argument('--healthboards-base', type=str, default='https://www.healthboards.com/boards/coronavirus/', help='HealthBoards forum base URL')
    parser.add_argument('--delay-min', type=float, default=1.0)
    parser.add_argument('--delay-max', type=float, default=2.0)
    parser.add_argument('--proxy', type=str, default=None, help='Optional proxy (http://user:pass@host:port)')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    proxies = None
    if args.proxy:
        proxies = {
            'http': args.proxy,
            'https': args.proxy
        }

    crawler = UnifiedCrawler(out_csv=args.out, checkpoint_path=args.checkpoint, proxies=proxies, verbose=args.verbose)

    # 중단점 로드
    cp = load_checkpoint(args.checkpoint)

    if args.site in ('both','webmd'):
        webmd_start = cp.get('webmd_last_page', 1)
        crawler.crawl_webmd_search(query=args.webmd_query, max_pages=args.max_pages, delay=(args.delay_min, args.delay_max), start_page=webmd_start)

    if args.site in ('both','healthboards'):
        hb_start = cp.get('healthboards_last_page', 1)
        crawler.crawl_healthboards_forum(forum_base_url=args.healthboards_base, max_pages=args.max_pages, start_page=hb_start, delay=(args.delay_min, args.delay_max))

    logging.info('Crawling finished')

if __name__ == '__main__':
    main()
