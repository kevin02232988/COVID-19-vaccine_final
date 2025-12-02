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
from dateutil import parser as dateparser

USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15',
]

DEFAULT_HEADERS = {'Accept-Language': 'en-US,en;q=0.9'}

COVID_KEYWORDS = [
    'covid','covid-19','coronavirus','vaccine','pfizer','moderna','astrazeneca',
    'johnson & johnson','janssen','sputnik','sinovac','side effect','side-effect',
    'side effects','injection','shot','booster','anti-vax','vax'
]

def clean_text(text):
    return ' '.join(text.replace('\r',' ').replace('\n',' ').split())

def contains_covid_keyword(text):
    t = text.lower()
    return any(kw in t for kw in COVID_KEYWORDS)

def hash_text(text):
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
    except Exception as e:
        logging.debug(f"HTTP GET failed: {url} -> {e}")
        return None

def extract_candidate_texts(soup):
    selectors = ['div.user-comment','div.user-review','div.review','div.post-content','div.post',
                 'div.message-body','div.msg','article','div.comment','div.entry','div#content']
    texts = []
    for sel in selectors:
        for f in soup.select(sel):
            txt = clean_text(f.get_text(" ", strip=True))
            if len(txt) > 30:
                texts.append(txt)
    if not texts:
        for p in soup.find_all('p'):
            txt = clean_text(p.get_text(" ", strip=True))
            if len(txt) > 60:
                texts.append(txt)
    if not texts and soup.body:
        txt = clean_text(soup.body.get_text(" ", strip=True))
        if 80 < len(txt) < 5000:
            texts.append(txt)
    return texts

def extract_date(soup, source):
    date_tag = None
    if source == 'webmd':
        date_tag = soup.select_one('div.review-date, span.article-date, time')
    elif source == 'healthboards':
        # ✅ HealthBoards용 날짜 추출 패턴 확장
        date_tag = soup.select_one('span.postdate, div.postdetails, time, span.date, div.post-date')
    if date_tag:
        try:
            dt = dateparser.parse(date_tag.get_text(), fuzzy=True)
            return dt.strftime('%Y-%m-%d')
        except:
            return 'unknown'
    return 'unknown'

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

class UnifiedCrawler:
    def __init__(self, out_csv, checkpoint_path, proxies=None, min_words=8, verbose=True):
        self.out_csv = out_csv
        self.checkpoint_path = checkpoint_path
        self.proxies = proxies
        self.min_words = min_words
        self.verbose = verbose
        self.seen_hashes = set()
        self.checkpoint = load_checkpoint(checkpoint_path)
        self.csv_header = ['source','url','date','text']
        if os.path.exists(out_csv):
            with open(out_csv,'r',encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts)>=4:
                        self.seen_hashes.add(hash_text(','.join(parts[1:4])))

    def save_texts(self, source, url, texts, date):
        rows = []
        new_count = 0
        for t in texts:
            if len(t.split()) < self.min_words:
                continue
            if not contains_covid_keyword(t):
                continue
            h = hash_text(','.join([url,date,t]))
            if h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)
            rows.append([source,url,date,t])
            new_count += 1
        if rows:
            append_to_csv(self.out_csv, rows, header=self.csv_header)
        if self.verbose:
            logging.info(f"Saved {new_count} new items from {url}")
        return new_count

    def process_page(self, source, url):
        r = http_get(url, proxies=self.proxies)
        if not r:
            return 0
        soup = BeautifulSoup(r.text,'html.parser')
        texts = extract_candidate_texts(soup)
        date = extract_date(soup, source)
        return self.save_texts(source,url,texts,date)

    def crawl_webmd_search(self, query, max_pages=200, delay=(1.0,2.0), start_page=1):
        base_search = f"https://www.webmd.com/search/search_results/default.aspx?query={requests.utils.quote(query)}"
        for page in tqdm(range(start_page, max_pages+1), desc="WebMD pages"):
            url = base_search + f"&page={page}"
            r = http_get(url, proxies=self.proxies)
            if not r:
                time.sleep(random.uniform(*delay))
                continue
            soup = BeautifulSoup(r.text,'html.parser')
            links = set(a['href'] for a in soup.find_all('a', href=True)
                        if 'webmd.com' in a['href'] or a['href'].startswith('/'))
            logging.info(f"[WebMD] Page {page}: {len(links)} links found")
            for link in tqdm(links, desc=f"WebMD links (page {page})", leave=False):
                self.process_page('webmd', urljoin('https://www.webmd.com', link))
            self.checkpoint['webmd_last_page'] = page
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

    def crawl_healthboards_forum(self, forum_base_url, max_pages=500, start_page=1, delay=(1.0,2.0)):
        for p in tqdm(range(start_page, max_pages+1), desc="HealthBoards pages"):
            page_url = forum_base_url.rstrip('/') + f'/page-{p}'
            r = http_get(page_url, proxies=self.proxies)
            if not r:
                time.sleep(random.uniform(*delay))
                continue
            soup = BeautifulSoup(r.text,'html.parser')
            # ✅ 링크 패턴 보완
            thread_links = set(a['href'] for a in soup.find_all('a', href=True)
                               if ('showthread.php?t=' in a['href']) or ('/threads/' in a['href']))
            logging.info(f"[HealthBoards] Page {p}: {len(thread_links)} threads found")
            for tlink in tqdm(thread_links, desc=f"HB threads (page {p})", leave=False):
                self.process_page('healthboards', urljoin(forum_base_url, tlink))
            self.checkpoint['healthboards_last_page'] = p
            save_checkpoint(self.checkpoint_path, self.checkpoint)
            time.sleep(random.uniform(*delay))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='covid_vaccine_reviews_fixed.csv')
    parser.add_argument('--checkpoint', type=str, default='crawler_checkpoint_fixed.json')
    parser.add_argument('--max-pages', type=int, default=1000)
    parser.add_argument('--site', choices=['both','webmd','healthboards'], default='both')
    parser.add_argument('--webmd-query', type=str, default='covid vaccine')
    parser.add_argument('--healthboards-base', type=str, default='https://www.healthboards.com/boards/coronavirus/')
    parser.add_argument('--delay-min', type=float, default=1.0)
    parser.add_argument('--delay-max', type=float, default=2.0)
    parser.add_argument('--proxy', type=str, default=None)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format='[%(asctime)s] %(levelname)s: %(message)s')

    proxies = {'http': args.proxy,'https': args.proxy} if args.proxy else None
    crawler = UnifiedCrawler(out_csv=args.out, checkpoint_path=args.checkpoint, proxies=proxies, verbose=args.verbose)

    cp = load_checkpoint(args.checkpoint)
    if args.site in ('both','webmd'):
        crawler.crawl_webmd_search(args.webmd_query, max_pages=args.max_pages, delay=(args.delay_min,args.delay_max), start_page=cp.get('webmd_last_page',1))
    if args.site in ('both','healthboards'):
        crawler.crawl_healthboards_forum(args.healthboards_base, max_pages=args.max_pages, start_page=cp.get('healthboards_last_page',1), delay=(args.delay_min,args.delay_max))

    logging.info("Crawling finished")

if __name__ == '__main__':
    main()
