import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import random
import os
import json
from tqdm import tqdm
import time
from datetime import datetime

BACKUP_FILE = "covid_vaccine_progress.json"
OUTPUT_FILE = "merged_covid_vaccine_reviews.csv"
BACKUP_CSV = "backup_covid_vaccine_reviews.csv"

# -------------------------
# 1️⃣ 저장 / 복원 기능
# -------------------------
def save_progress(data_list):
    with open(BACKUP_FILE, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False)

def load_progress():
    if os.path.exists(BACKUP_FILE):
        with open(BACKUP_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# -------------------------
# 2️⃣ WebMD 크롤러
# -------------------------
def crawl_webmd(max_pages=200, resume_data=None):
    print("[INFO] WebMD 수집 시작...")
    base_url = "https://www.webmd.com/vaccines/covid-19-vaccine/reviews?page="
    data = resume_data or []

    start_page = len([d for d in data if d["source"] == "webmd"]) // 50 + 1  # Rough resume point

    for page in tqdm(range(start_page, max_pages + 1), desc="WebMD Pages"):
        try:
            url = f"{base_url}{page}"
            res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if res.status_code != 200:
                continue
            soup = BeautifulSoup(res.text, "html.parser")

            reviews = soup.select(".user-comments, .review, .comments")
            for r in reviews:
                text = r.get_text(" ", strip=True)
                if len(text) < 30:
                    continue
                date_tag = soup.find("span", class_="date")
                date = date_tag.get_text(strip=True) if date_tag else None
                data.append({
                    "source": "webmd",
                    "date": date,
                    "content": text
                })
            if len(data) % 200 == 0:
                pd.DataFrame(data).to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")
                save_progress(data)
            time.sleep(0.5)
        except Exception:
            continue
        if len(data) >= 20000:
            break
    return data

# -------------------------
# 3️⃣ HealthBoards 크롤러
# -------------------------
def crawl_healthboards(max_pages=100, resume_data=None):
    print("[INFO] HealthBoards 수집 시작...")
    base_search_url = "https://www.healthboards.com/boards/search_google.php"
    data = resume_data or []

    start_page = len([d for d in data if d["source"] == "healthboards"]) // 10 + 1

    for page in tqdm(range(start_page, max_pages + 1), desc="HealthBoards Pages"):
        try:
            params = {
                "cx": "partner-pub-8247140117206678:125c5bc0u3i",
                "cof": "FORID:11",
                "ie": "UTF-8",
                "q": "covid vaccine",
                "sa": "search",
                "start": (page - 1) * 10
            }
            res = requests.get(base_search_url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if res.status_code != 200:
                continue

            soup = BeautifulSoup(res.text, "html.parser")
            links = [a["href"] for a in soup.select("a") if "boards/" in a.get("href", "")]

            for link in links:
                if not link.startswith("https"):
                    link = "https://www.healthboards.com" + link
                try:
                    post_res = requests.get(link, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                    if post_res.status_code != 200:
                        continue
                    post_soup = BeautifulSoup(post_res.text, "html.parser")

                    posts = post_soup.select("div.page, td, div.post")
                    text = " ".join(p.get_text(" ", strip=True) for p in posts if len(p.get_text(strip=True)) > 30)
                    if not text:
                        continue

                    # 날짜 추출 (예: Posted: 05-14-2022)
                    date_match = re.search(r"Posted:\s*(\d{2}-\d{2}-\d{4})", post_res.text)
                    date = date_match.group(1) if date_match else None

                    data.append({
                        "source": "healthboards",
                        "date": date,
                        "content": text
                    })
                    if len(data) % 200 == 0:
                        pd.DataFrame(data).to_csv(BACKUP_CSV, index=False, encoding="utf-8-sig")
                        save_progress(data)
                    time.sleep(0.5)
                except Exception:
                    continue
            time.sleep(1)
        except Exception:
            continue
        if len(data) >= 20000:
            break
    return data

# -------------------------
# 4️⃣ 통합 + 저장
# -------------------------
def main():
    start_time = datetime.now()
    print(f"🚀 크롤링 시작: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    data = load_progress()
    if data:
        print(f"🔄 이전 진행 데이터 불러옴 ({len(data)}건)")

    data = crawl_webmd(max_pages=200, resume_data=data)
    if len(data) < 20000:
        data = crawl_healthboards(max_pages=100, resume_data=data)

    df = pd.DataFrame(data)
    df.drop_duplicates(subset=["content"], inplace=True)
    df = df[df["content"].str.contains(r"[a-zA-Z]", na=False)]

    if len(df) > 20000:
        df = df.sample(20000, random_state=42)

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    end_time = datetime.now()

    print("\n✅ 크롤링 완료!")
    print(f"🕒 총 소요 시간: {end_time - start_time}")
    print(f"📁 저장 파일: {OUTPUT_FILE}")
    print(f"총 수집 개수: {len(df)}")
    print(df['source'].value_counts())

if __name__ == "__main__":
    main()
