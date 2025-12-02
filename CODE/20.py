from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
import os
import random

# ------------------- 설정 -------------------
# 목표 약물 리뷰 페이지 (타이레놀 성분: Acetaminophen)
DRUG_NAME = "Acetaminophen"
BASE_REVIEW_URL = "https://www.drugs.com/comments/acetaminophen/tylenol-arthritis-pain.html?page={}"
MAX_PAGES = 500  # 500 페이지 목표 (이 숫자는 Guesses)
OUTPUT_FILE = "drugs_com_reviews_cache_final.csv"

# ------------------- Selenium 설정 -------------------
try:
    service = Service(ChromeDriverManager().install())
    # Headless 모드로 조용히 실행
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--headless")
    driver = webdriver.Chrome(service=service, options=chrome_options)
except Exception as e:
    print(f"[ERROR] Selenium 드라이버 초기화 오류: {e}")
    exit()

result = []

# ------------------- 1단계: 구글 캐시 URL 획득 및 크롤링 -------------------
print(f"--- Drugs.com 리뷰 수집 시작 (구글 캐시 우회) ---")

for page in range(1, MAX_PAGES + 1):
    review_page_url = BASE_REVIEW_URL.format(page)

    # 1. 구글 검색: 해당 리뷰 페이지의 캐시 URL을 찾습니다.
    search_query = f"site:drugs.com inurl:{review_page_url}"
    search_url = f"https://www.google.com/search?q={search_query}"

    try:
        driver.get(search_url)
        time.sleep(random.uniform(2, 4))  # 검색 결과 로딩 대기

        # 2. 구글 캐시 링크 추출
        # '캐시' 또는 'Cached' 링크를 포함하는 요소를 찾아야 합니다. (CSS 선택자 변경 가능성 높음)
        cache_link_element = driver.find_elements(By.XPATH, "//a[contains(text(), 'Cached') or contains(text(), '캐시')]")

        if not cache_link_element:
            # 캐시 링크를 찾지 못했거나, 더 이상 검색 결과가 없습니다.
            print(f"[STOP] {page} 페이지에 대한 캐시 링크를 찾지 못했습니다. 크롤링 종료.")
            break

        # 가장 첫 번째 캐시 링크 사용
        cache_url = cache_link_element[0].get_attribute('href')

    except Exception as e:
        print(f"[ERROR] 구글 검색 중 오류 발생: {e}")
        time.sleep(5)
        continue

    # 3. 캐시 URL로 HTTP 요청 (Drugs.com 서버가 아닌 Google 서버에 요청)
    try:
        res = requests.get(cache_url, headers={'User-Agent': random.choice(USER_AGENTS)}, timeout=10)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")

        # 4. 리뷰 및 별점 추출 (Drugs.com 기존 로직 재활용)
        reviews = soup.select("div.ddc-comment")

        if not reviews:
            print(f"[INFO] 캐시 페이지에서 리뷰를 찾지 못했습니다. (URL: {review_page_url})")
            continue

        for review in reviews:
            comment_text_elem = review.select_one("p")
            comment_text = comment_text_elem.get_text(strip=True) if comment_text_elem else ""

            # 별점 (Rating) 추출
            rating_elem = review.select_one("div.ddc-rating-summary span.rating-num")
            rating = int(rating_elem.get_text(strip=True)) if rating_elem else 0

            # 날짜 추출 (Drugs.com 리뷰의 날짜 추출 로직은 복잡하여 일단 생략, 캐시 URL로 유추)

            result.append({
                "drug_name": DRUG_NAME,
                "review_text": comment_text,
                "rating": rating,
                "source_url": review_page_url,
                "date_info": "Inferred from Cache Date"  # 날짜 정보는 캐시 날짜로 추정
            })

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] 캐시 요청 중 오류 발생: {e}. 건너뜁니다.")
        continue

    # 5. 진행 상태 출력 및 지연
    print(f"[SUCCESS] Page {page} processed. Total collected: {len(result)}")
    time.sleep(random.uniform(3, 5))  # 다음 구글 검색 요청 전 대기 (차단 방지)

# ------------------- 6. 최종 저장 -------------------
driver.quit()
df = pd.DataFrame(result)
df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 최종 완료. 총 {len(df)}건의 데이터를 수집했습니다.")
print(f"파일 저장 완료: '{OUTPUT_FILE}'")