from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import requests, json, time, pandas as pd
from urllib.parse import quote
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# ----------------------------
# 1. 브라우저 설정 및 드라이버 초기화
# ----------------------------
chrome_options = Options()
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument(
    "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")

try:
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
except Exception as e:
    print(f"[ERROR] 드라이버 초기화 오류: {e}. 라이브러리 업데이트가 필요합니다.")
    exit()

# ----------------------------
# 2. 검색어 및 페이지 설정
# ----------------------------
query = "코로나 백신"
max_pages = 5
news_urls = []
url_set = set()

print(f"[INFO] '{query}' 관련 네이버 뉴스 수집 시작...")

# ----------------------------
# 3. 뉴스 링크 수집 (BeautifulSoup 제거, 순수 Selenium 사용)
# ----------------------------
for page in range(1, max_pages + 1):
    start = (page - 1) * 10 + 1
    query_encoded = quote(query)
    url = (
        f"https://search.naver.com/search.naver?where=news&query={query_encoded}"
        f"&sm=tab_pge&sort=0&photo=0&field=0&pd=3&ds=2020.12.01&de=2022.12.31&mynews=0&office_type=0"
        f"&start={start}"
    )

    driver.get(url)
    time.sleep(3)

    # 순수 Selenium: HTML 소스 대신 요소 자체를 찾습니다.
    # a.news_tit이 확실하므로, 이 요소를 직접 찾습니다.
    try:
        links = driver.find_elements(By.CSS_SELECTOR, "a.news_tit")
    except:
        links = []

    for link in links:
        href = link.get_attribute("href")  # Selenium에서 바로 href 속성 추출
        # 네이버 뉴스 기사 링크만 필터링하고 중복 방지
        if href and "news.naver.com" in href and href not in url_set:
            news_urls.append(href)
            url_set.add(href)

    print(f"  > {page}페이지 완료 ({len(news_urls)}개 누적)")
    time.sleep(1)

driver.quit()

# ----------------------------
# 4. CSV 저장 및 출력 (링크 수집 성공 시 다음 단계인 댓글 크롤링으로 넘어갑니다.)
# ----------------------------
print(f"\n[INFO] 총 {len(news_urls)}개의 뉴스 기사 URL 수집 완료.")

# 링크 수집에 성공했을 때만 다음 댓글 크롤링 단계를 진행합니다.
if len(news_urls) > 0:
    # 5. 댓글 API 기반 댓글 크롤링 (이 부분은 다음 단계에서 진행)
    # ...

    df = pd.DataFrame(news_urls, columns=['url'])
    output = "naver_vaccine_urls_final_success.csv"
    df.to_csv(output, index=False, encoding="utf-8-sig")
    print(f"💾 링크 수집 성공. CSV 저장 완료: {output} (링크 개수: {len(df)})")
else:
    print("🚨 링크 수집 실패. 다음 단계로 진행 불가.")