import time
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------
# 1. Chrome 옵션 및 드라이버 설정
# ---------------------------
chrome_options = Options()
chrome_options.add_argument("--headless")  # 브라우저 안 띄우기
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# ChromeDriver 자동 설치 및 실행
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=chrome_options)

# ---------------------------
# 2. 크롤링할 URL
# ---------------------------
BASE_URL = "https://www.healthboards.com/boards/covid-vaccine.1/"

# ---------------------------
# 3. 게시글 링크 수집
# ---------------------------
post_links = []

driver.get(BASE_URL)
time.sleep(3)  # 페이지 로딩 대기

try:
    # 게시글 링크 가져오기
    posts = driver.find_elements(By.CSS_SELECTOR, "a.title")  # HealthBoards 기준
    for post in posts:
        href = post.get_attribute("href")
        if href and "showthread.php" in href:
            post_links.append(href)
except Exception as e:
    print(f"[ERROR] 게시글 링크 수집 실패: {e}")

print(f"총 {len(post_links)}개 게시글 링크 수집됨.")

# ---------------------------
# 4. 댓글 크롤링
# ---------------------------
all_comments = []

for idx, link in enumerate(post_links, 1):
    try:
        driver.get(link)
        time.sleep(2)  # 페이지 로딩 대기

        # 댓글 가져오기 (HealthBoards 구조에 따라 클래스/태그 수정 가능)
        comments = driver.find_elements(By.CSS_SELECTOR, "div.post_message")
        for comment in comments:
            text = comment.text.strip()
            if text:
                all_comments.append({"post_link": link, "comment": text})

        print(f"[INFO] {idx}/{len(post_links)} 게시글 완료, 댓글 수: {len(comments)}")
    except Exception as e:
        print(f"[ERROR] 게시글 크롤링 실패: {link}, {e}")

print(f"총 {len(all_comments)}개 댓글 수집됨.")

# ---------------------------
# 5. CSV 저장
# ---------------------------
CSV_FILE = "healthboards_comments.csv"
keys = ["post_link", "comment"]

try:
    with open(CSV_FILE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_comments)
    print(f"크롤링 완료, CSV 저장 완료: {CSV_FILE}")
except Exception as e:
    print(f"[ERROR] CSV 저장 실패: {e}")

# ---------------------------
# 6. 종료
# ---------------------------
driver.quit()
