from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
import pandas as pd
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException

# 1. 드라이버 초기화
try:
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(service=service, options=options)
except Exception as e:
    print(f"[ERROR] 드라이버 초기화 오류: {e}. 수동 드라이버 설치가 필요할 수 있습니다.")
    exit()

# 2. 크롤링 설정
base_url = "https://gall.dcinside.com/mgallery/board/lists/?id=covidvaccine"
total_pages = 20
safety_delay = 3
result = []

print(f"--- DC Inside '코로나 백신 갤러리' 크롤링 시작 ---")

for page in range(1, total_pages + 1):
    print(f"\n[INFO] {page}페이지 크롤링 중...")
    page_url = base_url.format(page)
    driver.get(page_url)
    time.sleep(safety_delay)

    # 게시글 목록 추출
    try:
        post_link_elements = driver.find_elements(By.CSS_SELECTOR, "td.gall_tit > a")
    except NoSuchElementException:
        print("[WARN] 게시글 링크를 찾을 수 없습니다. 다음 페이지로 넘어갑니다.")
        continue

    if not post_link_elements:
        print("[STOP] 더 이상 게시글을 찾을 수 없습니다. 크롤링을 종료합니다.")
        break

    # 게시글 URL 목록화 (공지, 광고 제외)
    post_urls = []
    for link_element in post_link_elements:
        try:
            url = link_element.get_attribute("href")
            link_class = link_element.get_attribute("class")

            if not url or "view" not in url or "id=covid" not in url:
                continue
            if link_class and "notice" in link_class:
                continue
            if "noimg" in url or "ad" in url:
                continue

            post_urls.append(url)
        except StaleElementReferenceException:
            continue

    # 각 게시글 본문 크롤링
    for url in post_urls:
        try:
            driver.execute_script("window.open(arguments[0]);", url)
            driver.switch_to.window(driver.window_handles[-1])

            # 본문 로딩 대기
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.writing_view_box"))
            )

            # 제목
            try:
                title_elem = driver.find_element(By.CSS_SELECTOR, "span.title_subject")
                title_text = title_elem.text.strip()
            except NoSuchElementException:
                title_text = "(제목 없음)"

            # 본문 (fallback 처리)
            try:
                content_elem = driver.find_element(By.CSS_SELECTOR, "div.write_div")
            except NoSuchElementException:
                content_elem = driver.find_element(By.CSS_SELECTOR, "div.writing_view_box")
            content_text = content_elem.text.strip()

            # 결과 저장
            result.append({
                "topic": "코로나 백신",
                "url": url,
                "title": title_text,
                "content": content_text,
            })

            # 탭 닫기
            driver.close()
            driver.switch_to.window(driver.window_handles[0])

        except TimeoutException:
            print(f"[TIMEOUT] {url} 로딩 실패 - 스킵합니다.")
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            continue
        except Exception as e:
            print(f"[ERROR] 게시글 수집 중 오류 발생: {e}")
            if len(driver.window_handles) > 1:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            continue

        time.sleep(1)  # 게시글 간 대기

# 3. CSV로 저장
df = pd.DataFrame(result)
df.to_csv("dc_inside_covid_vaccine_posts.csv", index=False, encoding="utf-8-sig")

print(f"\n[INFO] 크롤링 완료 ✅ 총 {len(df)}건 수집됨")
driver.quit()
