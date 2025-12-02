import pandas as pd
import re

# 1️⃣ CSV 불러오기
df = pd.read_csv("FINAL_data_CLEANED_CHECK_2.csv")

# 2️⃣ [No Content], NaN, 공백만 있는 행 제거
df = df.dropna(subset=["text"])
df = df[~df["text"].astype(str).str.strip().isin(["", "[No Content]", "[no content]", "No Content"])]

# 3️⃣ 특수문자만 있는 잡음 제거
def is_meaningless(text):
    # 알파벳·숫자·한글이 전체의 30% 미만이면 잡음으로 간주
    text = str(text)
    alpha_ratio = len(re.findall(r"[A-Za-z가-힣0-9]", text)) / max(len(text), 1)
    return alpha_ratio < 0.3

df = df[~df["text"].apply(is_meaningless)]

# 4️⃣ 너무 짧은 문장 제거 (단어 수 3 미만)
df = df[df["text"].apply(lambda x: len(str(x).split()) >= 3)]

# 5️⃣ 중복 텍스트 제거
df = df.drop_duplicates(subset=["text"])

# 6️⃣ 키워드 태그 생성 (삭제 X, 단지 분류)
COVID_KEYWORDS = [
    "covid", "vaccine", "vaccinated", "corona", "pandemic",
    "booster", "pfizer", "moderna", "astrazeneca", "side effect",
    "reaction", "symptom", "mrna"
]

def has_covid_keyword(text):
    text = str(text).lower()
    return any(kw in text for kw in COVID_KEYWORDS)

df["related_to_vaccine"] = df["text"].apply(has_covid_keyword)

# 7️⃣ 정제된 CSV 저장
df.to_csv("FINAL_data_CLEANED_FILTERED.csv", index=False, encoding="utf-8-sig")

print(f"✅ 정제 완료! 남은 데이터 개수: {len(df)}")
print(df["related_to_vaccine"].value_counts())
