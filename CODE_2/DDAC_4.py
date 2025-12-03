import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ========================
# 0. 기본 설정
# ========================
BASE_DIR = Path(__file__).resolve().parent  # 이 파일이 있는 폴더
CSV_PATH = BASE_DIR / "DDDD.csv"

# 윈도우에서 한글 깨짐 방지 (맑은 고딕 기준)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ========================
# 1. 데이터 로드
# ========================
print(f"[INFO] Load CSV: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)

if "created_at" not in df.columns:
    raise ValueError("'created_at' 컬럼이 없습니다. DDDD.csv 컬럼명을 확인해 주세요.")
if "text" not in df.columns:
    raise ValueError("'text' 컬럼이 없습니다. DDDD.csv 컬럼명을 확인해 주세요.")

# 날짜 파싱
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.dropna(subset=["created_at"]).copy()
print(f"[INFO] After datetime parsing: {len(df)} rows")

# ========================
# 2. 부정 데이터만 사용할지 선택
# ========================
if "pred_label" in df.columns:
    # pred_label == 0 이 부정이라 가정
    df_neg = df[df["pred_label"] == 0].copy()
    print(f"[INFO] Negative rows only: {len(df_neg)}")
else:
    df_neg = df.copy()
    print("[WARN] 'pred_label' 컬럼이 없어 전체 데이터를 사용합니다.")
    print(f"[INFO] Rows used: {len(df_neg)}")

# 텍스트 정리
df_neg["text"] = df_neg["text"].fillna("").astype(str)


# ========================
# 3. 부작용 관련 키워드 플래그 생성
# ========================
SIDE_KEYWORDS = [
    "side effect", "side effects", "adverse", "adverse event", "adverse events",
    "reaction", "reactions",
    "symptom", "symptoms",
    "long covid", "long-covid",
    "myocarditis",
    "blood clot", "blood clots", "clotting",
    "died after", "died from",
    "vaccine injury", "injury from the vaccine"
]

def contains_side_kw(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SIDE_KEYWORDS)

print("[INFO] Detecting side-effect keywords...")
df_neg["has_side_kw"] = df_neg["text"].apply(contains_side_kw)

side_count = df_neg["has_side_kw"].sum()
print(f"[INFO] 총 부작용 키워드 언급 댓글 수: {side_count} / {len(df_neg)} "
      f"({side_count / len(df_neg):.3f})")


# ========================
# 4. 월별 집계 (부작용 vs 비-부작용)
# ========================
df_neg["month"] = df_neg["created_at"].dt.to_period("M").dt.to_timestamp()

monthly = (
    df_neg
    .groupby("month")
    .agg(
        n_total=("has_side_kw", "size"),
        n_side=("has_side_kw", "sum"),
    )
    .reset_index()
)

monthly["side_ratio"] = monthly["n_side"] / monthly["n_total"]
monthly["non_side_ratio"] = 1.0 - monthly["side_ratio"]

print("\n===== [월별 부작용 vs 비-부작용 비율 head] =====")
print(monthly.head())

# CSV로도 저장 (보고서용)
OUT_CSV = BASE_DIR / "sideeffect_trend_monthly.csv"
monthly.to_csv(OUT_CSV, index=False)
print(f"[INFO] Saved monthly trend CSV: {OUT_CSV}")

# ========================
# 5. 시계열 그래프
# ========================
plt.figure(figsize=(12, 6))
plt.plot(monthly["month"], monthly["side_ratio"],
         marker="o", label="부작용 이슈 비율")
plt.plot(monthly["month"], monthly["non_side_ratio"],
         marker="o", label="부작용 이슈 이외의 비율")

plt.title("월별 부정 댓글 내 부작용 언급 비율 추이")
plt.xlabel("월")
plt.ylabel("비율")
plt.grid(True)
plt.legend()
plt.tight_layout()

OUT_PNG = BASE_DIR / "sideeffect_vs_others_monthly_keywords.png"
plt.savefig(OUT_PNG, dpi=200)
plt.show()

print(f"[INFO] Saved figure: {OUT_PNG}")
