# topic_trend_from_bertopic.py
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path(__file__).resolve().parent
CSV_DOC_TOPIC = BASE / "bertopic_doc_topics.csv"

print(f"[INFO] Load: {CSV_DOC_TOPIC}")
df = pd.read_csv(CSV_DOC_TOPIC)

# 날짜 처리
df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
df = df.dropna(subset=["created_at", "topic_id"]).copy()

# 월 단위
df["month"] = df["created_at"].dt.to_period("M").dt.to_timestamp()

# =========================
# 1. 토픽 → 대분류 매핑
# =========================
# !!! 여기 리스트를 네가 직접 채워야 함 !!!
# bertopic_topic_info.csv 보고, 각 토픽 번호를 묶어줘.
TOPIC_GROUPS = {
    "side_effect":      [0, 5, 12],   # 예시: 부작용·후유증 관련 토픽 번호
    "mandate_policy":   [3, 7],       # 예시: 마스크/백신 의무화, 규정 적용
    "healthcare_cost":  [2, 8],       # 예시: 병원비, 빚, 보험, 의료 시스템
    "politics":         [4, 9],       # 예시: 트럼프, 정부 책임, 정치 공방
    "conspiracy":       [6, 10],      # 예시: hoax, fake, propaganda
}

GROUP_LABELS_KO = {
    "side_effect":     "부작용·후유증",
    "mandate_policy":  "마스크·백신 의무화",
    "healthcare_cost": "의료비·의료 시스템",
    "politics":        "정치·책임 공방",
    "conspiracy":      "음모론·hoax",
    "other":           "기타",
}

def map_group(tid: int) -> str:
    for g, ids in TOPIC_GROUPS.items():
        if tid in ids:
            return g
    return "other"

df["topic_group"] = df["topic_id"].apply(map_group)

# =========================
# 2. 월별 그룹 비율 계산
# =========================
month_total = df.groupby("month")["topic_id"].count().rename("total")

month_group = (
    df.groupby(["month", "topic_group"])["topic_id"]
      .count()
      .rename("count")
      .reset_index()
)

month_group = month_group.merge(month_total.reset_index(), on="month", how="left")
month_group["ratio"] = month_group["count"] / month_group["total"]

print("\n[INFO] 월별 그룹 비율 예시 (head):")
print(month_group.head())

pivot = month_group.pivot_table(
    index="month",
    columns="topic_group",
    values="ratio",
    fill_value=0.0,
)

# =========================
# 3. 시계열 그래프
# =========================
plt.figure(figsize=(14, 6))

plot_order = ["side_effect", "mandate_policy", "healthcare_cost", "politics", "conspiracy"]
for g in plot_order:
    if g in pivot.columns:
        plt.plot(pivot.index, pivot[g], marker="o", label=GROUP_LABELS_KO[g])

plt.xlabel("기간(월)")
plt.ylabel("해당 토픽 그룹 비율")
plt.title("월별 토픽 그룹 비율 변화 (부작용 vs 정책·의무화·의료비·정치 등)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

OUT_CSV = BASE / "topic_group_ratio_by_month_from_bertopic.csv"
month_group.to_csv(OUT_CSV, index=False)
print(f"[INFO] Saved summary: {OUT_CSV}")
