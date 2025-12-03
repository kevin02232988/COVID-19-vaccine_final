"""
topic_trend_from_bertopic.py

1) BERTopic 토픽 정보에서 자동으로 토픽 그룹(부작용/정책/의료비/정치) 후보를 추출하고
2) DDDD.csv + doc-topic 매핑을 합쳐서
3) 각 토픽 그룹의 시계열 추이(월별, 기간별)를 계산하고
4) CSV + PNG 그래프를 저장하는 스크립트

실행하면 콘솔에:
- 각 토픽 그룹에 자동 할당된 topic_id 리스트
- 각 기간별 비율 요약
이 찍히고,
폴더에:
- topic_group_trend_monthly.csv
- topic_group_trend_period.csv
- topic_group_trend_monthly.png
- topic_group_trend_period.png
이 생성된다.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# 0. 기본 경로 설정
# =========================
BASE_DIR = Path(r"C:\Users\user\PycharmProjects\PythonProject6")

PATH_DDDD = BASE_DIR / "DDDD.csv"
PATH_DOC_TOPICS = BASE_DIR / "bertopic_doc_topics.csv"
PATH_TOPIC_INFO = BASE_DIR / "bertopic_topic_info.csv"

OUT_MONTHLY_CSV = BASE_DIR / "topic_group_trend_monthly.csv"
OUT_PERIOD_CSV = BASE_DIR / "topic_group_trend_period.csv"
OUT_MONTHLY_PNG = BASE_DIR / "topic_group_trend_monthly.png"
OUT_PERIOD_PNG = BASE_DIR / "topic_group_trend_period.png"

# =========================
# 1. 한글 폰트 설정(윈도우 기준)
# =========================
plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 기본 한글 폰트
plt.rcParams["axes.unicode_minus"] = False     # 마이너스 깨짐 방지


# =========================
# 2. 유틸 함수: 토픽 자동 그룹핑
# =========================
def auto_group_topics(topic_info: pd.DataFrame):
    """
    topic_info(DataFrame)에 있는 각 토픽의 Name/Representation을 모두 합쳐서
    부작용 / 정책·의무화 / 의료비 / 정치 토픽 후보를 자동으로 뽑는다.
    """

    df = topic_info.copy()

    # Name + Representation 둘 다 이어 붙여서 검색용 텍스트로 사용
    name_col = df["Name"].astype(str) if "Name" in df.columns else ""
    rep_col = df["Representation"].astype(str) if "Representation" in df.columns else ""
    df["__text__"] = (name_col + " " + rep_col).str.lower()

    # --- 단어 단위 키워드 목록 (필요하면 여기만 수정하면 됨) ---

    # 부작용 / 장기 후유증
    side_effect_keywords = [
        "side", "effect", "effects", "reaction", "reactions", "adverse",
        "symptom", "symptoms", "myocarditis", "clot", "clots",
        "infection", "infections", "long", "died", "death", "risk", "risks"
    ]

    # 마스크·백신 의무화 / 직장·가게 규정
    policy_keywords = [
        "mandate", "policy", "mask", "masks", "store", "shop", "job",
        "work", "employer", "employees", "customer", "customers",
        "required", "requirement", "passport", "pass", "entry", "access"
    ]

    # 의료비·의료 시스템
    medical_cost_keywords = [
        "hospital", "hospitals", "debt", "bill", "bills", "insurance",
        "pay", "pays", "paying", "cost", "costs", "coverage", "medical",
        "nurse", "nurses", "icu", "bed", "beds"
    ]

    # 정치·책임 공방 / 음모론
    politics_keywords = [
        "trump", "biden", "president", "government", "politic", "politics",
        "hoax", "propaganda", "election", "republican", "democrat", "party"
    ]

    def match_topics(keywords):
        mask = df["__text__"].apply(lambda s: any(k in s for k in keywords))
        return df.loc[mask, ["Topic", "Count", "Name", "Representation"]]

    side_df = match_topics(side_effect_keywords)
    policy_df = match_topics(policy_keywords)
    med_df = match_topics(medical_cost_keywords)
    pol_df = match_topics(politics_keywords)

    side_topics = side_df["Topic"].tolist()
    policy_topics = policy_df["Topic"].tolist()
    med_topics = med_df["Topic"].tolist()
    pol_topics = pol_df["Topic"].tolist()

    topic_groups = {
        "side_effect": side_topics,
        "policy_mandate": policy_topics,
        "medical_cost": med_topics,
        "politics": pol_topics,
    }

    print("\n===== [자동 추출된 토픽 그룹 후보] =====")
    print("\n[부작용/장기 후유증 토픽 후보]")
    print(side_df)
    print("\n[정책/마스크·백신 의무화 토픽 후보]")
    print(policy_df)
    print("\n[의료비·의료 시스템 토픽 후보]")
    print(med_df)
    print("\n[정치·책임 공방 토픽 후보]")
    print(pol_df)

    print("\n[TOPIC_GROUPS 딕셔너리 초안]")
    print(topic_groups)

    return topic_groups



# =========================
# 3. 메인 로직
# =========================
def main():
    # ----- 3-1. 파일 로드 -----
    print(f"[INFO] Load DDDD: {PATH_DDDD}")
    df_all = pd.read_csv(PATH_DDDD)

    print(f"[INFO] Load doc-topic mapping: {PATH_DOC_TOPICS}")
    df_doc_topics = pd.read_csv(PATH_DOC_TOPICS)

    print(f"[INFO] Load topic info: {PATH_TOPIC_INFO}")
    df_topic_info = pd.read_csv(PATH_TOPIC_INFO)

    # ----- 3-2. 부정 댓글만 사용 (pred_label이 0인 경우) -----
    # pred_label 컬럼 이름이 다르면 여기만 수정
    if "pred_label" in df_all.columns:
        df_neg = df_all[df_all["pred_label"] == 0].copy()
        print(f"[INFO] Negative rows only: {len(df_neg)}")
    else:
        print("[WARN] 'pred_label' 컬럼이 없어 전체 데이터를 사용합니다.")
        df_neg = df_all.copy()

    # ----- 3-3. created_at을 datetime으로 변환 -----
    if "created_at" not in df_neg.columns:
        raise ValueError("DDDD.csv 안에 'created_at' 컬럼이 필요합니다.")

    df_neg["created_at"] = pd.to_datetime(df_neg["created_at"], errors="coerce")
    df_neg = df_neg.dropna(subset=["created_at"])
    print(f"[INFO] After datetime parsing: {len(df_neg)} rows")

    # ----- 3-4. doc-topic 매핑과 merge -----
    # bertopic_doc_topics.csv 안에는 text / topic_id 가 있다고 가정
    if "text" not in df_neg.columns or "text" not in df_doc_topics.columns:
        raise ValueError("DDDD.csv, bertopic_doc_topics.csv 모두에 'text' 컬럼이 필요합니다.")

    if "topic_id" not in df_doc_topics.columns:
        raise ValueError("bertopic_doc_topics.csv 안에 'topic_id' 컬럼이 필요합니다.")

    # 텍스트 기준 inner join (부정 댓글 중 토픽 붙은 것만 사용)
    df_merged = pd.merge(
        df_neg[["text", "created_at"]],
        df_doc_topics[["text", "topic_id"]],
        on="text",
        how="inner",
    )
    print(f"[INFO] Merged rows (negative + topic_id): {len(df_merged)}")

    # outlier 토픽(-1)은 제외해도 됨
    df_merged = df_merged[df_merged["topic_id"] != -1].copy()
    print(f"[INFO] After removing topic_id = -1: {len(df_merged)} rows")

    # ----- 3-5. 토픽 그룹 자동 추출 -----
    topic_groups_auto = auto_group_topics(df_topic_info)

    # == 필요시 여기서 topic_groups_auto를 손으로 조정해도 됨 ==
    # 예: topic_groups_auto["side_effect"] += [추가하고 싶은 topic_id]
    # 지금은 자동 결과 그대로 사용
    TOPIC_GROUPS = topic_groups_auto

    # ----- 3-6. 월별 집계 -----
    df_merged["month"] = df_merged["created_at"].dt.to_period("M").dt.to_timestamp()

    # 월별 전체 부정 댓글 수
    monthly_total = df_merged.groupby("month").size().rename("n_total").reset_index()

    # 그룹별 월별 카운트/비율 계산
    monthly_df = monthly_total.copy()

    for group_name, topic_ids in TOPIC_GROUPS.items():
        if not topic_ids:
            print(f"[WARN] 그룹 '{group_name}'에는 해당 토픽이 없어 건너뜁니다.")
            monthly_df[f"{group_name}_count"] = 0
            monthly_df[f"{group_name}_ratio"] = 0.0
            continue

        mask_group = df_merged["topic_id"].isin(topic_ids)
        group_counts = (
            df_merged[mask_group]
            .groupby("month")
            .size()
            .rename(f"{group_name}_count")
        )
        monthly_df = monthly_df.merge(
            group_counts.reset_index(),
            on="month",
            how="left",
        )
        monthly_df[f"{group_name}_count"] = (
            monthly_df[f"{group_name}_count"].fillna(0).astype(int)
        )
        monthly_df[f"{group_name}_ratio"] = (
            monthly_df[f"{group_name}_count"] / monthly_df["n_total"]
        )

    print("\n===== [월별 토픽 그룹 비율 head] =====")
    print(monthly_df.head())

    monthly_df.to_csv(OUT_MONTHLY_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved monthly trend CSV: {OUT_MONTHLY_CSV}")

    # ----- 3-7. 월별 시계열 그래프 -----
    plt.figure(figsize=(12, 6))
    for group_name in TOPIC_GROUPS.keys():
        ratio_col = f"{group_name}_ratio"
        if ratio_col in monthly_df.columns:
            plt.plot(
                monthly_df["month"],
                monthly_df[ratio_col],
                marker="o",
                label=group_name,
            )

    plt.title("월별 부정 댓글 내 토픽 그룹 비율 추이")
    plt.xlabel("월")
    plt.ylabel("비율")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(OUT_MONTHLY_PNG, dpi=200)
    print(f"[INFO] Saved monthly trend PNG: {OUT_MONTHLY_PNG}")
    plt.show()

    # ----- 3-8. 기간별(2019–2020 / 2021 / 2022 / 2023–2025) 집계 -----
    def assign_period(dt):
        y = dt.year
        if y <= 2020:
            return "2019–2020"
        elif y == 2021:
            return "2021"
        elif y == 2022:
            return "2022"
        else:
            return "2023–2025"

    df_merged["period"] = df_merged["created_at"].apply(assign_period)

    period_total = df_merged.groupby("period").size().rename("n_total").reset_index()
    period_df = period_total.copy()

    for group_name, topic_ids in TOPIC_GROUPS.items():
        if not topic_ids:
            period_df[f"{group_name}_count"] = 0
            period_df[f"{group_name}_ratio"] = 0.0
            continue

        mask_group = df_merged["topic_id"].isin(topic_ids)
        group_counts = (
            df_merged[mask_group]
            .groupby("period")
            .size()
            .rename(f"{group_name}_count")
        )
        period_df = period_df.merge(
            group_counts.reset_index(), on="period", how="left"
        )
        period_df[f"{group_name}_count"] = (
            period_df[f"{group_name}_count"].fillna(0).astype(int)
        )
        period_df[f"{group_name}_ratio"] = (
            period_df[f"{group_name}_count"] / period_df["n_total"]
        )

    print("\n===== [기간별 토픽 그룹 비율] =====")
    print(period_df)

    period_df.to_csv(OUT_PERIOD_CSV, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved period trend CSV: {OUT_PERIOD_CSV}")

    # ----- 3-9. 기간별 막대 그래프 -----
    periods_order = ["2019–2020", "2021", "2022", "2023–2025"]
    period_df["period"] = pd.Categorical(
        period_df["period"], categories=periods_order, ordered=True
    )
    period_df = period_df.sort_values("period")

    groups = list(TOPIC_GROUPS.keys())
    x = range(len(period_df))

    plt.figure(figsize=(10, 6))
    width = 0.15
    for i, group_name in enumerate(groups):
        ratio_col = f"{group_name}_ratio"
        if ratio_col not in period_df.columns:
            continue
        plt.bar(
            [xx + i * width for xx in x],
            period_df[ratio_col],
            width=width,
            label=group_name,
        )

    plt.xticks(
        [xx + width * (len(groups) - 1) / 2 for xx in x],
        period_df["period"],
    )
    plt.title("기간별(연도 구간) 부정 댓글 내 토픽 그룹 비율")
    plt.xlabel("기간")
    plt.ylabel("비율")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PERIOD_PNG, dpi=200)
    print(f"[INFO] Saved period trend PNG: {OUT_PERIOD_PNG}")
    plt.show()


if __name__ == "__main__":
    main()
