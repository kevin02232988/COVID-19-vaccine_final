import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# ================================
# 0️⃣ 한글 폰트 설정
# ================================
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows용 폰트
plt.rcParams['axes.unicode_minus'] = False     # 마이너스 깨짐 방지

# ================================
# 1️⃣ 데이터 로드
# ================================
binary_df = pd.read_csv("BERT_labeled_binary.csv")
three_df = pd.read_csv("BERT_labeled_three.csv")

# ================================
# 2️⃣ 날짜 처리
# ================================
binary_df['created_at'] = pd.to_datetime(binary_df['created_at'], errors='coerce')
three_df['created_at'] = pd.to_datetime(three_df['created_at'], errors='coerce')

binary_df = binary_df.dropna(subset=['created_at'])
three_df = three_df.dropna(subset=['created_at'])

binary_df['month'] = binary_df['created_at'].dt.to_period('M').astype(str)
three_df['month'] = three_df['created_at'].dt.to_period('M').astype(str)

# ================================
# 3️⃣ 월별 감정 비율 계산
# ================================
binary_trend = (
    binary_df.groupby(['month', 'sentiment_binary'])
    .size()
    .unstack(fill_value=0)
)
binary_trend = binary_trend.div(binary_trend.sum(axis=1), axis=0) * 100

three_trend = (
    three_df.groupby(['month', 'sentiment_three'])
    .size()
    .unstack(fill_value=0)
)
three_trend = three_trend.div(three_trend.sum(axis=1), axis=0) * 100

# ================================
# 4️⃣ 그래프 스타일 함수
# ================================
def style_plot(title, xlabel, ylabel):
    plt.title(title, fontsize=14, fontweight='bold', pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(rotation=45)
    plt.tight_layout()

# ================================
# 5️⃣ 이진 분류 (부정 비율)
# ================================
plt.figure(figsize=(14, 6))
if 'negative' in binary_trend.columns:
    plt.plot(binary_trend.index, binary_trend['negative'], color='red', marker='o', linewidth=1.8)
elif '부정' in binary_trend.columns:
    plt.plot(binary_trend.index, binary_trend['부정'], color='red', marker='o', linewidth=1.8)
else:
    plt.plot(binary_trend.index, binary_trend.iloc[:, 0], color='red', marker='o', linewidth=1.8)

style_plot("2. 월별 부정 감정 비율 변화 추이", "기간 (월)", "부정적 여론 비율 (%)")
plt.savefig("trend_binary.png", dpi=300, bbox_inches='tight')  # ✅ 저장
plt.show()

# ================================
# 6️⃣ 삼분류 (긍정/부정/중립)
# ================================
plt.figure(figsize=(14, 6))
for label in three_trend.columns:
    plt.plot(three_trend.index, three_trend[label], marker='o', linewidth=1.6, label=str(label))

style_plot("3. 월별 감정 비율 변화 추이 (삼분류)", "기간 (월)", "감정 비율 (%)")
plt.legend(title="감정 구분", fontsize=10)
plt.savefig("trend_three.png", dpi=300, bbox_inches='tight')  # ✅ 저장
plt.show()

# ================================
# 7️⃣ 결과 요약
# ================================
print("\n[이진 분류 월별 비율 요약]")
print(binary_trend.round(2).reset_index().head(10))

print("\n[삼분류 월별 비율 요약]")
print(three_trend.round(2).reset_index().head(10))

print("\n✅ 그래프 파일 저장 완료:")
print(" - trend_binary.png")
print(" - trend_three.png")
