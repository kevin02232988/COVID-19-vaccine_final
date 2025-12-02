import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------
# 1. 데이터 로드
# ---------------------------------------
df = pd.read_csv("DDDD.csv")

# ---------------------------------------
# 2. 긍정/부정 개수 계산
# ---------------------------------------
positive_count = (df['sentiment_label'] == 1).sum()
negative_count = (df['sentiment_label'] == 0).sum()

labels = ['부정', '긍정']
counts = [negative_count, positive_count]
colors = ['red', 'green']

# ---------------------------------------
# 3. 막대 그래프 생성 (사진 스타일)
# ---------------------------------------
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, counts, color=colors)

# 수치 라벨 표시
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height + 20,
        f"{height}",
        ha='center',
        fontsize=12
    )

plt.title("Binary Label Distribution", fontsize=16)
plt.xlabel("Sentiment (부정=0, 긍정=1)", fontsize=13)
plt.ylabel("Count", fontsize=13)

plt.tight_layout()
plt.show()
