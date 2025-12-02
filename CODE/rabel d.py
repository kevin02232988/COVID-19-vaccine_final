import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 0️⃣ 한글 폰트 설정 (Windows 기준)
font_path = "C:/Windows/Fonts/malgun.ttf"  # 맑은 고딕
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)

# 마이너스 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 1️⃣ CSV 파일 불러오기
df_binary = pd.read_csv("BERT_labeled_binary.csv")
df_three = pd.read_csv("BERT_labeled_three.csv")

# 2️⃣ 비율 계산
binary_counts = df_binary['sentiment_binary'].value_counts(normalize=True) * 100
three_counts = df_three['sentiment_three'].value_counts(normalize=True) * 100

print("✅ [이진 분류 결과 비율]")
print(binary_counts.round(2))
print("\n✅ [삼분류 결과 비율]")
print(three_counts.round(2))

# 3️⃣ 시각화
plt.figure(figsize=(12,5))

# --- (1) 이진 분류 그래프 ---
plt.subplot(1,2,1)
binary_counts.plot(kind='bar', color=['tomato','skyblue'])
plt.title("BERT 감정 분석 (이진 분류: 긍정 / 부정)", fontsize=13)
plt.ylabel("비율 (%)")
plt.xticks(rotation=0)
for i, v in enumerate(binary_counts):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=11)

# --- (2) 삼분류 그래프 ---
plt.subplot(1,2,2)
three_counts.plot(kind='bar', color=['tomato','gold','skyblue'])
plt.title("BERT 감정 분석 (삼분류: 긍정 / 중립 / 부정)", fontsize=13)
plt.ylabel("비율 (%)")
plt.xticks(rotation=0)
for i, v in enumerate(three_counts):
    plt.text(i, v + 1, f"{v:.1f}%", ha='center', fontsize=11)

plt.tight_layout()

# 4️⃣ 그래프 저장
plt.savefig("sentiment_distribution.png", dpi=300)
plt.show()

print("\n📊 그래프 저장 완료: sentiment_distribution.png")
