import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams

# ✅ 한글 폰트 설정 (Windows용)
rcParams['font.family'] = 'Malgun Gothic'  # 맑은 고딕
rcParams['axes.unicode_minus'] = False     # 마이너스 기호 깨짐 방지

# 이미지 불러오기
img_neg = mpimg.imread(r"C:\Users\user\PycharmProjects\PythonProject6\monthly_negative_ratio_trend_final.png")
img_vix = mpimg.imread(r"C:\Users\user\PycharmProjects\PythonProject6\fear.png")

# 그래프 크기 설정
plt.figure(figsize=(14, 6))
plt.title("공포지수와 부정 경향의 비교", fontsize=16, fontweight='bold')

# 두 이미지를 반투명하게 겹쳐 표시
plt.imshow(img_neg, alpha=0.6)  # 부정 감정 그래프
plt.imshow(img_vix, alpha=0.6)  # VIX 그래프

# 색상 대비용 선 (시각 강조)
plt.axhline(y=0, color='red', alpha=0.3)
plt.axhline(y=0, color='blue', alpha=0.3)

# 축 숨기기
plt.axis('off')

# 여백 정리 및 저장
plt.tight_layout()
plt.savefig(r"C:\Users\user\PycharmProjects\PythonProject6\comparison_final.png", dpi=300)
plt.show()

print("✅ 'comparison_final.png' 생성 완료 (한글 깨짐 해결됨)")
