import pandas as pd
from collections import Counter
import re
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm # 폰트 관리 라이브러리 추가

# --- 1. Matplotlib 한글 폰트 설정 ---
# Windows 환경에서 가장 범용적인 'Malgun Gothic' 폰트를 설정합니다.
try:
    plt.rc('font', family='Malgun Gothic')
    plt.rcParams['axes.unicode_minus'] = False # 마이너스 폰트 깨짐 방지
    print("[INFO] Matplotlib: 'Malgun Gothic' 폰트 설정 완료.")
except:
    # 폰트가 없을 경우 기본 폰트를 사용하며 경고를 출력합니다.
    print("[WARNING] Malgun Gothic 폰트를 찾을 수 없습니다. 기본 폰트로 출력됩니다.")
    plt.rcParams['axes.unicode_minus'] = False


# ------------------- 2. 데이터 준비 (이전 분석 결과 재활용) -------------------
# 이 부분은 이전 실행에서 계산된 최종 키워드 빈도 테이블입니다.
data = {
    'Keyword': ['mask', 'work', 'right', 'virus', 'masks', 'com', 'wear', 'shit', 'long', 'feel', 'money', 'wearing', 'reddit', 'message', 'youtube', 'adverse', 'side effect'],
    'Total Mentions': [5237, 4913, 4333, 4116, 3805, 3369, 2924, 2798, 2373, 2362, 2305, 2059, 1812, 904, 395, 248, 158]
}
df_counts = pd.DataFrame(data)


# 3. 도표 생성 (수평 막대 그래프)
plt.figure(figsize=(12, 8))
# 횟수가 높은 순으로 정렬하여 플로팅
sns.barplot(x='Total Mentions', y='Keyword', data=df_counts.sort_values(by='Total Mentions', ascending=True), color='#E34A33')

# 그래프 제목과 축 라벨을 한글로 다시 설정
plt.title('부정 여론의 핵심 논란 키워드 총 언급 횟수', fontsize=16, pad=15)
plt.xlabel('총 언급 횟수', fontsize=12)
plt.ylabel('핵심 키워드', fontsize=12)
plt.tight_layout()

plt.savefig("controversy_keyword_mentions_final.png")
plt.close()

print("\n✅ 최종 시각화 파일 저장 완료: 'controversy_keyword_mentions_final.png'")