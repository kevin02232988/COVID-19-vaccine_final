import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.font_manager as fm # 폰트 관리 라이브러리 추가

# ------------------- 1. 모델 성능 지표 (실제 학습 결과로 대체해야 함) -------------------
# 사용자님이 확인하신 최종 BERT 학습 결과 로그 값으로 설정됩니다.
ACCURACY = 0.8204
F1_SCORE = 0.6438
PRECISION = 0.7011
RECALL = 0.5985

metrics_data = pd.DataFrame({
    'Metric': ['Accuracy (전체 정확도)', 'F1 Score (균형 점수)', 'Precision (정밀도)', 'Recall (재현율)'],
    'Score': [ACCURACY, F1_SCORE, PRECISION, RECALL],
    'Type': ['General', 'Binary', 'Binary', 'Binary']
})

# ------------------- 2. 도표 생성 (막대 그래프) -------------------

# Matplotlib 한글 폰트 설정 (경고를 피하기 위해 설정은 유지)
try:
    plt.rc('font', family='Malgun Gothic')
except:
    pass
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(9, 6))

bar_metrics = metrics_data[metrics_data['Type'] == 'Binary']

# 경고 해결: hue=x 변수 할당 및 legend=False 설정 (Seaborn 최신 문법)
sns.barplot(
    x='Metric',
    y='Score',
    data=bar_metrics,
    hue='Metric', # 경고 해결을 위해 X 변수를 hue에 할당
    palette=['#E34A33', '#4393C3', '#74C476'], # 각 막대에 색상 지정
    legend=False # 범례는 필요 없으므로 False
)

# 값 표시
for index, row in bar_metrics.iterrows():
    plt.text(index, row['Score'] + 0.01, f'{row["Score"]:.4f}', color='black', ha="center")

# Accuracy 값을 가로선으로 강조
plt.axhline(ACCURACY, color='gray', linestyle='--', linewidth=1, alpha=0.7, label=f'Accuracy ({ACCURACY:.4f})')

plt.title('1. BERT 모델 성능 지표 비교 (F1-Score)', fontsize=16, pad=15)
plt.ylabel('점수', fontsize=12)
plt.ylim(0.5, 1.0)
plt.legend(loc='lower left')
plt.tight_layout()

plt.savefig("bert_performance_metrics_final_fixed.png")
plt.close()

print("✅ 시각화 파일 저장 완료: 'bert_performance_metrics_final_fixed.png'")