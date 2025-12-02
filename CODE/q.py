import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ------------------- 1. 모델 성능 지표 (수동 입력 필요) -------------------
# BERT 모델 학습 결과에서 얻은 실제 값으로 대체해야 합니다.
ACCURACY = 0.8204
F1_SCORE = 0.6438
PRECISION = 0.7011 # 가상 값 (보고서에 필요)
RECALL = 0.5985 # 가상 값 (보고서에 필요)

metrics_data = pd.DataFrame({
    'Metric': ['Accuracy (전체 정확도)', 'F1 Score (균형 점수)', 'Precision (정밀도)', 'Recall (재현율)'],
    'Score': [ACCURACY, F1_SCORE, PRECISION, RECALL],
    'Type': ['General', 'Binary', 'Binary', 'Binary']
})

# ------------------- 2. 도표 생성 (막대 그래프) -------------------
plt.figure(figsize=(9, 6))

# Accuracy는 별도로 강조하고, F1/Precision/Recall만 바 그래프로 비교
bar_metrics = metrics_data[metrics_data['Type'] == 'Binary']
sns.barplot(x='Metric', y='Score', data=bar_metrics, palette=['#E34A33', '#4393C3', '#74C476'])

# 값 표시
for index, row in bar_metrics.iterrows():
    plt.text(index, row['Score'] + 0.01, f'{row["Score"]:.3f}', color='black', ha="center")

plt.axhline(F1_SCORE, color='red', linestyle='--', linewidth=1, alpha=0.7) # F1 Score 선 강조

plt.title('1. BERT 모델 성능 지표 비교 (F1-Score)', fontsize=16, pad=15)
plt.ylabel('점수', fontsize=12)
plt.ylim(0.5, 1.0) # 0.5~1.0 사이로 y축 제한
plt.tight_layout()

plt.savefig("bert_performance_metrics.png")
plt.close()

print("✅ 시각화 파일 저장 완료: 'bert_performance_metrics.png'")