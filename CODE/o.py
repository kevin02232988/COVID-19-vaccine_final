import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

# 1. 파일 불러오기
file_path = 'FINAL_ANALYSIS_DATA_with_Sentiment.csv'
df = pd.read_csv(file_path)

# 2. 'related_to_vaccine' 컬럼의 값 개수 계산 및 비율 계산
ratio = df['related_to_vaccine'].value_counts(normalize=True) * 100

# 3. 한글 폰트 설정
title_text = 'Vaccine Related Content Ratio (related_to_vaccine)'
labels_text = ['Not Vaccine Related (False)', 'Vaccine Related (True)']

try:
    # 폰트 리스트에서 'NanumGothic'이 포함된 폰트 파일 경로를 찾습니다.
    nanum_font = next((f for f in font_manager.fontManager.ttflist if 'nanumgothic' in f.name.lower()), None)

    if nanum_font:
        # 찾은 폰트의 이름을 Matplotlib에 설정
        rc('font', family=nanum_font.name)
        # 한글 레이블 사용
        title_text = '백신 관련 콘텐츠 비율 (related_to_vaccine)'
        labels_text = ['백신 관련 없음 (False)', '백신 관련 있음 (True)']
    else:
        # NanumGothic이 없으면 기본 폰트 사용 및 영어 레이블로 전환
        rc('font', family='DejaVu Sans')

except Exception as e:
    # 예외 발생 시 기본 폰트 및 영어 레이블로 전환
    rc('font', family='DejaVu Sans')

# 마이너스 부호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 4. 비율 출력
print("--- 'related_to_vaccine' True/False 비율 ---")
print(ratio.to_string(float_format="%.2f%%"))

# 5. 파이 차트 생성 및 저장
plt.figure(figsize=(8, 8))
ratio.plot(
    kind='pie',
    autopct='%.2f%%',
    startangle=90,
    colors=['skyblue', 'lightcoral'],
    labels=labels_text,
    wedgeprops={'edgecolor': 'black'}
)

plt.title(title_text, fontsize=16)
plt.ylabel('')
plt.savefig('vaccine_ratio_pie_chart_fixed.png')