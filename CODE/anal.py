import pandas as pd

OUTPUT_PREDICTED_FILE = "FINAL_ANALYSIS_DATA_with_Sentiment.csv"

try:
    df_final = pd.read_csv(OUTPUT_PREDICTED_FILE)

    # 1. 감정 분포 계산
    sentiment_counts = df_final['Predicted_Sentiment'].value_counts()
    sentiment_ratio = df_final['Predicted_Sentiment'].value_counts(normalize=True) * 100

    print(f"--- 분석 결과: 총 {len(df_final)}건 ---")
    print(f"BERT 모델 Accuracy: (로그 파일에서 확인 필요)")
    print("\n--- 1. 전체 감정 분포 ---")
    print(pd.DataFrame({'Count': sentiment_counts, 'Ratio (%)': sentiment_ratio.round(2)}).to_markdown())

    # 2. 상위 토픽 추출을 위한 준비
    print("\n[NEXT STEP] 감정별 토픽 모델링을 위해 데이터가 준비되었습니다.")

except FileNotFoundError:
    print(f"[ERROR] 최종 예측 파일 ('{OUTPUT_PREDICTED_FILE}')을 찾을 수 없습니다. BERT 학습이 완료되었는지 확인해 주세요.")
except Exception as e:
    print(f"[ERROR] 분석 중 오류 발생: {e}")