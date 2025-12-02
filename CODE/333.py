import pandas as pd
import re
import os

# --- 설정 ---
# ⚠️ 파일 이름: 현재 폴더에 있는 원본 CSV 파일 이름으로 설정하세요.
INPUT_FILE = 'FINAL_DATA_FILTERED_TRUE.csv'
# 💾 출력 파일 이름: 클리닝된 결과가 저장될 파일 이름입니다.
OUTPUT_FILE = 'FINAL_DATA_ROWS_DELETED.csv'
# 🔗 링크를 감지할 정규 표현식: http 또는 https로 시작하는 모든 URL을 감지합니다.
URL_PATTERN = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def delete_rows_with_links(input_file: str, output_file: str, url_pattern: str):
    """
    지정된 입력 파일에서 'text' 열에 URL이 포함된 행 전체를 삭제하고 새 파일로 저장합니다.
    """
    if not os.path.exists(input_file):
        print(f"오류: 입력 파일 '{input_file}'을(를) 찾을 수 없습니다. 파일 경로를 확인해 주세요.")
        return

    try:
        # 1. 파일 불러오기
        df = pd.read_csv(input_file)
        print(f"✅ 원본 데이터 ({len(df)} 행) 불러오기 완료.")
    except Exception as e:
        print(f"오류: 파일 로드 중 문제가 발생했습니다: {e}")
        return

    # 2. 클리닝 (URL 포함 행 삭제)
    # df['text'].astype(str): 'text' 열을 문자열 타입으로 변환 (오류 방지)
    # .str.contains(url_pattern, regex=True): URL 패턴을 포함하는지 확인하여 True/False 시리즈 생성
    # ~ (틸드): True인 행(URL 포함)을 제외하고 False인 행(URL 미포함)만 선택
    rows_before = len(df)
    df_cleaned = df[~df['text'].astype(str).str.contains(url_pattern, regex=True)]
    rows_after = len(df_cleaned)
    rows_deleted = rows_before - rows_after

    print(f"✅ 클리닝 완료. 총 {rows_deleted}개의 행이 삭제되었습니다.")

    # 3. 결과 저장
    try:
        df_cleaned.to_csv(output_file, index=False, encoding='utf-8')
        print(f"🎉 클리닝된 데이터 ({rows_after} 행)가 '{output_file}'(으)로 성공적으로 저장되었습니다.")
    except Exception as e:
        print(f"오류: 파일 저장 중 문제가 발생했습니다: {e}")


if __name__ == "__main__":
    delete_rows_with_links(INPUT_FILE, OUTPUT_FILE, URL_PATTERN)