import os
import json
import pandas as pd
import glob


def get_latest_accuracy(log_dir='./results'):
    # 1. 모든 체크포인트 폴더 찾기 (e.g., ./results/checkpoint-500)
    checkpoint_dirs = glob.glob(os.path.join(log_dir, 'checkpoint-*'))

    if not checkpoint_dirs:
        print("[ERROR] 체크포인트 폴더가 없습니다. 모델 학습이 완료되었는지 확인하세요.")
        return None, None

    # 2. 가장 최근에 수정된 체크포인트 폴더 찾기
    latest_checkpoint = max(checkpoint_dirs, key=os.path.getmtime)

    # 3. trainer_state.json 파일 경로 설정
    log_file = os.path.join(latest_checkpoint, 'trainer_state.json')

    if not os.path.exists(log_file):
        # trainer_state.json이 없으면, 전체 폴더에서 가장 최근의 JSON 파일을 찾습니다.
        log_files = glob.glob(os.path.join(log_dir, '*/trainer_state.json'))
        if not log_files:
            print("[ERROR] 'trainer_state.json' 파일을 찾을 수 없습니다. 학습이 완료되지 않았을 수 있습니다.")
            return None, None
        log_file = max(log_files, key=os.path.getmtime)
        print(f"[INFO] 최신 로그 파일: {log_file}")

    # 4. JSON 파일에서 최종 평가 지표 추출
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # log_history 리스트의 마지막 항목 (최종 평가 결과)
            logs = data.get('log_history', [])

            # eval_accuracy 값이 있는 마지막 로그 찾기
            final_eval = next((log for log in reversed(logs) if 'eval_accuracy' in log), None)

            if final_eval:
                accuracy = final_eval.get('eval_accuracy')
                f1_score = final_eval.get('eval_f1')
                return accuracy, f1_score
            else:
                print("[WARNING] 로그 파일에서 'eval_accuracy' 값을 찾지 못했습니다.")
                return None, None
    except Exception as e:
        print(f"[ERROR] 로그 파일 분석 중 오류 발생: {e}")
        return None, None


# --- 실행 ---
accuracy, f1 = get_latest_accuracy()

if accuracy is not None:
    print("\n--- BERT 모델 최종 성능 ---")
    print(f"✅ Accuracy (정확도): {accuracy:.4f}")
    print(f"✅ F1 Score (균형 점수): {f1:.4f}")
    print("\n이 두 값이 보고서의 성공 지표로 사용됩니다.")
else:
    print("성능 지표 추출에 실패했습니다. './results' 폴더 구조를 확인해 주세요.")