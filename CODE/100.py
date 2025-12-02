from PIL import Image
import numpy as np
from scipy.stats import pearsonr
from dtaidistance import dtw


def preprocess_image(img_path, threshold=150):
    # 이미지 열기, 그레이스케일 변환
    img = Image.open(img_path).convert('L')
    img = img.resize((600, 300))  # 크기 통일 (필요 시 조절)

    # numpy 배열로 변환
    arr = np.array(img)

    # 임계값으로 선 추출 (어두운 부분만 추출)
    binary = arr < threshold

    # 각 행별로 선의 평균 위치 계산
    positions = []
    for row in binary:
        indices = np.where(row)[0]
        if len(indices) > 0:
            pos = np.mean(indices)
        else:
            pos = np.nan
        positions.append(pos)
    return np.array(positions)


def interp_nan(arr):
    nans = np.isnan(arr)
    not_nans = ~nans
    x = np.arange(len(arr))
    arr[nans] = np.interp(x[nans], x[not_nans], arr[not_nans])
    return arr


# 이미지 파일 경로
img1_path = 'monthly_negative_ratio_trend_final.png'  # 예: 'monthly_negative_ratio_trend_final.png'
img2_path = 'fear.png'  # 예: 'fear.png'

pos1 = preprocess_image(img1_path)
pos2 = preprocess_image(img2_path)

pos1 = interp_nan(pos1)
pos2 = interp_nan(pos2)

# 상관계수 계산
corr, _ = pearsonr(pos1, pos2)

# DTW 거리 계산 (dtaidistance 패키지 필요: pip install dtaidistance)
dtw_dist = dtw.distance(pos1, pos2)

print(f"선들의 상관계수: {corr:.4f}")
print(f"DTW 거리: {dtw_dist:.4f}")
