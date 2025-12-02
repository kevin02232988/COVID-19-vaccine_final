import os
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 폰트 설정 (한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 폴더 생성
out_dir = "./data"
os.makedirs(out_dir, exist_ok=True)

# ----------------------------
# 📥 1. 미국 VIX 다운로드
# ----------------------------
print("📥 VIX 데이터 다운로드 중...")
vix = yf.download("^VIX", start="2019-01-01", progress=False)[['Close']]
vix.rename(columns={'Close': 'VIX'}, inplace=True)
vix.to_csv(os.path.join(out_dir, "VIXCLS.csv"))
print(f"✅ VIX 다운로드 완료: {os.path.join(out_dir, 'VIXCLS.csv')}")
print(f"📋 데이터 개수: {len(vix)}")

# ----------------------------
# 📊 2. 월별/연도별 평균 계산
# ----------------------------
vix_monthly = vix.resample('ME').mean()
vix_yearly = vix.resample('YE').mean()

# 저장
vix_monthly.to_csv(os.path.join(out_dir, "vix_monthly_avg_2019_to_latest.csv"))
vix_yearly.to_csv(os.path.join(out_dir, "vix_yearly_avg_2019_to_latest.csv"))

print("💾 저장 완료:")
print(f" - {os.path.join(out_dir, 'vix_monthly_avg_2019_to_latest.csv')}")
print(f" - {os.path.join(out_dir, 'vix_yearly_avg_2019_to_latest.csv')}")

# ----------------------------
# 📈 3. 그래프 생성
# ----------------------------
plt.figure(figsize=(12, 6))
plt.plot(vix_monthly.index, vix_monthly['VIX'], color='red', label='VIX (미국)')
plt.title("미국 공포지수(VIX) 월별 평균 추이 (2019~현재)")
plt.xlabel("연도")
plt.ylabel("지수 값")
plt.legend()
plt.grid(True)
plt.tight_layout()

plot_path = os.path.join(out_dir, "vix_monthly_trend.png")
plt.savefig(plot_path, dpi=150)
plt.show()
print(f"📊 그래프 저장 완료: {plot_path}")

# ----------------------------
# ⚡ 4. 스파이크(급등 구간) 감지
# ----------------------------
mean_vix = vix_monthly['VIX'].mean()
std_vix = vix_monthly['VIX'].std()
spikes = vix_monthly[vix_monthly['VIX'] > mean_vix + 2 * std_vix]
spikes.index = spikes.index.strftime('%Y-%m')

print("\n=== 스파이크(급등) 시점 (VIX > 평균 + 2σ) ===")
print(spikes)

# ----------------------------
# 🧠 5. 주요 해석
# ----------------------------
print("\n📈 주요 공포지수 급등 구간 (2019~현재)")
print("───────────────────────────────")
print("• 2020 03: 코로나19 팬데믹 초기 – VIX 약 82 기록")
print("• 2020 10: 코로나 재확산 + 美 대선 불확실성")
print("• 2022 02: 러시아-우크라이나 전쟁 발발")
print("• 2023 03: 실리콘밸리은행(SVB) 붕괴")
print("• 2025 전반: 금리·인플레이션 불안, 지정학 리스크")
print("───────────────────────────────")
print("📊 요약:")
print("- 코로나 초기엔 전 세계 공포 확산 → VIX 급등")
print("- 이후 점진적 안정화, 하지만 지정학/금융 리스크마다 단기 급등 반복")
