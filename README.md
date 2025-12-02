![header](https://capsule-render.vercel.app/api?type=soft&color=auto&height=300&section=header&text=vaccine%20Review💉&fontSize=90)

# 🦠 COVID Vaccine Controversy Analysis by BERT/DeBERTa
**온라인 댓글 기반 코로나 백신 여론 분석 프로젝트**

[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=Python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-FFD21C?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Transformers](https://img.shields.io/badge/Transformers-FF9A00?style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/docs/transformers/index)
[![DeBERTa v3](https://img.shields.io/badge/DeBERTa%20v3-NLP?style=flat-square&color=0A1F44)](https://huggingface.co/docs/transformers/model_doc/deberta_v2)
[![KoELECTRA](https://img.shields.io/badge/KoELECTRA-Korean%20NLP-blue?style=flat-square)](https://huggingface.co/monologg/koelectra-base-v3-discriminator)
[![BERTopic](https://img.shields.io/badge/BERTopic-Topic%20Modeling-0A1F44?style=flat-square)](https://maartengr.github.io/BERTopic/)
[![HDBSCAN](https://img.shields.io/badge/HDBSCAN-Clustering-FF6F00?style=flat-square)](https://hdbscan.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=Matplotlib&logoColor=white)](https://matplotlib.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=Jupyter&logoColor=white)](https://jupyter.org/)

---

## 1. 연구 개요 (Overview)

### 1.1 왜 “백신 논란”을 댓글로 봤는가?

코로나 팬데믹 동안 사람들의 감정과 생각은  
뉴스 기사 본문이 아니라 **댓글·포럼·SNS**에 훨씬 더 직접적으로 드러났다.

- 뉴스 기사: “정책이 발표되었다”는 **사실** 중심
- 온라인 댓글: “그래서 나는 **불안/분노/찬성/냉소**를 느낀다”는 **감정과 해석** 중심

기존 연구들은 주로 **확진자 수, 사망자 수, 금융 지표**처럼  
“숫자로 정리된 지표”와 여론의 상관관계를 다루는 경우가 많다.

하지만 실제 논란의 핵심은 다음과 같은 질문에 가깝다.

> 사람들은 **정확히 무엇 때문에** 화가 났고, 불안했을까?  
> **백신 부작용** 때문이었을까,  
> 아니면 **마스크·백신 의무화, 경제적 부담, 정치 갈등** 때문이었을까?

이 프로젝트는 이 질문에 답하기 위해  
**대규모 온라인 댓글 데이터**를 모으고,  
**감성 분석 + 토픽 모델링**으로  
“코로나 백신 논란의 축이 어디에 있었는지”를 추적한다.

---

### 1.2 연구 질문

이 프로젝트는 다음 네 가지 질문에 집중한다.

- **Q1.** 코로나/백신 관련 온라인 댓글의  
  **감성 분포(긍·부정)**는 어떠한가?
- **Q2.** 전체 여론에서 **부정 감성의 비중은 얼마나 높은가?**
- **Q3.** 부정/긍정 댓글은 각각 **어떤 주제(토픽)**를 중심으로 모이는가?
- **Q4.** 논란의 중심축은
  - **부작용·의학적 위험**인가,  
  - 아니면 **정책·의무화·경제·정치 갈등**인가?

---

### 1.3 이 프로젝트가 한 일 (요약)

1. 여러 사이트(포럼·리뷰·Q&A 등)에서  
   **코로나/백신 관련 댓글·리뷰 약 10만 건**을 **키워드 기반**으로 크롤링
2. **다단계 전처리 + 주제 관련성 필터링**을 통해  
   **실제로 코로나/백신 논의에 해당하는 텍스트만 남긴 고순도 데이터셋** 구축
3. 약 **2,200개** 댓글을 사람이 직접 라벨링하여  
   **DeBERTa v3 기반 감성 분류 모델 (Val. Acc ≈ 0.87)** 학습
4. 감성 라벨과 **BERTopic 토픽 모델링**을 결합해  
   “**무슨 이슈가 부정 여론을 끌어올렸는지**”를 분석

---

## 2. 데이터 수집 (Data Collection)

### 2.1 수집 대상 사이트 & 키워드

크롤링은 **“코로나/백신 논의를 실제로 볼 수 있는 공간”**에 집중했다.

| 구분 | 플랫폼/사이트 | 언어 | 주요 수집 대상 | 크롤링 방식 / 키워드 예시 |
|------|---------------|------|----------------|----------------------------|
| 커뮤니티 | Reddit (여러 서브레딧) | 🇺🇸 | 게시글/댓글 | `covid`, `vaccine`, `pfizer`, `moderna`, `jab`, `side effect` 등으로 제목/본문 검색 후 댓글 수집 |
| 헬스 포럼 | WebMD, HealthBoards, Patient.info 등 | 🇺🇸 | 증상/부작용 관련 게시글·댓글 | `vaccine`, `shot`, `reaction`, `booster`, `symptoms` 등 키워드 기반 포럼 글/댓글 수집 |
| 약 리뷰 | Drugs.com 등 | 🇺🇸 | 백신·약물 리뷰 | 약 이름 + `covid`, `vaccine` 등 조합으로 리뷰 검색 |
| 국내 Q&A/커뮤니티(실험) | Naver 지식iN, DC Inside 등 | 🇰🇷 | 관련 질문/게시글 | `백신`, `코로나`, `부작용`, `타이레놀` 등으로 크롤링 시도 (최종 분석에서는 대부분 제외) |

> 최종 분석은 **언어적 일관성**을 위해  
> **영어 텍스트 중심 데이터셋**을 사용했다.

---

### 2.2 전처리 단계별 데이터 수 변화

교수님 피드백을 반영해,  
**각 단계에서 데이터가 어떻게 줄어들었는지**를 한눈에 볼 수 있게 정리했다.

> 수치는 모두 **약(approximately)** 값이다.

| 단계 | 설명 | 남은 데이터 수(건) |
|------|------|--------------------|
| **0. Raw merged** | 여러 사이트에서 크롤링한 뒤, 기본적인 형식 통일만 수행한 상태 | **약 108,000** |
| **1. 구조적 노이즈 제거** | `[deleted]`, 너무 짧은 잡담(lol, ok 등), 내용 없는 문장 제거 | 약 100,000 내외 |
| **2. 언어 필터링** | 비영어 텍스트 제거 (비영어 비율이 높은 문장 필터링) | **약 99,000** |
| **3. 주제 관련성 필터링** | 코로나/백신 관련 키워드가 **하나도 없는 문장 제거** | ✅ 관련 있음: **23,939**<br>❌ 관련 없음: 75,338 (제외) |
| **4. 링크·정보공유 위주 문장 제거** | 기사/논문 링크만 공유하거나, 의견/감정이 거의 없는 문장 추가 삭제 | **약 23,352** |
| **5. 모델 학습/분석용 최종본** | 날짜/텍스트/사이트 정보가 완비되고, 전처리가 완전히 끝난 데이터 | **20,929** (부정 18,024 / 긍정 2,905) |

→ **108k → 20,929건**으로 줄어드는 과정에서  
“**주제와 감정을 갖고 있는 문장**만 남기는 것”에 방점을 찍었다.

---

### 2.3 수동 라벨링 데이터

- 전체 데이터 중 **약 10% (2,200~2,300개)**를 무작위로 샘플링
- 사람(연구자)이 직접 **두 가지 라벨**을 동시에 부여
  - Binary: `0 = 부정`, `1 = 긍정`
  - Three-Class: `0 = 부정`, `1 = 중립`, `2 = 긍정`
- 이후 성능·안정성을 고려해  
  최종 파이프라인은 **Binary 분류(부정 vs 긍정)**에 집중

---

## 3. 방법론 요약 (Methods)

### 3.1 전처리 파이프라인

전처리는 크게 네 단계로 구성된다.

1. **구조적 노이즈 제거**
   - `[deleted]`, `[No Content]` 등 삭제된 게시물 제거  
   - `lol`, `ok`처럼 **20자 미만**, **5단어 미만**의 의미 없는 짧은 문장 삭제

2. **언어 필터링**
   - 비영어 문자 비율이 **일정 기준 이상**인 문장 제거  
   - 영어 기반 사전학습 모델 사용을 고려한 일관성 확보

3. **형식적 노이즈 제거**
   - URL, 특수문자, 이모티콘 등 제거
   - 불용어(stopwords) 제거  
     (너무 일반적인 단어 + 전처리 후에도 의미를 흐리는 잔여 단어)

4. **주제 관련성 필터링 (Keyword-based Relevance)**
   - 아래와 같은 코로나/백신 핵심 키워드 중  
     **하나도 포함되지 않은 문장은 “주제 무관”으로 제거**

   ```python
   KEYWORDS = [
       'vaccine', 'covid', 'coronavirus', 'side effect', 'adverse', 'pfizer', 'moderna',
       'booster', 'jab', 'shot', 'vax', 'myocarditis', 'astrazeneca', 'janssen',
       'symptoms', 'mandate', 'mask', 'masked', 'unvaccinated', 'vaxxed', 'unvaxxed',
       'hospital', 'death', 'long covid', 'long-covid', 'spike protein', 'mrna'
   ]
   ```
   → 전처리의 핵심은 **“양을 줄이더라도, 코로나/백신 논쟁에 실제로 해당하는 문장만 남기자”**는 원칙이다.

---

### 3.2 감성 분석 모델 (Sentiment Classification)

**모델 후보**

- KoELECTRA (한국어, 실험용)
- BERT 계열 (영어, baseline)
- DeBERTa / DeBERTa v3 (영어, **최종 선택**)

**최종 설정 (요약)**

- **모델:** DeBERTa v3 기반 Binary 분류
- **학습 데이터:** 약 2,200개 수동 라벨링 데이터
- **손실 함수:** CrossEntropy + 클래스 불균형 대응 (class weight 등)
- **결과:** Validation Accuracy ≈ **0.86 ~ 0.87**

KoELECTRA/Three-Class 등 다양한 시도는  
**부록(시행착오 요약)**에 간단히 정리했다.

---

### 3.3 토픽 모델링 (Topic Modeling)

- **1차:** LDA (Latent Dirichlet Allocation)
  - 전처리 문제와 주요 주제 축(경제·정치·마스크·의료비 등) 파악
- **최종:** BERTopic + HDBSCAN
  - 고차원 임베딩 + 밀도 기반 클러스터링
  - 노이즈 문서(`Topic = -1`) 자동 분리
  - 사람이 읽고 해석 가능한 토픽 이름 수동 부여

감성 라벨(부정/긍정)과 토픽을 결합해

> **“어떤 이슈가 어떤 감성에 연결되는지”**를 보는 것이 핵심이다.

---

## 4. 주요 결과 (Results)

### 4.1 감성 분포

최종 데이터(20,929건)에 대해  
DeBERTa v3 Binary 모델이 예측한 결과:

| 감성 | 개수   | 비율     |
|------|-------:|---------:|
| 부정 (Negative) | 18,024 | 약 **86%** |
| 긍정 (Positive) | 2,905  | 약 14%  |

→ 코로나/백신 관련 온라인 댓글은  
전반적으로 **부정 여론이 절대적으로 우세한 구조**를 보인다.

> (README 상에서는 `B.png`, `C.png` 등 시각화 이미지로 함께 제시)

---

### 4.2 부정 토픽 요약

부정으로 분류된 댓글(18,024건) 중  
BERTopic으로 뽑은 **대표 토픽 축(요약)**은 다음과 같다.

| 축 | 대표 내용 (요약) | 예시 키워드 |
|----|------------------|-------------|
| **의료비·의료 시스템 불만** | 병원비, 빚, 의료 접근성, 간호 인력 부족 등 | hospitals, debt, pay, healthcare, nurses |
| **마스크·백신 의무화 갈등** | 상점/직장에서의 마스크 착용 강제, 고객–직원 갈등 | walmart, store, customers, masks, enforce |
| **정치·책임 공방** | 트럼프, 정부, 누구 책임인가에 대한 논쟁 | trump, president, responsible, politics |
| **백신 부작용·장기 후유증 걱정** | 접종 후 증상, 사망·장기 코로나에 대한 두려움 | effects, infection, long, died, side |
| **아동·취약계층 걱정** | 아이 접종, 소아과, 기저질환자 마스크 착용 문제 | kids, children, parents, asthma, breathing |
| **음모론·hoax 프레이밍** | “covid는 사기다”, “propaganda다”라는 담론 | hoax, propaganda, fake, deny |

**핵심 포인트**

- “백신 부작용”만이 아니라,
- **의료비·병원 시스템**,  
- **마스크/백신 의무화**,  
- **정치적 갈등**,  
- **음모론 프레이밍**이 모두 뒤섞여  
  **부정 여론을 형성**하고 있다.

---

### 4.3 긍정 토픽 요약

긍정 댓글(2,905건)에서 나타난 주요 토픽 축은 다음과 같다.

| 축 | 대표 내용 (요약) | 예시 키워드 |
|----|------------------|-------------|
| **mRNA 백신 옹호·효과** | 효과를 경험했다, 면역이 생겼다 등 | vaccine, vaccinated, mrna, immune |
| **경미한 부작용 경험 공유** | 팔이 아팠지만 괜찮았다, 부작용이 생각보다 약했다 | pfizer, moderna, side, second, dose |
| **의료진·병원의 역할 인정** | 의료진의 희생과 시스템 유지에 대한 감사 | hospital, patients, care, nurses |
| **마스크/규정 준수 긍정** | 마스크 착용이 타인을 보호한다는 인식 | wear, mask, protect, medical |

→ 같은 “병원/마스크/백신”이라도  
어떤 사람에게는 **부정(비용·강제성)**,  
어떤 사람에게는 **긍정(보호·안전)**으로 받아들여진다는  
**여론의 양면성**이 뚜렷하게 드러난다.

---

### 4.4 “논란의 축”에 대한 정리

토픽 결과를 감성과 함께 요약하면:

1. **초기에는** 백신 자체의 **효능·부작용**이 중요한 이슈였지만,
2. **시간이 갈수록**
   - 마스크/백신 **의무화**,
   - 직장·상점에서의 **규정 적용 방식**,
   - **정치적 책임 공방**,
   - **의료비·병원 시스템에 대한 불만**  
   같은 **정책·제도·경제·정치 이슈**가  
   **부정 여론의 중심축**으로 점점 더 부상한다.

즉,

> 논란의 핵심은 단순한  
> “백신이 위험하냐, 아니냐”를 넘어서  
> **“누가 무엇을 강제하고, 그 비용과 책임을 누가 지는가”**에  
> 더 가까운 문제로 이동한다.

---

## 5. 논의 (Discussion)

### 5.1 공중보건 커뮤니케이션 관점

이 분석은 부정 감성이  
**“부작용 자체”보다 “정책·의무화·경제·정치 갈등”**에 더 민감하다는 점을 시사한다.

향후 팬데믹 대응에서

- “백신은 안전하다”는 메시지 **하나만으로는 부족**하며,  
- 동시에 다음 내용을 함께 설명하는 전략이 필요하다.

1. **정책의 공정성**
   - 누구에게 어떤 기준으로 적용되는지
2. **강제성의 한계와 예외 규정**
   - 건강상 사유, 직업적 특성 등에 따른 예외
3. **보상 구조**
   - 부작용 발생 시 지원, 생계 보장 등
4. **정책 결정 과정의 투명성**
   - 어떤 데이터와 전문가 의견을 바탕으로 결정했는지

---

### 5.2 의료 정책·의료기관 관점

부정 토픽에서 지속적으로 등장한 키워드:

- 병원(**hospital**)
- 빚(**debt**), 비용(**pay**)
- 간호 인력 부족(**nurses**)

→ 백신 논란은 단순히  
“맞을까, 말까”를 넘어서

> **“이 시스템 안에서 아프면 나는 얼마나 빚을 지게 되는가?”**

라는 질문과 결합되어 있음.

정책적으로는

- **무료 접종**
- **보험 적용 범위**
- **의료비 지원**
- **상담 창구 안내**

같은 정보를 **적극적으로 알리는 것 자체**가  
부정 여론 완화에 중요할 수 있다.

---

### 5.3 정보 플랫폼·언론·팩트체크 관점

부정 토픽 중 **“covid hoax / propaganda / 음모론”** 토픽은  
상당히 또렷한 클러스터로 분리되었다.

이를 활용하면:

1. 특정 키워드/프레이밍  
   (예: `hoax`, `fake`, `plandemic` 등)의  
   등장 빈도와 맥락을 **상시 모니터링**하고,
2. 해당 프레이밍이 급증하는 시점에
   - **팩트체크 콘텐츠 노출 강화**
   - **공신력 있는 설명글 상단 고정**
   - **추천 알고리즘 조정**

같은 **조기 개입 전략**을 설계할 수 있다.

---

## 6. 한계 및 향후 과제

### 6.1 한계

1. **모델 성능의 한계**
   - Validation Accuracy ≈ **0.87** 수준으로 나쁘진 않지만,
   - 라벨링 오차, 미세한 뉘앙스 오분류 가능성 존재

2. **데이터 편향**
   - Reddit/영어권 포럼 비중이 높아  
     특정 국가·문화·정치 환경에 치우친 여론을 반영

3. **중립 감성 구분의 어려움**
   - Three-Class 모델에서는 **중립 클래스 분리**가 매우 어려웠고,
   - 최종적으로 **Binary 분류**에 집중하게 됨

---

### 6.2 향후 과제

1. **모델 고도화**
   - RoBERTa, GPT 계열 등 다른 아키텍처와 비교
   - 감성 + 토픽/속성 동시 예측하는 **Multi-task 학습** 시도

2. **Relevance 필터 자동화**
   - 키워드 기반 필터 대신  
     별도의 **주제 관련성 분류기(ML 모델)** 도입

3. **다국어·다지역 확장**
   - 한국/유럽/기타 지역 포럼을 추가해  
     국가별 백신 논란 패턴 비교

4. **실시간 대시보드화**
   - 감성·토픽 흐름을 실시간으로 시각화하는  
     **온라인 여론 모니터링 시스템**으로 확장

---



   ## 📂 데이터 / 폴더 구조 (Data & Folder Layout)


```text
data/
├─ raw/                     # 크롤링 직후 또는 최소 전처리 상태의 원본 데이터
│   ├─ FINAL_DATA_CLEANED_READY.csv
│   ├─ FEAR_raw.csv / FEAR_source.csv   # (옵션) 공포지수 원본
│   └─ ... (개별 사이트별 원본 CSV들)
│
├─ interim/                 # 중간 전처리/필터링 결과
│   ├─ FINAL_DATA_FILTERED_#TRUE.csv
│   │   # is_related_topic = True 인 코로나/백신 관련 텍스트만 남긴 버전
│   ├─ FINAL_DATA_FILTERED_#FALSE.csv
│   │   # 관련성이 낮아 제거된 텍스트 (분석에는 사용 X, 검증용으로 보관)
│   ├─ FINAL_DATA_ROWS_#DELETED.csv
│   │   # 링크 공유 위주·의견/감정이 거의 없는 중립 문장 추가 삭제본
│   └─ ... (필요한 중간 버전들)
│
├─ processed/               # 분석/모델 학습에 사용되는 최종본
│   ├─ DDDD.csv
│   │   # 최종 분석용 메인 데이터셋
│   │   # (정제 완료 텍스트 + 날짜 + 사이트 정보 + 모델 예측 감성 등)
│   ├─ labeled_output#.csv
│   │   # 전체 데이터의 약 10% 샘플에 대해
│   │   # 사람이 직접 Binary/Three-Class 감성 라벨을 붙인 결과
│   ├─ 10_per#_final.csv
│   │   # 수동 라벨링 정제본 (학습/검증에 실제 사용한 버전)
│   └─ ... (토픽모델링/시계열용으로 가공된 추가 CSV가 있다면 여기에)
│
└─ external/                # 외부 지표/보조 데이터
    ├─ FEAR#.csv
    │   # 공포·탐욕 지수(Fear-Greed Index) 시계열
    │   # 날짜(date) 기준으로 DDDD.csv의 부정 비율과 merge해서 사용
    └─ ... (향후 추가할 다른 외부 지표들)
```



---

## 주요 CSV 파일 설명

| 파일명 | 역할/내용 |
| --- | --- |
| `FINAL_DATA_CLEANED_READY.csv` | 여러 소스에서 크롤링한 원본 데이터를 기본적인 정제(삭제된 글, 너무 짧은 글, 비영어 등)까지 마친 통합본 |
| `FINAL_DATA_FILTERED_#TRUE.csv` | 위 통합본에서 코로나/백신 관련 키워드가 포함된 행만 남긴 버전. True/False 중, 분석에 사용하는 “관련 있음” 데이터 |
| `FINAL_DATA_ROWS_#DELETED.csv` | TRUE 데이터 중에서도 링크만 공유하거나 의견/감정이 거의 없는 문장을 추가로 제거한 버전 |
| `FEAR#.csv` | 외부에서 가져온 공포·탐욕 지수(Fear-Greed Index) 시계열 데이터. 날짜 기준으로 부정 비율 시계열과 합쳐 상관/DTW 분석에 사용 |
| `DDDD.csv` | 최종 분석용 메인 데이터셋. 전처리 + 주제 필터링 + 링크/중립 삭제까지 거친 후, DeBERTa Binary 모델의 감성 라벨이 부여된 상태의 데이터 |
| `labeled_output#.csv`, `10_per#_final.csv` | 전체 데이터 중 약 10%를 샘플링해 사람이 직접 부정/중립/긍정 라벨을 붙인 결과. 모델 학습/검증에 사용되는 골드 레이블 세트 |
