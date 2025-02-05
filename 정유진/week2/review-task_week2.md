---
name: 📝 리뷰 과제
about: YB심화 세션 회고 템플릿
title: '[Week 2] 2주차 리뷰 - 정유진'
labels: ['review','Homework']
assignees: ''
---

## 주제
***<!-- 이번 주차에 다룬 주요 주제를 작성해주세요 -->
2장 ML 기본 지식
3장 ML 코딩
---

1️⃣ 모델 평가와 성능 지표
- Precision & Recall Trade-off의 실제 적용
- nDCG를 활용한 추천시스템 랭킹 평가
- 지니 지수의 의미와 해석
- 상황별 최적 평가 지표 선정

2️⃣ 심층 신경망 구조와 응용
- 오토인코더와 Transformer의 관계
- 이미지 처리를 위한 오토인코더 활용
- GAN의 기본 작동 원리
- 다양한 거리 측정 방식의 특징

3️⃣ 데이터 처리와 알고리즘
- EM 알고리즘과 K-means 클러스터링
- 저수지 샘플링의 개념과 적용
- TF-IDF를 통한 텍스트 중요도 계산
- 차원 축소 기법 (PCA, KLD)

## 내용
### 핵심 개념 1 | 모델 평가와 성능 지표
---
#### Precision & Recall Trade-off |
- **정밀도(Precision)** | 모델이 Positive로 예측한 것 중 실제 Positive의 비율
  - $Precision = \frac{TP}{TP + FP}$
  - 스팸 메일 필터링처럼 False Positive가 치명적인 경우 중요

- **재현율(Recall)** | 실제 Positive 중 모델이 맞춘 비율
  - $Recall = \frac{TP}{TP + FN}$
  - 암 진단처럼 False Negative가 치명적인 경우 중요

```mermaid
graph LR
    A[임계값 ↑] -->|정밀도 ↑ / 재현율 ↓| B[신중한 예측]
    C[임계값 ↓] -->|정밀도 ↓ / 재현율 ↑| D[적극적인 예측]
    
    style B fill:#bbf,stroke:#333
    style D fill:#bfb,stroke:#333
```

- **실제 적용 예시** |
  1. 의료 진단 | 높은 재현율 필요 (진짜 환자를 놓치면 안 됨)
  2. 스팸 필터 | 높은 정밀도 필요 (정상 메일을 스팸으로 분류하면 안 됨)
  3. 추천 시스템 | 비즈니스 목적에 따라 조절 (CTR vs 다양성)

![alt text](image-1.png)


#### nDCG를 활용한 추천시스템 평가 |
- **DCG(Discounted Cumulative Gain)** |
  - $DCG@k = \sum_{i=1}^k \frac{rel_i}{\log_2(i+1)}$
  - 순위가 뒤로 갈수록 가중치를 로그적으로 감소
  - rel_i는 i번째 아이템의 관련성 점수

- **nDCG(normalized DCG)** |
  - $nDCG@k = \frac{DCG@k}{IDCG@k}$
  - IDCG는 이상적인(perfect) 순위에서의 DCG
  - 0~1 사이 값으로 정규화되어 비교 용이

```python
def calculate_ndcg(predicted_ranks, true_ranks, k):
    dcg = sum((2**true_ranks[i] - 1) / np.log2(i + 2) 
              for i in range(min(k, len(predicted_ranks))))
    idcg = sum((2**sorted(true_ranks, reverse=True)[i] - 1) / np.log2(i + 2) 
               for i in range(min(k, len(true_ranks))))
    return dcg / idcg if idcg > 0 else 0
```

#### 지니 지수 |
- **의미** | 불평등도를 측정하는 지표로, 0(완전 평등)에서 1(완전 불평등) 사이의 값
- **결정 트리에서의 활용** | 노드 분할 기준으로 사용
  - $Gini = 1 - \sum_{i=1}^c p_i^2$
  - c는 클래스 수, p_i는 각 클래스의 비율

```python
def gini_index(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return 1 - sum(p * p for p in probabilities)
```


#### 상황별 최적 평가 지표 |
| 문제 유형 | 상황 | 추천 지표 | 설명 |
|---------|------|----------|------|
| **분류** | 균형 데이터 | Accuracy, F1-score | 클래스 간 데이터 수가 비슷할 때 |
| | 불균형 데이터 | AUC-ROC, PR-AUC | 한 클래스가 매우 적은 경우 |
| | 다중 클래스 | Macro/Micro F1 | 여러 클래스 분류 시 |
| **회귀** | 일반적 경우 | MSE, RMSE, MAE | 연속값 예측의 기본 지표 |
| | 이상치 많음 | MAE, MAPE | 이상치에 덜 민감한 지표 |
| | 상대오차 중요 | MAPE, RMSPE | 크기에 따른 오차 중요도가 다를 때 |
| **추천** | 순위 중요 | nDCG, MAP | 추천 순서가 중요한 경우 |
| | 개인화 중요 | User-wise Precision | 개인별 추천 정확도가 중요할 때 |
| | 다양성 중요 | Coverage, Diversity | 추천의 다양성이 중요한 경우 |
| **시계열** | 추세 중시 | MASE, RMSE | 전반적인 추세 예측이 중요할 때 |
| | 계절성 중시 | Seasonal Adjusted | 주기적 패턴이 중요한 경우 |
| | 이상치 탐지 | Precision@k | 특이 패턴 감지가 중요할 때 |

### 평가 지표의 종류와 특성 |

```mermaid
flowchart TB
    M[평가 지표 분류] --> C[분류 문제]
    M --> R[회귀 문제]
    M --> RS[추천 시스템]
    M --> TS[시계열 분석]
    M --> AD[이상치 탐지]
    
    C --> C1[Accuracy<br>F1-score<br>Macro F1<br>Micro F1<br>AUC-ROC<br>PR-AUC]
    R --> R1[MSE<br>RMSE<br>MAE<br>MAPE<br>RMSPE]
    RS --> RS1[nDCG<br>MAP<br>User-wise Precision<br>Coverage<br>Diversity]
    TS --> TS1[MASE<br>Seasonal Adjusted]
    AD --> AD1[Precision@k]
    
    style M fill:#f9f,stroke:#333
    style C fill:#bbf,stroke:#333
    style R fill:#bfb,stroke:#333
    style RS fill:#fbb,stroke:#333
    style TS fill:#bff,stroke:#333
    style AD fill:#fbf,stroke:#333
```

#### 분류 문제 관련 지표 |
| 지표 | 수식 | 계산 요소 | 특징 및 장점 |
|------|------|-----------|--------------|
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | TP, TN, FP, FN | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 간단하고 직관적인 지표<br>• 균형 데이터셋에 적합 |
| F1-score | $\frac{2 * Precision * Recall}{Precision + Recall}$ | Precision, Recall | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 정밀도와 재현율의 조화평균<br>• 불균형 데이터셋에 유용 |
| Macro F1 | 각 클래스의 F1-score의 평균 | 각 클래스의 F1-score | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 다중 클래스 분류에 적합<br>• 각 클래스에 동일한 가중치 부여 |
| Micro F1 | 전체 TP, FP, FN으로 계산한 F1-score | 전체 TP, FP, FN | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 다중 클래스 분류에 적합<br>• 클래스 불균형을 고려함 |
| AUC-ROC | ROC 곡선 아래 면적 | TPR, FPR | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 분류 임계값에 불변<br>• 불균형 데이터셋에 유용 |
| PR-AUC | PR 곡선 아래 면적 | Precision, Recall | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 불균형 데이터셋에 특히 유용<br>• 양성 클래스에 초점 |

#### 회귀 문제 관련 지표 |
| 지표 | 수식 | 계산 요소 | 특징 및 장점 |
|------|------|-----------|--------------|
| MSE | $\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2$ | 실제값, 예측값 | • 0에 가까울수록 좋음<br>• 큰 오차에 민감<br>• 수학적으로 다루기 쉬움 |
| RMSE | $\sqrt{\frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_i)^2}$ | 실제값, 예측값 | • 0에 가까울수록 좋음<br>• MSE와 같은 단위 사용<br>• 직관적 해석 가능 |
| MAE | $\frac{1}{n}\sum_{i=1}^n \|y_i - \hat{y}_i\|$ | 실제값, 예측값 | • 0에 가까울수록 좋음<br>• 이상치에 덜 민감<br>• 해석이 쉬움 |
| MAPE | $\frac{100\%}{n}\sum_{i=1}^n \|\frac{y_i - \hat{y}_i}{y_i}\|$ | 실제값, 예측값 | • 0에 가까울수록 좋음<br>• 상대적 오차 측정<br>• 다른 스케일의 데이터 비교 가능 |
| RMSPE | $\sqrt{\frac{1}{n}\sum_{i=1}^n (\frac{y_i - \hat{y}_i}{y_i})^2}$ | 실제값, 예측값 | • 0에 가까울수록 좋음<br>• 상대적 오차 측정<br>• MAPE보다 큰 오차에 민감 |

#### 추천 시스템 관련 지표 |
| 지표 | 수식 | 계산 요소 | 특징 및 장점 |
|------|------|-----------|--------------|
| nDCG | $DCG_p / IDCG_p$ | 관련성 점수, 순위 | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 순위를 고려한 평가<br>• 불완전한 랭킹에도 적용 가능 |
| MAP | 각 쿼리의 AP의 평균 | Precision, Recall | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 순위와 관련성을 모두 고려<br>• 여러 쿼리에 대한 성능 평가 가능 |
| User-wise Precision | 사용자별 정확도의 평균 | 사용자별 추천 정확도 | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 개인화된 추천 평가에 적합<br>• 사용자 만족도 반영 |
| Coverage | $\frac{\text{추천된 고유 아이템 수}}{\text{전체 아이템 수}}$ | 추천된 아이템, 전체 아이템 | • 0~1 사이 값, 1에 가까울수록 다양함<br>• 추천의 다양성 측정<br>• 롱테일 아이템 추천 평가에 유용 |
| Diversity | 추천 목록 내 아이템 간 평균 거리 | 아이템 간 거리 | • 높을수록 다양함<br>• 추천의 다양성 측정<br>• 사용자 경험 향상 평가에 유용 |

#### 시계열 분석 관련 지표 |
| 지표 | 수식 | 계산 요소 | 특징 및 장점 |
|------|------|-----------|--------------|
| MASE | $\frac{MAE}{MAE_{naive}}$ | MAE, 나이브 예측의 MAE | • 1보다 작을수록 좋음<br>• 스케일에 독립적<br>• 다른 모델과 비교 용이 |
| Seasonal Adjusted | 원 시계열에서 계절성 제거 | 원 시계열, 계절성 요소 | • 계절성 제거 후 추세 파악 용이<br>• 비계절적 변동 분석에 유용<br>• 장기 추세 예측에 도움 |

#### 이상치 탐지 관련 지표 |
| 지표 | 수식 | 계산 요소 | 특징 및 장점 |
|------|------|-----------|--------------|
| Precision@k | $\frac{TP@k}{TP@k + FP@k}$ | 상위 k개 예측 중 TP, FP | • 0~1 사이 값, 1에 가까울수록 좋음<br>• 상위 k개 예측의 정확도 측정<br>• 이상치 탐지의 실용성 평가에 유용 |

```mermaid
flowchart LR
    P[문제 유형] --> C[분류]
    P --> R[회귀]
    P --> T[추천]
    
    C --> C1[균형]
    C --> C2[불균형]
    
    R --> R1[일반]
    R --> R2[이상치]
    
    T --> T1[정확도]
    T --> T2[다양성]
    
    style P fill:#f9f,stroke:#333
```

#### 개인적으로 헷갈리는 평가지표 모음
1. **DCG와 nDCG** (추천시스템에서 많이 사용)
- 작동 원리 |
 DCG는 추천된 항목의 **관련성(relevance)과 위치(position)**를 모두 고려한다. 상위에 나온 관련 항목에 더 높은 가중치 부여하게 되는데 예를 들어, 5점짜리 영상을 1위에 추천하면 높은 점수, 10위에 추천하면 낮은 점수를 주게 된다. nDCG는 이를 0~1 사이 값으로 정규화(가장 이상적인 추천 순서로 나눔)
```python
# 예시: 음악 추천 시스템
# rel_i는 사용자가 실제로 준 평점 (예: 1~5점)
# 추천 순서: [5점짜리 노래, 3점짜리 노래, 4점짜리 노래, 1점짜리 노래]

# DCG 계산
rel_scores = [5, 3, 4, 1]
dcg = (2**5 - 1)/log2(1+1) + (2**3 - 1)/log2(2+1) + (2**4 - 1)/log2(3+1) + (2**1 - 1)/log2(4+1)

# 이상적인 순서 (IDCG 계산용): [5, 4, 3, 1]
ideal_scores = sorted(rel_scores, reverse=True)
idcg = (2**5 - 1)/log2(1+1) + (2**4 - 1)/log2(2+1) + (2**3 - 1)/log2(3+1) + (2**1 - 1)/log2(4+1)

ndcg = dcg / idcg  # 1에 가까울수록 좋은 추천 순서
```


2. **MAP** (정보 검색, 추천시스템에서 사용)
- 작동 원리 | 각 사용자(또는 검색)별로 관련 있는 항목이 얼마나 상위에 있는지 평가한다. 예를들어 첫 페이지에 관련 결과 5개, 두 번째 페이지에 1개보다는
첫 페이지에 6개가 모두 있는 것이 더 좋은 결과라고 할 수 있다. 
```python
# 예시: 검색엔진 결과
# 검색어 "파이썬 머신러닝"에 대한 상위 5개 결과가 관련있는지 (1: 관련있음, 0: 관련없음)
results = [1, 0, 1, 1, 0]  # 1번, 3번, 4번 결과가 관련있음

# AP(Average Precision) 계산
# 관련있는 결과 나올 때마다의 precision 값의 평균
precisions = []
relevant_count = 0
for i, is_relevant in enumerate(results, 1):
    if is_relevant:
        relevant_count += 1
        precision_at_i = relevant_count / i
        precisions.append(precision_at_i)

ap = sum(precisions) / len(precisions)

# 여러 검색어에 대해 이를 평균낸 것이 MAP
```



3. **MASE** (시계열 예측에서 사용)
- 작동원리 | 가장 단순한 예측(naive prediction: 이전 값을 그대로 사용)과 비교
- MASE < 1: 단순 예측보다 더 잘함
- MASE > 1: 단순 예측보다 못함
- MASE = 0.8이면, 단순 예측 대비 20% 더 정확하다는 의미
```python
# 예시: 월별 판매량 예측
actual = [100, 120, 130, 150, 160]  # 실제 판매량
predicted = [105, 125, 135, 145, 155]  # 예측 판매량

# 나이브 예측 = 이전 값을 그대로 사용
naive_pred = actual[:-1]  # [100, 120, 130, 150]
naive_actual = actual[1:]  # [120, 130, 150, 160]

# MAE 계산
mae = np.mean(abs(np.array(actual) - np.array(predicted)))
mae_naive = np.mean(abs(np.array(naive_actual) - np.array(naive_pred)))

mase = mae / mae_naive  # 1보다 작으면 나이브 예측보다 좋은 성능
```

4. **SMAPE** (시계열 예측에서 사용)
- 작동 원리 |오차를 상대적인 비율로 평가하여 실제값과 예측값이 크게 차이나도 0~100% 사이의 값 유지
- MAPE의 단점(실제값이 0에 가까울 때 불안정) 보완
```python
# 예시: 주식 가격 예측
actual = [100, 110, 90, 95, 105]  # 실제 가격
predicted = [95, 105, 95, 100, 100]  # 예측 가격

# SMAPE 계산 (백분율)
smape = 100 * np.mean(2 * abs(np.array(actual) - np.array(predicted)) / 
                     (abs(np.array(actual)) + abs(np.array(predicted))))

# SMAPE는 0%면 완벽한 예측, 100%면 최악의 예측
# 실제값과 예측값의 크기 차이가 클 때 MAPE보다 안정적
```


### 핵심 개념 2 | 심층 신경망 구조와 응용
---
#### 오토인코더와 Transformer의 관계

####  이미지 처리를 위한 오토인코더 활용

#### GAN의 기본 작동 원리

#### 다양한 거리 측정 방식의 특징


### 핵심 개념 3 | 데이터 처리와 알고리즘
---
#### EM 알고리즘과 K-means 클러스터링

#### 저수지 샘플링의 개념과 적용

#### TF-IDF를 통한 텍스트 중요도 계산

#### 차원 축소 기법 (PCA, KLD)

## 참고 문험
---
1. [나무위키 혼동행렬](https://namu.wiki/w/%ED%98%BC%EB%8F%99%ED%96%89%EB%A0%AC)
2. [모두의 연구소 평가지표])(https://modulabs.co.kr/blog/information-retrieval-map-ndcg)
3. https://www.cuemath.com/accuracy-formula/
4. https://wandb.ai/mostafaibrahim17/ml-articles/reports/An-Introduction-to-the-F1-Score-in-Machine-Learning--Vmlldzo2OTY0Mzg1
5. https://blog-ko.superb-ai.com/learn-the-metrics-used-for-model-diagnostics-and-how-to-use-them-part-2/