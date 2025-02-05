---
name: 📝 리뷰 과제
about: 2장 머신러닝 기본지식, 3장 머신러닝 코딩
title: '[Week 2] 주차 리뷰 - 최예지'
labels: ['review']
assignees: ''
---

## 주제
<!-- 이번 주차에 다룬 주요 주제를 작성해주세요 -->
- nDCG 계산법, 적용
- Precision, Recall Trade-off 가 적용되는 다른 예시들
  
## 내용
<!-- 주요 개념과 내용을 정리해주세요 -->

**1. nDCG (Normalized Discounted Cumulative Gain) 란?**

: 랭킹 추천 시스템에서 많이 사용되는 평가 지표이다. 랭킹 추천 시스템이란, 사용자에게 제공할 아이템(영화, 상품, 음악 등)을
중요도 순으로 정렬해서 추천하는 시스템이다. 단순한 추천이 아니라 추천 순서가 중요한 시스템인 것이다.

**2. nDCG의 계산법**

![image](https://github.com/user-attachments/assets/04353b02-6f06-49a1-bd17-13643bb21218)


a. CG (Cumulative Gain)

: 단순히 검색 결과 상위 p개의 아이템의 관련성 점수를 합한 값. 위 수식에서 rel(i)는 i번째 아이템의 관련성 점수이다.


b. DCG (Discounted Cumulative Gain)

: CG에서 더 나아가 단순 합이 아니라, 검색 결과가 앞쪽에 있을수록(순위가 높을수록) 더 중요한 가중치를 부여한다.
이를 위해서 로그 함수를 사용해서 가중치를 조절한다.
- 순위가 낮아질수록 점수가 감소하도록 조정됨

c. IDCG (Ideal DCG)

: 이상적인 DCG로, 관련성이 가장 높은 순서대로 문서를 정렬했을 때의 DCG 값이다. 즉, 최상의 검색 결과일 경우의 DCG 값을 의미한다.
- 예시
- 관련성 점수를 내림차순 정렬 (e.g. [3, 3, 2, 0]
- 이를 DCG 공식에 대입
- 결과가 절대적인 점수이기 때문에 다른 ㄴ추천 모델과 직접 비교하기 어려움 -> 이때 NDCG 사용

d. NDCG (Normalized DCG)

: DCG를 IDCG로 나누어 0과 1 사이의 값으로 정규화하여 모델 간 비교를 가능하게 함.

**3. 추천시스템의 성능평가 방법**



**a. Precision & Recall**

precision 공식 = TP / (TP + FP)
- 예측한 양성 중에서 실제로 양성인 비율
- precision deals with the 'Predicted' row of the confusion matrix
  
Recall 공식 = TP / (TP + FN) = TPR (True Positive Rate)
- 실제 양성 중에서 모델이 양성으로 예측한 비율
- recall deals with the 'Expected' column of the confusion matrix
  
**Recall Trade-off**

완벽한 모델은 정밀도(Precision) = 1, 재현율(Recall) = 1의 값을 가진다.

이는 모든 양성(Positive) 예제를 정확하게 예측하며, 실제 양성인 모든 데이터를 양성으로 예측한다는 의미이다.
그러나 실제로는 정밀도와 재현율 사이에 트레이드오프(trade-off)가 존재한다. 즉, 정밀도를 높이면 재현율이 낮아질 수 있고, 재현율을 높이면 정밀도가 낮아질 가능성이 있다.

![image](https://github.com/user-attachments/assets/76b6dde9-03ed-4683-b097-8f152745d8c4)


**사용예시**

모델의 평가 지표로 정확도(Accuracy)가 가장 널리 사용되지만, 데이터셋이 imbalanced할 경우에는 precision과 recall이 더 중요한 역할을 한다.

**ex1**. COVID-19 감염 여부를 판별하는 모델
- Recall이 더 중요함
  코로나 환자를 놓치면(False Negative) 추가 감염을 유발할 수 있기 때문에,
  실제 감염자를 최대한 찾아내는 것이 우선순위가 된다.
- 이처럼 암과 같은 고위험 질병을 진단하는 경우에도 recall이 중요한 평가 지표로 사용된다.

**ex2**. 추천시스템이나 광고 모델 (유튜브)
- Precision이 더 중요함
- 유튜브 추천 시스템에서는 사용자가 원하지 않는 영상(False positive)을 추천하는 것을 줄이는 게 중요
  만약 추천된 영상이 사용자의 관심과 맞지 않으면, 사용자는 애플리케이션을 닫거나 사용률이 감소할 것이다.
 - 자동화된 마케팅 캠페인에도 이와 같이 precision이 높은 모델을 설계해야 한다.  



## 참고자료
<!-- 주요 개념과 내용을 정리해주세요 -->
- https://www.evidentlyai.com/ranking-metrics/ndcg-metric
- https://velog.io/@whdgnszz1/%EC%B6%94%EC%B2%9C-%EC%8B%9C%EC%8A%A4%ED%85%9C-NDCG
- https://medium.com/analytics-vidhya/precision-recall-tradeoff-79e892d43134
- https://www.shaped.ai/blog/evaluating-recommendation-systems-roc-auc-and-precision-recall
- https://www.v7labs.com/blog/precision-vs-recall-guide
