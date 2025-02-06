---
name: 📝 리뷰 과제
about: ML 시스템설계 1 추천시스템, 2 응용
title: '[Week 3] 주차 리뷰 - 한지훈'
labels: ['review']
assignees: ''
---

## 주제
<!-- 이번 주차에 다룬 주요 주제를 작성해주세요 -->
- 실제 면접 시에는 STAR 기법으로 요약하는 것이 중요하며, 간접 경험을 위해 녹음을 많이 해보는 것도 좋다.
- 추천 시스템은 필터버블 등 문제 때문에 정확도 외 세렌디피티 등 다른 요인들도 고민한다.
- squeeze and excitation 방식은 attention의 일종으로 CNN 등에 적용되어 성능을 높일 수 있다. 

## 내용
<!-- 주요 개념과 내용을 정리해주세요 -->

### 핵심 개념 1: 두개 이상 제품을 묶음으로 추천해주는 Amazon
<aside>

### 개요

추천시스템은 그 유용성으로 인해 정말 많은 서비스들에 적용되고 있다. 그 중에서 추천시스템이 흥미롭게 적용된 예시 한가지를 살펴보고자 하는데, 바로 아마존이다.

- 아마존 사이트에서 구매하고 싶은 물건을 누르면, 하단에 관련 상품들 뜨는 것 확인 가능
- 다른 여러 쇼핑 사이트도 제공하는 기능, but 아마존의 Frequently bought together’의 차이점은 사용자가 구매할 것 같은 2개 이상의 상품을 ‘번들’로 추천하는 것
    - EX) 아마존에서 게이밍 마우스를 사려고 클릭하면, 아래와 같이 마우스 뿐만 아니라 느낌이 비슷한 장패드와 키보드까지 하나의 번들로 추천해주는 걸 볼 수 있다.
    
    ![image (9)](https://github.com/user-attachments/assets/19d406cf-4709-443c-b5c4-c5fe409e724e)

### 방법

아마존의 ‘번들 추천’은 크게 ‘**아이템 기반 협업 필터링’과 ‘연관 규칙 학습’을 통해 진행된다.**

- **아이템 기반 협업 필터링(Item-Based Collaborative Filtering)**
    - 특정 상품을 구매한 사용자가 **또 다른 특정 상품도 구매하는 패턴을 분석**하는 방식.
    - 예) "맥북을 산 고객들은 70% 확률로 애플 정품 충전기도 구매했다."
- **연관 규칙 학습(Association Rule Learning)**
    - 데이터 마이닝 기법을 활용해, 어떤 제품들이 **함께 구매되는지 패턴을 학습**.
    - 대표적인 알고리즘: **Apriori Algorithm**, **FP-Growth Algorithm**
    - 예) "아이폰을 구매한 사람들 중 60%가 정품 케이스도 같이 구매했다."

→ **과거 구매 이력 + 상품 속성(**제품의 카테고리, 리뷰, 가격 등)**까지 고려 가능**

### 다른 서비스와의 차별점

기존 서비스

- 주로 "비슷한 제품", "같은 브랜드의 제품" 추천
    - 아이폰 → **비슷한 아이폰 모델, 같은 브랜드의 다른 기기** 추천
    - **맥북 → 다른 맥북 모델** 추천

아마존

- "**이 제품을 산 사람이 실제로 추가로 구매한 상품**"을 추천
- 모델이 "**가장 자연스럽게 번들로 묶이는 조합**"을 학습하여 제안
    - 노트북을 보면 **"무선 마우스, 노트북 거치대, 모니터 확장 케이블"** 추천
- **"비슷한 제품 추천"이 아닌 "함께 사는 제품 추천"**
</aside>

### 핵심 개념 2: 세렌디피티와 필터버블
<aside>

세렌디피티 정의: 추천 결과가 얼마나 "어쩌면 발견하지 못할 수도 있었을 정도로 놀라운지”

![image (10)](https://github.com/user-attachments/assets/ff5f9c30-7e10-4804-9c18-69b2f13d51e4)

- R : 유저에 대해 생성된 추천 아이템 전체 집합
- R(unexp): 유저에 대해 *unexpected* 한 아이템 부분집합
- R(useful): 유저에 대해서 *useful* 한 아이템 부분집합
    - 사용자 rating에 의해 결정

### 필터버블

정의: 사용자의 기존 관심사에 갇혀 새로운 정보를 접하기 어려운 현상

- 유튜브 한 분야 영상 보다보면 알고리즘 점령되는 현상

추천시스템 평가요소: 정확도 말고도 **Diversity, Serendipity, Novelty, Coverage 등 다양**

- Accuracy : 유저의 평점/ 소비에 맞게 예측하는지에 대한 척도
- Diversity : 다양한 유형의 아이템이 추천되는지에 대한 척도
- Serendipity : 예상할 수 없는 아이템이 추천되는지에 대한 척도
- Novelty : 그동한 경험하지 못한 참신하고 새로운 아이템이 추천되는지에 대한 척도
- Coverage : 얼마나 많은 아이템이 추천되는지에 대한 척도

이 중 Accuracy에만 집중하면 필터버블에 빠질 수 있음

핵심: 그냥 새로운게 아니라 흥미 있게(좋은 방향으로) 새로운 것을 추천해야 함

- 협업필터링, 인기도 등 고려 필요

활용 예시

Spotify - Tastebreakers

![image (11)](https://github.com/user-attachments/assets/5b0605d5-62b1-46ff-a817-4e75ebfa25a0)

Youtube - explore

![image (12)](https://github.com/user-attachments/assets/23207079-3da1-4802-8d61-a07eadf49c24)

</aside>

## 참고 문헌
<!-- 참고한 자료의 제목과 링크를 작성해주세요 -->
1. Amazon
  - [https://www.amazon.com](https://www.amazon.com/)

2. 예시와 함께 아마존 추천엔진 이해하기 : 아이템 기반 필터링 기법을 중심으로
  - [https://blog.bizspring.co.kr/테크/아마존-추천엔진-아이템-기반-필터링/](https://blog.bizspring.co.kr/%ED%85%8C%ED%81%AC/%EC%95%84%EB%A7%88%EC%A1%B4-%EC%B6%94%EC%B2%9C%EC%97%94%EC%A7%84-%EC%95%84%EC%9D%B4%ED%85%9C-%EA%B8%B0%EB%B0%98-%ED%95%84%ED%84%B0%EB%A7%81/)

3. 논문 리뷰 - Diversity, Serendipity, Novelty, and Coverage: A Survey and Empirical Analysis of Beyond-Accuracy Objectives in Recommender Systems
  - [https://velog.io/@sj970806/논문-리뷰-Diversity-Serendipity-Novelty-and-Coverage-A-Survey-and-Empirical-Analysis-of-Beyond-Accuracy-Objectives-in-Recommender-Systems](https://velog.io/@sj970806/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Diversity-Serendipity-Novelty-and-Coverage-A-Survey-and-Empirical-Analysis-of-Beyond-Accuracy-Objectives-in-Recommender-Systems)
