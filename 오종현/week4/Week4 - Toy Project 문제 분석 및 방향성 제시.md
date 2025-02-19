---
name: 📝 리뷰 과제
about: ML/DL 주차별 리뷰 과제 템플릿
title: '[4주차 리뷰]'
labels: ['review']
assignees: '오종현'
---
 
## 주제
Week4 Toy Project 문제 분석 및 방향성 제시

---

## 내용

**<유사한 서비스>**

1. 네이버 한글날 맞춤법 시험 - 해설 제공 x, 양질의 문제, 매번 새로운 문제 제공

    [https://campaign.naver.com/hangeulquiz/#](https://campaign.naver.com/hangeulquiz/#)
2. Apple 한글 달인 - 해설 제공 o, 오답 노트 기능, 많은 양의 문제
   
    [https://apps.apple.com/kr/app/한글-달인-lite-맞춤법-퀴즈/id682276246](https://apps.apple.com/kr/app/%ED%95%9C%EA%B8%80-%EB%8B%AC%EC%9D%B8-lite-%EB%A7%9E%EC%B6%A4%EB%B2%95-%ED%80%B4%EC%A6%88/id682276246)
3. 외않되 상담소 - 문제만 제공, 해설 제공 x, 매번 동일한 문제 제공, 훌륭한 UI

   [https://smore.im/embed/9XEqPjmTOk](https://smore.im/embed/9XEqPjmTOk)
4. 키출판사 맞춤법 퀴즈 - 해설 제공 x, 매번 동일한 문제 제공, 훌륭한 UI

   [https://keymedia.co.kr/spelling-quiz/](https://keymedia.co.kr/spelling-quiz/)
5. AI 맞춤법 퀴즈 - AI를 이용해 생성한 퀴즈 제공, 문제의 질 떨어짐, 훌륭햔 UI

   [https://quiz.zep.us/quiz/67724b9d1131232bfe17efba](https://quiz.zep.us/quiz/67724b9d1131232bfe17efba)
    
</br>

**<제시 가능한 서비스 차별점>**

1. 사람들이 자주 틀리는 경향을 반영한 많은 양의 양질의 문제 (자동화된 생성 과정 필수)
    
    [https://keymedia.co.kr/spelling-quiz/](https://keymedia.co.kr/spelling-quiz/)[https://news.kawf.kr/?searchVol=20&subPage=02&searchCate=09&idx=317](https://news.kawf.kr/?searchVol=20&subPage=02&searchCate=09&idx=317)
    
2. 문제에 따른 해설 제공으로 학습의 질 향상

</br>

**<이용 가능한 데이터>**

1. Ai hub - 인터페이스별 고빈도 오류 교정 데이터
2. 국립국어원 - 맞춤법 교정 데이터
3. 네이버 맞춤법 검사기 활용 데이터 생성

</br>

**<어떤 문제를 풀 것인가?>**

- **어떻게 퀴즈를 생성할 것인가**

  1. 틀린 문장 > 올바른 문장 (오류 교정 모델)
      - 선행된 연구가 많음 (맞춤법 교정 데이터와 KoBart를 이용하여 분석한 사례) 
      - SNS 특유의 말투를 담고 있는 데이터의 문제
      - ML을 이용하지 않아도 접근 가능한 방식
  
  2. 올바른 문장 > 틀린 문장 (노이즈 생성 모델)
      - 최대한 실제 한국 사람들처럼 문장을 생성하는 접근
      - 기존의 데이터셋의 feature와 target을 뒤집어서 접근 가능
      - 문맥을 고려할 수 있는 트랜스포머 기반의 모델 (koT5, llama, koGpt, koAlpaca) + 규칙 기반

  3. GPT와 같은 생성형 모델을 통한 예문 생성
      - 직접 모델링을 할 수 있는 여지가 적음

</br>

- **해설을 어떻게 제공할 것인가**
  - 이용할 수 없는 데이터가 없어 해설 제공을 위한 labeling에 어려움을 겪을 수 있음
  - labeling에 따른 해설을 어떻게 생성할지의 문제

</br>

**<제안>**
- 생성형 모델 API를 이용한 문제와 해설 동시 생성
- 사용자가 자주 틀리는 유형에 대한 문제를 추가로 제공하는 추천시스템을 모델링
