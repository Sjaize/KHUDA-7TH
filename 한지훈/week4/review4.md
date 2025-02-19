---
name: 📝 리뷰 과제
about: 토이 프로젝트 '나랏말싸미' 관련 조사 및 아이디어
title: '[Week 4] 주차 리뷰 - 이름'
labels: ['review']
assignees: ''
---

## 관련 조사
<!-- 이번 주차에 다룬 주요 주제를 작성해주세요 -->

<aside>

### 개요

모델을 학습도 시키면서 교정이 타당하게 진행되기 위해서 아래의 과정을 생각해 보았습니다.

1. 틀린 부분이 있는 문장들을 맞춤법 교정기에 돌려서 올바른 문장과 그것에 상응하는 문법 규칙을 추출한다.
2. 1의 내용으로 모델을 학습시킨다.
3. 모델에게 올바른 문장을 주었을 때 규칙에 기반하여 틀린 문장을 생성한다.
</aside>

<aside>

### 부산대 맞춤법 교정기

이를 위해 먼저 코랩에서 부산대 맞춤법 교정기를 돌려보았습니다. 임시로 해보기 위해서 틀린 문장으로는 AI HUB의 “인터페이스(자판/음성)별 고빈도 오류 교정 데이터”의 샘플 데이터를 사용하였습니다.

[AI-Hub](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&dataSetSn=71560)

![image (9)](https://github.com/user-attachments/assets/2416fce3-2a34-4137-8832-8fbec934d2d5)

### 코드

[Google Colab](https://colab.research.google.com/drive/1u65GMDvzTTtu9tZ2sVSlG-CNZUpdolT1#scrollTo=dsb6_OLIA3E3)

결과 부분 위주로 봐주시면 될 거 같습니다. 확실히 교정 성능도 아주 높고, 해설과 예시도 제시되어 있어 퀴즈 정답 제공에 사용할 수 있을 거 같아요.

![image (10)](https://github.com/user-attachments/assets/2285d99e-c0f7-443a-b75c-6ca2774ce9f0)

이걸 어떤 방식으로 응용할지는 좀 더 고민을 해봐야 할 거 같습니다. 의견 있으신 분들은 알려주세요!

</aside>

<aside>

### 네이버 맞춤법 교정기

네이버 맞춤법 교정기도 깃이 있어서 사용해보았는데, 여러 면에서 부산대 맞춤법 대비 아쉬운 점이 많네요. 

[GitHub - ssut/py-hanspell: 파이썬 한글 맞춤법 검사 라이브러리. (네이버 맞춤법 검사기 사용)](https://github.com/ssut/py-hanspell?tab=readme-ov-file#words)

![image (11)](https://github.com/user-attachments/assets/af292966-b0af-48da-a360-f4945846f710)

우선 성능도 부산대 비해서 좀 떨어지는 걸 느꼈습니다. ‘하구’ 같은거를 못잡기도 하고요. 그리고 교정의 근거를 구체적으로 알려주지 않고 다음의 5가지 유형으로 분류만 되어있어 해설 제공에 어려움이 있어 보입니다.

- 0: 맞춤법 검사 결과 문제가 없는 단어 또는 구절
- 1: 맞춤법에 문제가 있는 단어 또는 구절
- 2: 띄어쓰기에 문제가 있는 단어 또는 구절
- 3: 표준어가 의심되는 단어 또는 구절
- 4: 통계적 교정에 따른 단어 또는 구절

+ 그대로 사용하면 “**KeyError: Result” 때문에 아래 글처럼 해야 합니다.**

[KeyError: Result 에러에 대한 간단한 해결책 공유드립니다. · Issue #47 · ssut/py-hanspell](https://github.com/ssut/py-hanspell/issues/47)

그런데 확실히 부산대 모델보다는 더 가벼운 모델인거 같아서 사용하게 될 수도 있을 거 같네요.

</aside>

## 아이디어
<!-- 주요 개념과 내용을 정리해주세요 -->

<aside>

### 개요

Open AI API에 세부 프롬프트를 넣는 방식으로 문장을 생성해 보았습니다.

- 사용 모델: gpt-4o
</aside>

<aside>

### 프롬프트 구성

content(대주제): "너는 한국어 맞춤법을 일부러 틀리게 만드는 AI야. 한국인이 자주 실수하는 맞춤법을 포함한 문장을 만들어줘.”

프롬프트

- 아래에는 틀린 단어와 맞는 단어의 쌍이 정의되어 있어(words). 이 쌍들 중 10개를 무작위로 골라서 문법이 틀린 한국어 문장들을 10개 만들어줘.
- 이때 문장의 주제는 친구간 대화로 만들어줘.
- 추가로 각 문장의 정보들을 취합한 CSV 형태의 데이터셋을 만들어줘.
</aside>

<aside>

### 코드 공유(GPT 문장 생성 프롬프트):

[Google Colab](https://colab.research.google.com/drive/1BjkybW2psItD_0_m7XPQ56JloTCimZ8a#scrollTo=0sWxjUL4IxsU)

</aside>

<aside>

### 결과

실험해본 부분은 사전에 언급했던 “주제”를 한정해서 생성하는 부분이었습니다. 프롬프트의 파란색 부분만 바꿔가며 생성해 보았는데, 결과적으로 확실히 각 주제에 맞는 문장들이 생성되는 것을 볼 수 있었고, 주제가 구체화됨에 따라 문장이 더 풍부해지는 것을 느낄 수 있었습니다.

### 주제 1: 친구간 대화

![image (12)](https://github.com/user-attachments/assets/6caaf0bd-defe-4d5f-ad98-d3888b49303d)

### 주제 2: 비즈니스 대화

![image (13)](https://github.com/user-attachments/assets/98d18537-ac31-428b-be7d-2b7630c954b0)

### 주제 3: 영화 대사

![image (14)](https://github.com/user-attachments/assets/6a8698e9-7078-4eb9-9e54-d3b1b287adb9)

- 영화 대사와 같이 정답이 있는(대사가 고정된) 경우에는 실제 대사와 같게 하면 틀린 어법을 반영하기 어려워 GPT가 대사를 지어내는 경향이 있었습니다.
</aside>

<aside>

### 생각하는 데이터셋 형태

![image (15)](https://github.com/user-attachments/assets/b34eccf6-ff4e-414a-85a0-216384c9be93)

위 사진과 같이 우선 생각하는 데이터셋 칼럼 구성은 다음과 같습니다.

- 올바른 문장
- 틀린 문장
- 오류 단어
- 올바른 단어
- 오류 위치
- 도움말
</aside>
