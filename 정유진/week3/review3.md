---
name: 📝 리뷰 과제
about: YB심화 세션 회고 템플릿
title: '[Week 2] 2주차 리뷰 - 정유진'
labels: ['review']
assignees: ''
---

## 주제

- 추천 시스템 접목 실제 예시 (Youtube 이외): 음악 스트리밍 서비스 추천 시스템 심층 분석 및 비교 (Apple Music, Spotify, etc.)
- 이번 주 교재 내용 딥다이브: 심층 신경망(DNN) 기반 추천 시스템 모델 심층 분석
- (선택) 경량화 DNN (발제 내용) 코드 이해 및 적용 방안 연구

## 내용

### 1. 추천 시스템 접목 실제 예시: 음악 스트리밍 서비스 추천 시스템 심층 분석 및 비교

#### 1.1. 음악 스트리밍 서비스 추천 시스템 개요

음악 스트리밍 서비스는 사용자에게 개인화된 음악 추천을 제공하기 위해 다양한 추천 알고리즘을 사용합니다. 각 서비스는 고유한 데이터와 알고리즘을 활용하여 사용자 만족도를 높이고 있습니다. 이번 섹션에서는 Apple Music, Spotify, YouTube Music을 비교 분석합니다.

#### 1.2. 음악 스트리밍 서비스별 추천 시스템 비교

| 서비스      | 주요 추천 알고리즘                               | 특징                                                                                                                                  | 장점                                                                                                                                     | 단점                                                                                                                            |
| ----------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Apple Music | 협업 필터링, 콘텐츠 기반 필터링, 딥러닝 기반 모델 | 전문가 큐레이션, 사용자 취향 분석, Apple 생태계 연동                                                                                    | 높은 음질, 큐레이션 기능, 오프라인 재생                                                                                                  | 개인화 추천 정확도 편차, 폐쇄적인 생태계                                                                                       |
| Spotify     | 협업 필터링, 자연어 처리, 딥러닝 기반 모델         | Discover Weekly, Release Radar, 사용자 플레이리스트 분석                                                                                | 개인화 추천 강점, 다양한 장르 지원, 소셜 기능                                                                                            | 음질 상대적 열세, 팟캐스트 광고                                                                                              |
| YouTube Music | 딥러닝 기반 모델, 콘텐츠 기반 필터링               | YouTube 데이터 활용, 사용자의 시청 기록 분석, 음악 외 다양한 콘텐츠 추천                                                                 | 방대한 음원 라이브러리, 뮤직비디오 지원, 다른 Google 서비스 연동                                                                         | 개인화 추천 정확도 개선 필요, 광고                                                                                              |

#### 1.3. 기술적 분석

##### 1.3.1. 협업 필터링 (Collaborative Filtering)

**정의:** 사용자-아이템 상호작용 패턴을 기반으로 추천을 제공하는 방법. 유사한 사용자의 행동을 분석하거나 유사한 아이템을 찾아 추천합니다.

**수식:**

*   사용자 기반 협업 필터링:

    ```
    similarity(u, v) = Σ(i ∈ Iuv) rui * rvi / (sqrt(Σ(i ∈ Iu) rui^2) * sqrt(Σ(i ∈ Iv) rvi^2))
    prediction(u, i) = Σ(v ∈ Nu) similarity(u, v) * rvi / Σ(v ∈ Nu) |similarity(u, v)|
    ```

    여기서,

    *   `u`, `v`: 사용자
    *   `i`: 아이템
    *   `rui`: 사용자 `u`가 아이템 `i`에 준 평점
    *   `Iuv`: 사용자 `u`와 `v`가 모두 평가한 아이템 집합
    *   `Nu`: 사용자 `u`와 유사한 사용자 집합

*   아이템 기반 협업 필터링:

    ```
    similarity(i, j) = Σ(u ∈ Uij) rui * ruj / (sqrt(Σ(u ∈ Ui) rui^2) * sqrt(Σ(u ∈ Uj) ruj^2))
    prediction(u, i) = Σ(j ∈ Ni) similarity(i, j) * ruj / Σ(j ∈ Ni) |similarity(i, j)|
    ```

    여기서,

    *   `i`, `j`: 아이템
    *   `u`: 사용자
    *   `Uij`: 아이템 `i`와 `j`를 모두 평가한 사용자 집합
    *   `Ni`: 아이템 `i`와 유사한 아이템 집합

**적용 예시:** Spotify의 Discover Weekly는 협업 필터링을 통해 사용자와 유사한 취향을 가진 다른 사용자들이 즐겨 듣는 음악을 추천합니다.

##### 1.3.2. 콘텐츠 기반 필터링 (Content-Based Filtering)

**정의:** 아이템의 메타데이터(장르, 아티스트, 가사 등)를 분석하여 사용자가 선호할 만한 아이템을 추천하는 방법.

**수식:**

*   아이템 프로필 생성:

    ```
    profile(i) = Σ(f ∈ Fi) wif * feature(f)
    ```

    여기서,

    *   `i`: 아이템
    *   `Fi`: 아이템 `i`의 특징 집합
    *   `wif`: 특징 `f`의 가중치
    *   `feature(f)`: 특징 `f`의 값

*   사용자 프로필 생성:

    ```
    profile(u) = Σ(i ∈ Iu) rui * profile(i) / Σ(i ∈ Iu) rui
    ```

    여기서,

    *   `u`: 사용자
    *   `Iu`: 사용자 `u`가 평가한 아이템 집합
    *   `rui`: 사용자 `u`가 아이템 `i`에 준 평점

*   추천 점수 계산:

    ```
    score(u, i) = cosine_similarity(profile(u), profile(i))
    ```

**적용 예시:** Apple Music은 콘텐츠 기반 필터링을 통해 사용자가 이전에 들었던 음악과 유사한 장르나 아티스트의 음악을 추천합니다.

##### 1.3.3. 딥러닝 기반 모델 (Deep Learning-Based Models)

**정의:** 심층 신경망을 사용하여 사용자-아이템 간의 복잡한 관계를 모델링하고 추천을 제공하는 방법.

**모델 종류:**

*   **NeuMF (Neural Matrix Factorization):** 행렬 분해와 신경망을 결합하여 사용자-아이템 간의 잠재적 상호작용 학습.
*   **DeepFM (Deep Factorization Machine):** Factorization Machine (FM)과 심층 신경망을 결합하여 특징 간의 상호작용 모델링.

**수식:**

*   NeuMF:

    ```
    prediction(u, i) = σ(h^T * φ(pu, qi))
    ```

    여기서,

    *   `pu`: 사용자 `u`의 임베딩 벡터
    *   `qi`: 아이템 `i`의 임베딩 벡터
    *   `φ`: 신경망 모델
    *   `h`: 출력 가중치 벡터
    *   `σ`: 시그모이드 함수

**적용 예시:** YouTube Music은 딥러닝 기반 모델을 사용하여 사용자의 시청 기록과 검색 기록을 분석하고, 개인화된 음악 추천을 제공합니다.

#### 1.4. 음악 스트리밍 서비스 추천 시스템의 한계 및 개선 방향

*   **데이터 희소성 문제:** 사용자의 평가나 청취 기록이 부족할 경우 추천 정확도가 낮아지는 문제.
    *   **개선 방향:** 콜드 스타트 문제 해결을 위해 콘텐츠 메타데이터 활용, 사용자 초기 행동 패턴 분석, 능동적 피드백 유도.
*   **다양성 부족 문제:** 사용자의 기존 선호에 치우친 추천으로 새로운 음악 발견 기회 감소.
    *   **개선 방향:** 탐색-활용 균형 알고리즘 적용, 음악 다양성 확보, 실험적 추천 도입.
*   **설명 가능성 부족 문제:** 추천 이유에 대한 명확한 설명 부재로 사용자의 신뢰도 하락.
    *   **개선 방향:** 추천 이유 시각화, 사용자 피드백 반영, 투명성 강화.

#### 1.5. 머메이드 다이어그램

```
graph LR
    A[사용자] --> B(데이터 수집);
    B --> C{데이터 분석};
    C --> D[추천 알고리즘];
    D --> E{개인화된 음악 추천};
    E --> F[사용자 피드백];
    F --> B;
```

#### 1.6. 이미지

![음악 스트리밍 서비스 추천 시스템](https://example.com/music_recommendation.png)

### 2. 심층 신경망(DNN) 기반 추천 시스템 모델 심층 분석 (이전 내용 재활용)

### 3. 경량화 DNN 코드 이해 및 적용 방안 연구 (이전 내용 재활용)

## 참고 자료

*   Carlos A. Gomez-Uribe and Neil Hunt. 2016. The Netflix Recommender System: Algorithms, Business Value, and Innovation. ACM Trans. Manage. Inf. Syst. 6, 4, Article 13 (January 2016), 19 pages. [https://doi.org/10.1145/2843948](https://doi.org/10.1145/2843948)
*   He, Xiangnan, et al. "Neural collaborative filtering." Proceedings of the 26th international conference on world wide web. 2017.
*   Cheng, Heng-Tze, et al. "Deep & cross network for ad click predictions." Proceedings of the 1st workshop on deep learning for recommender systems. 2016.
*   Howard, Andrew G., et al. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv preprint arXiv:1704.04861 (2017).
*   Apple Music 추천 시스템 분석: [https://www.apple.com/](https://www.apple.com/)
*   Spotify 추천 시스템 분석: [https://www.spotify.com/](https://www.spotify.com/)
*   YouTube Music 추천 시스템 분석: [https://music.youtube.com/](https://music.youtube.com/)

=