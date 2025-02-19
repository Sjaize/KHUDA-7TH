# WEEK4_toy project 기초공부

# BERT4Rec 모델

### 📌 주요 특징

- BERT를 추천 시스템에 적용한 모델
- 사용자의 행동 데이터를 순차적 추천 방식으로 학습
- 마스킹 기법을 사용하여 양방향으로 문맥 학습
- 기존 RNN 기반 모델보다 더 먼 과거 정보까지 활용 가능하고 병렬 처리 가능

<aside>
💡

BERT

: 구글에서 만든 자연어 처리 모델

- 마스킹 적용
- 양방향 학습
</aside>

<aside>
💡

병렬 처리

: 여러 작업을 동시에 실행하는 것

↔ 직렬 처리: 하나씩 순차적으로 진행

- RNN(기존 NLP 모델): 단어를 한 글자씩 순서대로 처리
- BERT(Transformer 기반): 문장을 한꺼번에 여러 부분으로 나눠서 처리
</aside>

### 📌 동작 방식

1. 사용자 행동 데이터를 시퀀스로 변환
2. 마스킹 적용: 일부 아이템을 [MASK]로 가려두고, 모델이 그 빈칸을 채우도록 학습
3. Transformer를 이용해 패턴 학습
4. 마스킹된 아이템 예측

### 📌 예제 코드 (영화 추천)

#기본 환경 설정

```python
pip install torch transformers numpy pandas tqdm
```

#데이터 준비 (영화 시청 데이터)

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from transformers import BertModel, BertConfig

# 가상의 사용자-영화 시청 데이터 (각 숫자는 영화 ID를 의미)
user_movie_sequences = [
    [1, 2, 3, 4, 5],  # 사용자 1이 본 영화 리스트
    [10, 20, 30, 40, 50],  # 사용자 2
    [100, 200, 300, 400, 500]  # 사용자 3
]

# 영화 ID를 BERT 모델이 이해할 수 있도록 변환
movie_vocab = {movie_id: idx for idx, movie_id in enumerate(set(sum(user_movie_sequences, [])))}
num_movies = len(movie_vocab)

# 데이터를 Tensor 형태로 변환
sequences = [[movie_vocab[movie] for movie in sequence] for sequence in user_movie_sequences]
sequences = torch.tensor(sequences, dtype=torch.long)
```

#BERT4Rec 모델 구현

```python
class BERT4Rec(nn.Module):
    def __init__(self, num_items, hidden_size=128, num_layers=2, num_heads=2, dropout=0.1):
        super(BERT4Rec, self).__init__()
        self.config = BertConfig(
            vocab_size=num_items,  # 영화 개수
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout
        )
        self.bert = BertModel(self.config)
        self.output_layer = nn.Linear(hidden_size, num_items)  # 다음 영화 예측

    def forward(self, input_ids):
        attention_mask = (input_ids != 0).float()  # 패딩이 아닐 때만 처리
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        return self.output_layer(outputs.last_hidden_state)

# 모델 초기화
model = BERT4Rec(num_items=num_movies)
```

#마스킹 방식으로 모델 학습

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, data, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for sequence in data:
            sequence = sequence.unsqueeze(0)  # 배치 차원 추가
            masked_seq = sequence.clone()
            
            # 일부 영화 ID를 [MASK]로 변경
            mask_idx = np.random.randint(1, len(sequence[0]))
            masked_seq[0, mask_idx] = 0  # 0을 마스킹 값으로 사용
            
            optimizer.zero_grad()
            output = model(masked_seq)  # 예측 값
            loss = criterion(output[:, mask_idx, :], sequence[:, mask_idx])  # 정답과 비교
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

train(model, sequences)
```

#다음 영화 추천

```python
def recommend_next_movie(model, sequence):
    model.eval()
    sequence = torch.tensor(sequence, dtype=torch.long).unsqueeze(0)
    with torch.no_grad():
        output = model(sequence)
    next_movie_id = torch.argmax(output[:, -1, :], dim=-1).item()
    
    # ID를 영화 이름으로 변환
    recommended_movie = [k for k, v in movie_vocab.items() if v == next_movie_id][0]
    return recommended_movie

# 사용자 1의 다음 추천 영화
test_sequence = [movie_vocab[movie] for movie in [1, 2, 3, 4]]  # 마지막 영화(5)를 제외한 시청 기록
next_movie = recommend_next_movie(model, test_sequence)
print(f"추천된 다음 영화: {next_movie}")
```

# Frontend 구성

### UI Markup & Styling (UI 구조 & 스타일링)

: 웹페이지의 구조와 디자인을 담당하는 부분 →  정적인 페이지

- HTML: 웹페이지의 구조를 정의
- CSS: HTML 요소의 스타일 지정
- Tailwind CSS: CSS 클래스를 조합해 빠르게 디자인
- Bootstap: 미리 정의된 UI 요소가 기본으로 포함된 프레임워크

### Core Programing Language (프론트엔드 핵심 프로그래밍 언어)

: 웹사이트의 기능(버튼 클릭, 애니메이션, 데이터 처리)을 구현하는 프로그래밍 언어 → 동적인 기능 추가

- JavaScript: 웹사이트의 동적 기능을 담당하는 기본 언어
- TypeScript: JavaScript의 상위 버전 - JavaScript에 타입을 추가해서 오류를 줄여줌
- WebAssembly: 고성능 언어 실행 가능

### UI Framework & Library

: 웹사이트를 효율적으로 개발할 수 있도록 도와주는 툴

→ JavaScript 만으로는 코드가 너무 길고 복잡하므로 효율성 높이기 위함

- React: Facebook이 만든 UI 라이브러리
- Vue.js: 배우기 쉬운 프레임워크
- Angular: Google이 개발한 대규모 프로젝트용 프레임워크

### State Management

: 웹사이트에서 여러 컴포넌트들이 같은 데이터를 공유할 수 있도록 하는 기능

→ 페이지 간 데이터 공유

ex. 쇼핑몰에서 장바구니에 추가한 상품을 모든 페이지에서 유지하는 기능

- Redux: Flux 아키텍처 기반 중앙 상태 관리
- Recoil: Facebook이 개발한 React 상태 관리 라이브러리
- Zustand: Redux보다 가벼운 상태 관리 라이브러리
- MobX: 반응형 상태 관리 라이브러리
- Context API: React 기본 제공 상태 관리 기능

### Data Fetching & API (데이터 가져오기 & API 연동)

: 웹사이트가 서버에서 데이터를 가져오거나 저장할 때 필요한 기능

→ 서버와 데이터 주고받기

ex. 로그인, 회원가입, 상품 목록 불러오기… 

- Fetch API: 브라우저 내장 API
- Axios: 가장 많이 쓰이는 HTTP 요청 라이브러리
- SWR: React에서 사용하는 데이터 패칭 라이브러리
- React Query: 상태 관리 + API 요청 최적화
- GraphQL Client: GraphQL API 호출을 위한 클라이언트

### Build & Tooling (개발 환경 & 빌드 도구)

: 코드를 최적화하고, 실행 속도를 빠르게 해주는 도구

- Webpack: 코드 번들링 및 최적화 도구
- Vite: 빠른 개발 서버 및 번들링 도구
- Babel: 최신 JavaScript 코드를 구버전에서 실행 가능하게 변환
- ESLint: 코드 스타일 검사 도구
- Prettier: 코드 포매팅 자동화 도구

### Routing (페이지 이동 관리)

: 사용자가 웹사이트에서 여러 페이지를 이동할 수 있도록 도와주는 기능

- React Router: React에서 사용하는 기본 라우팅 라이브러리
- Next.js Router → 서버 사이드 렌더링 지원
- Vue Router → Vue.js 프로젝트에서 사용

# Backend 구성

<aside>
💡

백엔드: 웹사이트나 앱에서 사용자가 직접 보지 못하는 부분 (데이터 저장, 사용자 인증, 비즈니스 로직 처리 등) 을 담당하는 시스템

</aside>

### Server

: 웹사이트나 앱의 요청을 처리하는 컴퓨터

- Web Server: 사용자의 요청을 받아 웹페이지를 보여주는 서버
- Application Server: 비즈니스 로직을 실행하는 서버
- Database Server: 데이터를 저장하고 관리하는 서버

### Database

: 사용자 정보, 게시글, 결제 내역 등 데이터를 저장하고 관리하는 시스템

- 관계형 데이터베이스 - MySQL…
- 비관계형 데이터베이스 - MongoDB…

### 백엔드 프로그래밍 언어

: 서버에서 비즈니스 로직 (사용자 요청 처리, 데이터 저장, API 제공) 을 실행하는 언어

- JavaScript: JavaScript 기반 서버 개발
- Python, Flask: 빠르게 개발 가능
- Java: 대기업에서 많이 사용하는 강력한 언어
- C#: 마이크로소프트 생태계에서 사용
- Go (Golang): 빠르고 안정적인 서버 언어
- PHP: 오래된 웹 개발 언어

### API

: 프론트엔드와 백엔드가 데이터를 주고 받는 규칙

- REST API: URL을 이용해 데이터를 주고받는 가장 일반적인 방식
- GraphQL: 필요한 데이터만 요청할 수 있는 API 방식
- gPRC: 빠른 통신을 위한 프로토콜

### 인증 & 보안

: 사용자 로그인, 접근 권한, 데이터 보호 등을 담당하는 시스템

- JWT: 로그인 인증을 위한 토큰 방식
- OAuth: 구글, 페이스북 등으로 간편 로그인 가능
- HTTPS: 데이터를 암호화해서 주고받는 방식

### 백엔드 프레임워크

: 백엔드 개발을 빠르고 효율적으로 할 수 있게 도와주는 툴

- Express.js (Node.js): 가벼운 웹 서버 프레임워크
- Django (Python): 강력한 기능 제공
- Flask (Python): 간단하고 가벼운 웹 서버
- Spring Boot (Java): 대규모 프로젝트에 적합
- .NET (C#): 마이크로소프트 환경에 최적화

### 캐시 & 성능 최적화

: 서버의 응답 속도를 높이기 위한 기술들

- Redis: 자주 사용되는 데이터를 메모리에 저장하여 빠르게 제공
- CDN: 정적 파일을 여러 지역에서 빠르게 제공
- Load Balancer: 서버 부하를 여러 대의 서버로 분산