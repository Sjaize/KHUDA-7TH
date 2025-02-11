## 머신러닝과 딥러닝의 차이

- 머신러닝: 데이터를 분석하고, 데이터로부터 학습한 다음, 학습한 것을 적용해 정보에 입각한 결정을 내리는 알고리즘을 포함하는 것
    - 주어진 데이터를 사용하여 기능 수행, 시간이 지남에 따라 기능이 점차적으로 향상됨
    - 일반적으로 새로운 데이터가 유입됨에 따라 특정 기능을 수행하는 데 점점 능숙해지만, 여전히 인간의 개입 필요
- 딥러닝: 알고리즘을 계층으로 구성하여 자체적으로 배우고, 똑똑한 결정을 내릴 수 있는 인공신경망을 만드는 딥러닝의 하위 분야
    - 인공신경망이라는 계층화된 알고리즘 구조를 사용
    - 알고리즘이 자체 신경망을 통해 예측의 정확성 여부를 스스로 판단할 수 있어, 인간의 도움이 필요하지 않음
- 인공지능 ⊃ 머신러닝 ⊃ 딥러닝

## 경사하강법에서 최적화 방법

- 기본적인 경사하강법 최적화
    
    $\theta = \theta - \lambda \nabla _\theta J(\theta)$
    
    - SDG(확률적 경사하강법)
    - 현재 위치에서 기울어진 방향이 전체적인 최솟값과 다른 방향을 가리키므로 지그재그 모양으로 탐색하게 됨
- Momentum
    
    $v=\alpha v-\lambda \frac{\partial L}{\partial W}$
    
    $W=W+v$
    
    - $v$: 속도
    - 최종적으로는 지그재그 정도가 덜하고 물체가 구르는 듯한 모습이 나오게 된다
- AdaGrad
    
    $h = h+\frac{\partial L}{\partial W}\odot\frac{\partial L}{\partial W}$
    
    $W=W-\lambda \frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}$
    
    - $h$: 기울기 값을 제곱한 값을 더하여 학습률 조정하기 위함
    - $\odot$: 행렬의 원소별 곱셈
    - 처음에는 크게 학습하다가 점점 학습 정도를 줄여가게 됨
- RMSProp
    
    $h_i=ph_{i-1}+(1-p)\frac{\partial L_i}{\partial W}\odot\frac{\partial L_i}{\partial W}$
    
    $W=W-\lambda \frac{1}{\sqrt{h}}\frac{\partial L}{\partial W}$
    
    - AdaGrad는 과거의 기울기를 제곱하여 더한다
        - 갱신 정도가 점점 약해짐
        - 무한히 학습한다면 갱신량이 0
    - 먼 과거의 기울기는 조금 반영, 최신의 기울기는 많이 반영하는 방법
    - 지수이동평균(EMA, Exponential Moving Average)
- Adam
    
    $m_t=\beta_1m_{t-1}+(1-\beta_1)\nabla f(x_{t-1})$
    
    $g_t=\beta_2g_{t-1}+(1-\beta_2)(\nabla f(x_{t-1}))^2$
    
    $\hat m_t=\frac{m_t}{1-\beta_1^t}$
    
    $\hat g_t=\frac{g_t}{1-\beta^t_2}$
    
    $x_t=x_{t-1}-\lambda\frac{1}{\sqrt{\hat g_t+\epsilon}}\hat m_t$
    
    - $\beta_1$: Momentum의 지수이동평균 (0.9)
    - $\beta_2$: RMSProp의 지수이동평균 (0.999)
    - $\hat m, \hat g$: 학습 초기 시 $m_t, g_t$가 $0$이 되는 것을 방지하기 위한 보정값
    - $\epsilon$: 분모가 $0$이 되는 것을 방지하기 위한 작은 값($10^{-8}$)
    - 학습의 방향과 크기 모두 개선된다