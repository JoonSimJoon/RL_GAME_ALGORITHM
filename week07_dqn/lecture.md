# Week 7: Deep Q-Network (DQN)

## 목차
1. [복습: Q-Learning의 한계](#1-복습-q-learning의-한계)
2. [함수 근사 (Function Approximation)](#2-함수-근사-function-approximation)
3. [신경망 기초](#3-신경망-기초)
4. [PyTorch 기본 사용법](#4-pytorch-기본-사용법)
5. [DQN 핵심 기법 1: Experience Replay](#5-dqn-핵심-기법-1-experience-replay)
6. [DQN 핵심 기법 2: Target Network](#6-dqn-핵심-기법-2-target-network)
7. [DQN 전체 알고리즘](#7-dqn-전체-알고리즘)
8. [CartPole 환경 소개](#8-cartpole-환경-소개)
9. [DQN 하이퍼파라미터](#9-dqn-하이퍼파라미터)
10. [Betris에 DQN 적용하기](#10-betris에-dqn-적용하기)
11. [핵심 정리](#11-핵심-정리)
12. [다음 주 예고: Policy Gradient](#12-다음-주-예고-policy-gradient)

---

## 1. 복습: Q-Learning의 한계

### 1.1 지난 주 복습

지난 주에는 Q-Learning을 배웠습니다:
- Q(s, a): 상태 s에서 행동 a를 했을 때 기대되는 총 보상
- Q-table: 모든 (상태, 행동) 쌍의 Q값을 저장하는 표
- Bellman 방정식: Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]

### 1.2 Q-Learning이 잘 작동하는 경우

Q-Learning은 **상태 공간과 행동 공간이 작을 때** 매우 효과적입니다:

**예시: GridWorld (5×5)**
- 상태 수: 25개 (각 칸)
- 행동 수: 4개 (상, 하, 좌, 우)
- Q-table 크기: 25 × 4 = 100개 셀 → 메모리에 저장 가능 ✅

**예시: Tic-Tac-Toe (3×3)**
- 상태 수: 약 5,478개 (게임 이론상 가능한 보드 상태)
- 행동 수: 최대 9개
- Q-table 크기: ~50,000개 셀 → 여전히 관리 가능 ✅

### 1.3 Q-Learning의 심각한 한계

하지만 **상태 공간이 커지면** Q-Learning은 사용할 수 없습니다:

#### 예시 1: ATAXX (7×7)
- 보드: 7×7 = 49칸
- 각 칸의 상태: 빈 칸(0), 흰색(1), 검은색(2) → 3가지
- 가능한 보드 상태 수: 3^49 ≈ **2경 (2×10^23)**

이것이 얼마나 큰 수인가?
- 1초에 1억 개 상태를 처리해도 **600억 년** 걸림
- 우주의 나이(138억 년)보다 훨씬 김
- Q-table을 메모리에 저장하는 것조차 불가능

#### 예시 2: 바둑 (19×19)
- 가능한 상태 수: 약 10^170
- 우주의 원자 개수(10^80)의 제곱보다 많음
- Q-table 사용 **완전히 불가능**

#### 예시 3: Atari 게임 (Breakout, Pong 등)
- 화면: 210×160 픽셀, 각 픽셀 256가지 색상
- 가능한 화면 수: 256^(210×160) ≈ **무한대**
- Q-table로는 절대 해결 불가능

### 1.4 근본적인 문제

Q-Learning의 Q-table은 다음과 같은 한계가 있습니다:

1. **메모리 문제**: 상태가 많으면 Q-table을 저장할 수 없음
2. **일반화 불가능**: 본 적 없는 상태에 대해 추론할 수 없음
3. **학습 비효율**: 각 상태를 개별적으로 방문해야 함

**핵심 질문**: 본 적 없는 상태에 대해서도 Q값을 추정할 수 있는 방법은 없을까?

---

## 2. 함수 근사 (Function Approximation)

### 2.1 핵심 아이디어

Q-table을 버리고, **함수**로 Q값을 근사합니다:

```
Q-table 방식:
Q(s, a) = 표에서 찾아보기

함수 근사 방식:
Q(s, a; θ) = 함수로 계산하기
```

여기서:
- θ (세타): 함수의 **파라미터** (조절 가능한 값들)
- Q(s, a; θ): 파라미터 θ로 정의된 함수

### 2.2 실생활 비유

**Q-table 방식** = 모든 질문과 답을 외우기
- "2 + 3 = 5"
- "7 + 9 = 16"
- "123 + 456 = 579"
- 모든 경우를 일일이 외워야 함

**함수 근사 방식** = 덧셈 규칙을 배우기
- 덧셈 규칙을 한 번 배우면
- 본 적 없는 "9876 + 1234"도 계산 가능
- **일반화 능력**

### 2.3 왜 신경망인가?

함수를 근사하는 방법은 여러 가지가 있습니다:
- 선형 함수: Q(s,a) = w₁·s₁ + w₂·s₂ + ... + b
- 다항식 함수: Q(s,a) = w₁·s₁² + w₂·s₁·s₂ + ...
- **신경망**: 복잡한 비선형 패턴을 학습 가능 ⭐

신경망의 장점:
1. **범용 근사 정리(Universal Approximation Theorem)**: 충분히 큰 신경망은 어떤 함수든 근사 가능
2. **자동 특징 추출**: 중요한 패턴을 스스로 발견
3. **확장성**: 고차원 입력(이미지 등)에도 효과적

### 2.4 DQN의 탄생

**DQN (Deep Q-Network)**:
- 2013년 DeepMind가 발표
- 신경망으로 Q 함수를 근사
- Atari 게임 49개 중 29개에서 인간 수준 달성
- 강화학습의 획기적인 돌파구

핵심 공식:
```
Q(s, a) ≈ Q(s, a; θ)

θ: 신경망의 가중치(weights)와 편향(biases)
```

### 2.5 함수 근사의 이점

#### 이점 1: 일반화
```python
# Q-table: 정확히 본 상태만 알 수 있음
Q_table[(5, 3, 'state_A')] = 10.5
Q_table[(5, 4, 'state_A')] = ???  # 모름

# 함수 근사: 비슷한 상태에 비슷한 값
Q_network(state=[5, 3, ...]) = 10.5
Q_network(state=[5, 4, ...]) ≈ 10.3  # 유사한 값 추론
```

#### 이점 2: 메모리 효율성
```
Q-table: 10^23개 값 저장 필요 (불가능)
신경망: 파라미터 θ 몇 만 개만 저장 (가능)
```

#### 이점 3: 학습 효율성
```
Q-table: 각 상태를 직접 방문해야 학습
신경망: 한 상태 학습 → 비슷한 상태들도 함께 개선
```

---

## 3. 신경망 기초

### 3.1 뉴런 (Neuron)

신경망의 기본 단위는 **뉴런**입니다:

```
입력: x₁, x₂, x₃
가중치: w₁, w₂, w₃
편향: b

뉴런 출력 = f(w₁·x₁ + w₂·x₂ + w₃·x₃ + b)
```

여기서 f는 **활성화 함수**입니다.

#### 예시
```python
x = [0.5, 0.3, 0.8]  # 입력
w = [0.2, -0.4, 0.6]  # 가중치
b = 0.1              # 편향

# 가중합 계산
z = 0.2*0.5 + (-0.4)*0.3 + 0.6*0.8 + 0.1
  = 0.1 - 0.12 + 0.48 + 0.1
  = 0.56

# ReLU 활성화 함수 적용
output = max(0, z) = 0.56
```

### 3.2 활성화 함수

활성화 함수는 뉴런의 출력을 **비선형**으로 만듭니다.

#### ReLU (Rectified Linear Unit) - 가장 많이 사용
```python
ReLU(x) = max(0, x)

예시:
ReLU(-5) = 0
ReLU(0) = 0
ReLU(3) = 3
```

그래프:
```
      |
    3 |       /
    2 |      /
    1 |     /
    0 |____/
      |
   -5 -3 -1 0 1 2 3
```

#### 왜 비선형 함수가 필요한가?

선형 함수만 사용하면:
```
Layer 1: y = W₁·x + b₁
Layer 2: z = W₂·y + b₂
        = W₂·(W₁·x + b₁) + b₂
        = (W₂·W₁)·x + (W₂·b₁ + b₂)
        = W₃·x + b₃  (결국 선형 함수)
```

아무리 많은 층을 쌓아도 **선형 함수**가 됩니다.
비선형 활성화 함수를 사용해야 **복잡한 패턴**을 학습할 수 있습니다.

### 3.3 레이어 (Layer)

여러 뉴런을 모아서 **레이어**를 만듭니다:

```
입력 레이어:   [x₁, x₂, x₃]
              ↓  ↓  ↓
은닉 레이어 1: [h₁, h₂, h₃, h₄]
              ↓  ↓  ↓  ↓
은닉 레이어 2: [h₅, h₆, h₇]
              ↓  ↓  ↓
출력 레이어:   [y₁, y₂]
```

### 3.4 순전파 (Forward Pass)

입력부터 출력까지 값을 계산하는 과정:

```python
# Layer 1
h1 = ReLU(W1 @ x + b1)

# Layer 2
h2 = ReLU(W2 @ h1 + b2)

# Output
output = W3 @ h2 + b3
```

여기서 `@`는 행렬 곱셈입니다.

### 3.5 역전파 (Backpropagation)

신경망을 학습시키는 핵심 알고리즘입니다.

#### 목표
손실 함수 L을 최소화하는 파라미터 θ를 찾기:
```
L = (예측값 - 실제값)²
```

#### 과정
1. **순전파**: 입력 → 출력 계산
2. **손실 계산**: L = (output - target)²
3. **역전파**: 출력 ← 입력 방향으로 그래디언트 계산
4. **파라미터 업데이트**: θ ← θ - α·∇L

#### 그래디언트 (Gradient)
```
∇L = [∂L/∂w₁, ∂L/∂w₂, ..., ∂L/∂b₁, ∂L/∂b₂, ...]

각 파라미터가 손실에 얼마나 영향을 미치는지 나타냄
```

### 3.6 경사하강법 (Gradient Descent)

파라미터를 조금씩 개선하는 방법:

```python
for epoch in range(1000):
    # 1. 순전파
    output = model(input)

    # 2. 손실 계산
    loss = (output - target) ** 2

    # 3. 그래디언트 계산 (역전파)
    loss.backward()

    # 4. 파라미터 업데이트
    for param in model.parameters():
        param.data -= learning_rate * param.grad
        param.grad.zero_()  # 그래디언트 초기화
```

#### 학습률 (Learning Rate)
```
θ_new = θ_old - α·∇L

α: 학습률 (얼마나 크게 움직일 것인가)

α가 너무 크면: 발산 (overshooting)
α가 너무 작으면: 학습 너무 느림
적절한 값: 0.001, 0.0001 등
```

---

## 4. PyTorch 기본 사용법

### 4.1 PyTorch란?

- Facebook(Meta)에서 개발한 딥러닝 프레임워크
- 직관적이고 사용하기 쉬움
- 동적 계산 그래프 → 디버깅 편함
- 강화학습 연구에서 널리 사용

### 4.2 텐서 (Tensor)

PyTorch의 기본 데이터 구조:

```python
import torch

# 스칼라
x = torch.tensor(5.0)

# 벡터
x = torch.tensor([1.0, 2.0, 3.0])

# 행렬
x = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])

# 3D 텐서
x = torch.tensor([[[1, 2], [3, 4]],
                  [[5, 6], [7, 8]]])
```

### 4.3 신경망 정의하기

```python
import torch.nn as nn

class SimpleNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # 레이어 정의
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 순전파 정의
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 네트워크 생성
model = SimpleNetwork(input_size=4, hidden_size=128, output_size=2)
```

#### nn.Linear란?
```python
fc = nn.Linear(in_features=3, out_features=2)

# 내부적으로:
# weight: (2, 3) 행렬
# bias: (2,) 벡터
# output = x @ weight.T + bias
```

### 4.4 Q-Network 구현

DQN에서 사용할 Q-Network:

```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # 출력층은 활성화 함수 없음
```

#### 입력과 출력
```python
# CartPole 예시
state_size = 4   # [위치, 속도, 각도, 각속도]
action_size = 2  # [왼쪽, 오른쪽]

q_net = QNetwork(4, 2)

# 사용 예시
state = torch.tensor([0.1, -0.5, 0.2, 0.3])
q_values = q_net(state)  # [Q(s, 왼쪽), Q(s, 오른쪽)]
# 출력 예: tensor([2.3, 1.8])
```

### 4.5 최적화기 (Optimizer)

파라미터를 업데이트하는 도구:

```python
import torch.optim as optim

# Adam 옵티마이저 (가장 많이 사용)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습 루프
for epoch in range(100):
    # 1. 예측
    output = model(input)

    # 2. 손실 계산
    loss = criterion(output, target)

    # 3. 그래디언트 초기화
    optimizer.zero_grad()

    # 4. 역전파
    loss.backward()

    # 5. 파라미터 업데이트
    optimizer.step()
```

#### 주요 옵티마이저
- **SGD**: 기본 경사하강법
- **Adam**: 적응적 학습률 (가장 인기)
- **RMSprop**: DQN 원 논문에서 사용

### 4.6 손실 함수 (Loss Function)

DQN에서는 **MSE (Mean Squared Error)** 사용:

```python
criterion = nn.MSELoss()

predicted = torch.tensor([2.5, 3.0])
target = torch.tensor([2.0, 3.5])

loss = criterion(predicted, target)
# = ((2.5-2.0)² + (3.0-3.5)²) / 2
# = (0.25 + 0.25) / 2
# = 0.25
```

### 4.7 그래디언트 자동 계산

PyTorch의 강력한 기능:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1

# 역전파
y.backward()

# 그래디언트 확인
print(x.grad)  # dy/dx = 2x + 3 = 2*2 + 3 = 7
```

신경망에서:
```python
# 자동으로 모든 파라미터의 그래디언트 계산
loss.backward()

# 각 파라미터의 그래디언트 확인
for name, param in model.named_parameters():
    print(f"{name}: {param.grad}")
```

---

## 5. DQN 핵심 기법 1: Experience Replay

### 5.1 문제: 연속된 경험의 상관관계

Q-Learning에서는 경험을 즉시 사용했습니다:
```python
s, a, r, s' = env.step(action)
Q[s, a] += α * (r + γ * max(Q[s']) - Q[s, a])  # 즉시 업데이트
```

신경망에서도 같은 방식을 사용하면?
```python
# Episode 1
state1 -> action1 -> reward1 -> state2  # 학습
state2 -> action2 -> reward2 -> state3  # 학습
state3 -> action3 -> reward3 -> state4  # 학습
```

**문제점**:
1. **강한 상관관계**: 연속된 상태들은 매우 비슷함
2. **학습 불안정**: 신경망이 최근 경험에만 과적합
3. **재앙적 망각**: 이전에 학습한 내용을 잊어버림

#### 실제 예시
```
CartPole에서:
state_t   = [0.1, 0.5, 0.01, 0.1]
state_t+1 = [0.11, 0.48, 0.011, 0.09]  # 거의 동일!
```

이런 비슷한 데이터로만 학습하면:
- 신경망이 특정 상황에만 과적합
- 다양한 상황을 일반화하지 못함

### 5.2 해결책: Experience Replay

**핵심 아이디어**: 경험을 저장했다가 나중에 **랜덤 샘플링**해서 학습

```python
# 1. 경험 저장
replay_buffer.store(state, action, reward, next_state, done)

# 2. 랜덤 샘플링
batch = replay_buffer.sample(batch_size=32)

# 3. 샘플링된 경험으로 학습
for (s, a, r, s', done) in batch:
    # Q-learning 업데이트
    ...
```

### 5.3 Replay Buffer 구현

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        # 경험 저장
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        # 랜덤 샘플링
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
```

#### deque란?
- Double-Ended Queue
- 양쪽에서 삽입/삭제가 빠름
- maxlen 설정 시 자동으로 오래된 항목 삭제

```python
buffer = deque(maxlen=3)
buffer.append(1)  # [1]
buffer.append(2)  # [1, 2]
buffer.append(3)  # [1, 2, 3]
buffer.append(4)  # [2, 3, 4]  ← 1이 자동 삭제
```

### 5.4 Experience Replay의 장점

#### 장점 1: 데이터 효율성
```
일반 Q-Learning: 각 경험을 1번만 사용
Experience Replay: 각 경험을 여러 번 재사용
```

#### 장점 2: 상관관계 제거
```python
# 랜덤 샘플링된 배치
batch = [
    (state_100, action_100, ...),  # Episode 5
    (state_5,   action_5,   ...),  # Episode 1
    (state_237, action_237, ...),  # Episode 12
    (state_42,  action_42,  ...),  # Episode 2
]
# 다양한 에피소드의 경험이 섞임 → 상관관계 감소
```

#### 장점 3: 안정적 학습
```
순차 학습: 신경망이 요동침
랜덤 학습: 신경망이 안정적으로 수렴
```

### 5.5 하이퍼파라미터

#### Buffer Size (버퍼 크기)
```python
buffer = ReplayBuffer(capacity=10000)

너무 작으면 (100):
  - 다양성 부족
  - 최근 경험에 과적합

너무 크면 (1000000):
  - 메모리 부족
  - 오래된 잘못된 경험도 학습

적절한 값: 10000 ~ 100000
```

#### Batch Size (배치 크기)
```python
batch = buffer.sample(batch_size=32)

너무 작으면 (4):
  - 학습 불안정
  - 그래디언트 노이즈 큼

너무 크면 (512):
  - 메모리 많이 사용
  - 학습 속도 느림

적절한 값: 32, 64, 128
```

### 5.6 실제 사용 예시

```python
# 초기화
replay_buffer = ReplayBuffer(capacity=10000)

# Episode 실행
for episode in range(1000):
    state = env.reset()

    for t in range(max_steps):
        # 행동 선택
        action = agent.select_action(state)

        # 환경 실행
        next_state, reward, done = env.step(action)

        # 경험 저장
        replay_buffer.store(state, action, reward, next_state, done)

        # 충분한 경험이 쌓이면 학습
        if replay_buffer.size() > batch_size:
            batch = replay_buffer.sample(batch_size)
            agent.learn(batch)

        state = next_state
        if done:
            break
```

---

## 6. DQN 핵심 기법 2: Target Network

### 6.1 문제: 움직이는 목표

Q-Learning 업데이트:
```
Q(s,a) ← Q(s,a) + α[r + γ·max_a' Q(s',a') - Q(s,a)]
         └─────┘     └──────────────┘
         현재 값          목표 값
```

신경망 버전:
```
Loss = (Q(s,a;θ) - [r + γ·max_a' Q(s',a';θ)])²
        └──────┘     └────────────────────┘
        예측값              목표값
```

**문제**: 예측값과 목표값 **둘 다** θ에 의존
- θ를 업데이트하면 예측값 변함 ✓
- θ를 업데이트하면 목표값도 변함 ✗

이는 마치 **움직이는 과녁을 맞추는 것**과 같습니다.

#### 구체적 예시
```python
# Step 1
Q(s,a;θ) = 5.0
target = r + γ·max Q(s',a';θ) = 6.0
Loss = (5.0 - 6.0)² = 1.0

# θ 업데이트

# Step 2
Q(s,a;θ) = 5.3  # 목표 6.0에 가까워짐
target = r + γ·max Q(s',a';θ) = 7.2  # 목표도 변함!
Loss = (5.3 - 7.2)² = 3.61  # Loss가 오히려 증가?
```

이런 현상을 **부트스트랩 문제**라고 합니다.

### 6.2 해결책: Target Network

**핵심 아이디어**: 목표값 계산용 별도 네트워크 사용

```python
# 메인 네트워크 (θ): 예측 및 학습
Q(s, a; θ)

# 타겟 네트워크 (θ⁻): 목표값 계산만
Q(s', a'; θ⁻)

# Loss
Loss = (Q(s,a;θ) - [r + γ·max_a' Q(s',a';θ⁻)])²
```

### 6.3 Target Network 업데이트

타겟 네트워크는 **천천히** 업데이트합니다:

```python
# 방법 1: 주기적 Hard Update
if step % target_update_freq == 0:
    θ⁻ = θ  # 메인 네트워크를 그대로 복사

# 방법 2: Soft Update (더 부드러움)
τ = 0.001  # 작은 값
θ⁻ = τ·θ + (1-τ)·θ⁻  # 조금씩 섞기
```

DQN 원 논문에서는 **Hard Update** 사용:
```python
target_update_freq = 1000  # 1000 step마다 업데이트
```

### 6.4 구현 예시

```python
import copy

class DQNAgent:
    def __init__(self, state_size, action_size):
        # 메인 네트워크
        self.q_network = QNetwork(state_size, action_size)

        # 타겟 네트워크 (메인 네트워크 복사)
        self.target_network = copy.deepcopy(self.q_network)

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=0.001)
        self.step_count = 0

    def learn(self, batch):
        states, actions, rewards, next_states, dones = batch

        # 현재 Q값 (메인 네트워크)
        current_q = self.q_network(states).gather(1, actions)

        # 목표 Q값 (타겟 네트워크)
        with torch.no_grad():  # 그래디언트 계산 안 함
            next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + 0.99 * next_q * (1 - dones)

        # Loss 계산 및 업데이트
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 타겟 네트워크 업데이트
        self.step_count += 1
        if self.step_count % 1000 == 0:
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )
```

### 6.5 왜 효과적인가?

#### 안정성
```
타겟 네트워크 없음:
Episode 1: 목표 = 10
Episode 2: 목표 = 15
Episode 3: 목표 = 8
Episode 4: 목표 = 20  ← 요동침

타겟 네트워크 사용:
Episode 1-1000: 목표 = 10 (고정)
Episode 1001-2000: 목표 = 12 (고정)
Episode 2001-3000: 목표 = 13 (고정)  ← 안정적
```

#### 수렴성
```
움직이는 목표: 수렴하기 어려움
고정된 목표: 수렴 가능 → 일정 시간 후 목표 업데이트
```

### 6.6 하이퍼파라미터: Target Update Frequency

```python
target_update_freq = 100   # 너무 빈번
target_update_freq = 1000  # 적당 (DQN 논문)
target_update_freq = 10000 # 너무 느림

너무 빈번: 목표가 자주 바뀜 → 불안정
너무 느림: 최신 정보 반영 안 됨 → 학습 느림
```

---

## 7. DQN 전체 알고리즘

### 7.1 의사 코드

```
초기화:
  - Replay Buffer D (capacity = 10000)
  - Q-network: Q(s,a;θ) with random weights θ
  - Target network: Q(s,a;θ⁻) with θ⁻ = θ
  - Optimizer: Adam with lr = 0.001

for episode = 1 to M:
    state = env.reset()

    for t = 1 to T:
        # 1. ε-greedy 행동 선택
        if random() < ε:
            action = random_action()
        else:
            action = argmax_a Q(state, a; θ)

        # 2. 환경 실행
        next_state, reward, done = env.step(action)

        # 3. 경험 저장
        D.store(state, action, reward, next_state, done)

        # 4. 학습 (버퍼가 충분히 찬 경우)
        if D.size() >= batch_size:
            # 4-1. 미니배치 샘플링
            batch = D.sample(batch_size)

            # 4-2. 목표값 계산 (타겟 네트워크)
            for (s, a, r, s', done) in batch:
                if done:
                    y = r
                else:
                    y = r + γ·max_a' Q(s', a'; θ⁻)

            # 4-3. Loss 계산
            Loss = Σ (Q(s, a; θ) - y)²

            # 4-4. 그래디언트 하강
            θ ← θ - α·∇_θ Loss

        # 5. 타겟 네트워크 업데이트
        if t % C == 0:
            θ⁻ ← θ

        state = next_state

        if done:
            break

    # 6. ε 감소
    ε = max(ε_min, ε * ε_decay)
```

### 7.2 핵심 컴포넌트

#### 1. Q-Network
```python
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

#### 2. Replay Buffer
```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
```

#### 3. DQN Agent
```python
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = optim.Adam(self.q_network.parameters())
        self.replay_buffer = ReplayBuffer(capacity=10000)

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, action_size - 1)
        else:
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values.argmax().item()

    def learn(self, batch_size, gamma):
        if self.replay_buffer.size() < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        # ... (학습 로직)
```

### 7.3 전체 학습 루프

```python
# 하이퍼파라미터
num_episodes = 500
max_steps = 200
batch_size = 32
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 1000

# 초기화
agent = DQNAgent(state_size=4, action_size=2)
env = gym.make('CartPole-v1')
epsilon = epsilon_start
step_count = 0

# 학습
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for t in range(max_steps):
        # 행동 선택
        action = agent.select_action(state, epsilon)

        # 환경 실행
        next_state, reward, done, _ = env.step(action)

        # 경험 저장
        agent.replay_buffer.store(state, action, reward, next_state, done)

        # 학습
        agent.learn(batch_size, gamma)

        # 타겟 네트워크 업데이트
        step_count += 1
        if step_count % target_update_freq == 0:
            agent.target_network.load_state_dict(
                agent.q_network.state_dict()
            )

        episode_reward += reward
        state = next_state

        if done:
            break

    # ε 감소
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # 진행 상황 출력
    if episode % 10 == 0:
        print(f"Episode {episode}, Reward: {episode_reward}, ε: {epsilon:.3f}")
```

---

## 8. CartPole 환경 소개

### 8.1 CartPole이란?

CartPole은 강화학습의 "Hello World"와 같은 환경입니다.

**목표**: 막대가 쓰러지지 않도록 카트를 좌우로 움직이기

```
         |
        /|\
       / | \
      /  |  \
     막대(pole)

  ┌─────────┐
  │  카트   │
  └─────────┘
 ══════════════ 트랙
```

### 8.2 상태 공간

4차원 연속 상태:
```python
state = [x, x_dot, theta, theta_dot]

x: 카트의 위치 (-4.8 ~ 4.8)
x_dot: 카트의 속도 (-∞ ~ ∞)
theta: 막대의 각도 (-0.418 rad ~ 0.418 rad ≈ ±24°)
theta_dot: 막대의 각속도 (-∞ ~ ∞)
```

예시:
```python
state = [0.02, -0.5, 0.1, 0.8]
# 카트가 오른쪽(0.02)에 있고
# 왼쪽으로 움직이며(-0.5)
# 막대가 오른쪽으로 기울어져(0.1)
# 오른쪽으로 쓰러지는 중(0.8)
```

### 8.3 행동 공간

2개의 이산 행동:
```python
action = 0  # 왼쪽으로 밀기
action = 1  # 오른쪽으로 밀기
```

### 8.4 보상

```python
reward = +1  # 매 타임스텝마다

목표: 가능한 한 오래 막대를 세우기
최대 점수: 500 (500 타임스텝)
```

### 8.5 종료 조건

다음 중 하나가 발생하면 에피소드 종료:
1. 막대 각도가 ±12° 초과
2. 카트 위치가 ±2.4 초과
3. 200 스텝 달성 (v0) 또는 500 스텝 (v1)

### 8.6 사용 방법

```python
import gymnasium as gym

# 환경 생성
env = gym.make('CartPole-v1')

# 초기화
state = env.reset()  # 초기 상태 반환

# 한 스텝 실행
action = 1  # 오른쪽
next_state, reward, done, truncated, info = env.step(action)

# 종료 확인
if done or truncated:
    print("Episode finished!")
```

### 8.7 성능 기준

```
초보: 평균 50 이하
중급: 평균 100~150
고급: 평균 200 이상 (안정적으로 해결)
```

DQN으로 학습하면:
- 100 에피소드: 평균 50
- 200 에피소드: 평균 150
- 300 에피소드: 평균 200 (해결!)

---

## 9. DQN 하이퍼파라미터

### 9.1 주요 하이퍼파라미터

DQN의 성능은 하이퍼파라미터에 크게 의존합니다.

#### 1. Learning Rate (학습률)
```python
lr = 0.001   # 일반적 값
lr = 0.0001  # 더 안정적이지만 느림
lr = 0.01    # 빠르지만 불안정

적절한 값: 0.0001 ~ 0.001
```

#### 2. Gamma (할인율)
```python
gamma = 0.99   # 장기 보상 중시
gamma = 0.9    # 단기 보상 중시
gamma = 0.999  # 매우 장기적 계획

적절한 값: 0.95 ~ 0.99
```

#### 3. Epsilon (탐험률)
```python
epsilon_start = 1.0    # 초기: 완전 탐험
epsilon_end = 0.01     # 최종: 1% 탐험
epsilon_decay = 0.995  # 감소율

# 예시
ε = 1.0 → 0.995 → 0.990 → ... → 0.01
```

#### 4. Replay Buffer Size
```python
buffer_size = 10000   # 작은 환경
buffer_size = 100000  # 중간 환경
buffer_size = 1000000 # Atari (큰 환경)

트레이드오프:
- 크면: 다양한 경험, 메모리 많이 사용
- 작으면: 최신 경험 위주, 메모리 절약
```

#### 5. Batch Size
```python
batch_size = 32    # 일반적
batch_size = 64    # 더 안정적
batch_size = 128   # 메모리 여유 있을 때

트레이드오프:
- 크면: 안정적, 느림
- 작으면: 빠름, 불안정
```

#### 6. Target Update Frequency
```python
target_update_freq = 100    # 빈번 업데이트
target_update_freq = 1000   # DQN 논문 기본값
target_update_freq = 10000  # 느린 업데이트

트레이드오프:
- 빈번: 최신 정보, 불안정
- 느림: 안정적, 오래된 정보
```

### 9.2 하이퍼파라미터 튜닝 전략

#### 단계 1: 기본값으로 시작
```python
# 검증된 기본값
lr = 0.001
gamma = 0.99
epsilon_decay = 0.995
buffer_size = 10000
batch_size = 32
target_update_freq = 1000
```

#### 단계 2: 한 번에 하나씩 변경
```python
# 잘못된 방법
lr = 0.0001  # 변경
batch_size = 64  # 변경
epsilon_decay = 0.99  # 변경
# → 어떤 것이 영향을 미쳤는지 모름

# 올바른 방법
실험 1: lr = 0.0001 (나머지 고정)
실험 2: lr = 0.001 (기본)
실험 3: lr = 0.01
# → lr의 영향을 명확히 파악
```

#### 단계 3: 학습 곡선 관찰
```python
# 좋은 학습 곡선
Reward: 50 → 100 → 150 → 200  ✓

# 나쁜 학습 곡선
Reward: 50 → 80 → 40 → 90 → 60  ✗ (불안정)
Reward: 50 → 50 → 50 → 50  ✗ (학습 안 됨)
```

### 9.3 CartPole 권장 설정

```python
# CartPole에 최적화된 하이퍼파라미터
config = {
    'lr': 0.001,
    'gamma': 0.99,
    'epsilon_start': 1.0,
    'epsilon_end': 0.01,
    'epsilon_decay': 0.995,
    'buffer_size': 10000,
    'batch_size': 32,
    'target_update_freq': 1000,
    'num_episodes': 500,
    'max_steps': 200
}
```

### 9.4 일반적인 문제와 해결책

#### 문제 1: 학습이 안 됨
```
증상: 보상이 계속 낮음
원인:
  - Learning rate가 너무 낮음
  - Epsilon이 너무 낮음 (탐험 부족)
  - Network가 너무 작음

해결:
  - lr을 높이기 (0.0001 → 0.001)
  - epsilon_decay를 느리게 (0.99 → 0.995)
  - Hidden layer 크기 증가 (64 → 128)
```

#### 문제 2: 학습이 불안정함
```
증상: 보상이 요동침
원인:
  - Learning rate가 너무 높음
  - Batch size가 너무 작음
  - Target update가 너무 빈번

해결:
  - lr을 낮추기 (0.01 → 0.001)
  - batch_size 증가 (16 → 32)
  - target_update_freq 증가 (100 → 1000)
```

#### 문제 3: 과적합
```
증상: 특정 상황에서만 잘함
원인:
  - Buffer size가 너무 작음
  - Epsilon이 너무 빨리 감소

해결:
  - buffer_size 증가 (1000 → 10000)
  - epsilon_decay 느리게 (0.99 → 0.995)
```

---

## 10. Betris에 DQN 적용하기

### 10.1 Betris 게임 소개

Betris는 ALPHANO 문제 3번으로, 베팅과 블록 배치를 결합한 게임입니다.

**게임 규칙**:
- 5×5 보드
- 테트리스 블록을 배치
- 줄이 완성되면 제거 후 점수 획득
- 각 라운드마다 코인을 베팅
- 베팅 금액에 따라 점수 배수 증가

### 10.2 상태 공간 설계

Betris의 상태는 매우 복잡합니다:

```python
# 방법 1: 단순 표현
state = [
    board[0,0], board[0,1], ..., board[4,4],  # 25개 셀
    score,                                     # 현재 점수
    coins,                                     # 남은 코인
    current_block_type                         # 현재 블록 종류
]
# 총 28차원

# 방법 2: 더 자세한 표현
state = [
    board,           # 5×5 = 25
    score,           # 1
    coins,           # 1
    block_encoding,  # 블록 형태를 인코딩 (예: 10차원)
    lines_cleared,   # 지금까지 제거한 줄 수
    ...
]
```

### 10.3 행동 공간 설계

Betris는 **두 가지 결정**을 해야 합니다:

#### 1. 베팅 금액
```python
bet_actions = [0, 1, 2, 5, 10]  # 가능한 베팅 금액
```

#### 2. 블록 배치
```python
# 블록 배치 = (row, col, rotation)
# 예: I-block은 2가지 회전, L-block은 4가지 회전

총 행동 수 = 베팅 × 배치 위치 × 회전
           ≈ 5 × 25 × 4
           = 500개 (매우 많음)
```

#### 해결 방법: 계층적 행동 선택
```python
# 방법 1: 두 단계로 나누기
action_1 = select_bet(state)      # DQN 1
action_2 = select_placement(state, bet)  # DQN 2

# 방법 2: 휴리스틱 + DQN
bet = heuristic_bet(state)        # 규칙 기반
placement = dqn_select(state)     # DQN으로 배치만
```

### 10.4 보상 설계

```python
# 방법 1: 단순 보상
reward = score_gained * bet_multiplier

# 방법 2: 형태 보상 (Reward Shaping)
reward = 0
reward += 10 * lines_cleared              # 줄 제거 보너스
reward += 1 * empty_cells_remaining       # 빈 공간 유지
reward -= 5 if game_over else 0           # 게임 오버 페널티
reward += score_gained * bet_multiplier   # 실제 점수
```

### 10.5 네트워크 구조

```python
class BetrisQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()

        # 보드를 CNN으로 처리
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Flatten + FC layers
        self.fc1 = nn.Linear(64 * 5 * 5 + 3, 256)  # +3 for score, coins, block
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, board, meta):
        # board: (batch, 1, 5, 5)
        # meta: (batch, 3) - [score, coins, block]

        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten

        x = torch.cat([x, meta], dim=1)  # Concatenate
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

### 10.6 실전 고려사항

#### 1. 학습 시간
```
Betris는 CartPole보다 훨씬 복잡
CartPole: 500 에피소드로 해결
Betris: 10000+ 에피소드 필요 가능
```

#### 2. 탐험 전략
```python
# ε-greedy만으로 부족할 수 있음
# 추가 전략:

# 1. Curiosity-driven exploration
reward += bonus_for_new_states

# 2. Prioritized Experience Replay
# 중요한 경험(큰 reward)을 더 자주 샘플링

# 3. Noisy Networks
# 네트워크 파라미터에 노이즈 추가
```

#### 3. 베팅 전략
```python
# DQN 학습 초기: 보수적 베팅
def safe_bet(coins, confidence):
    if confidence < 0.5:
        return min(1, coins)  # 최소 베팅
    else:
        return min(5, coins)  # 중간 베팅

# DQN이 학습되면 점차 공격적으로
```

### 10.7 단계별 개발 전략

#### 단계 1: 단순화된 버전
```python
# 베팅 고정, 배치만 학습
def fixed_bet():
    return 1

# 행동 = 배치 위치만
actions = [(row, col, rot) for row in range(5)
                            for col in range(5)
                            for rot in range(4)]
```

#### 단계 2: 베팅 추가
```python
# 배치는 휴리스틱, 베팅만 학습
def greedy_placement(board, block):
    # 가장 좋은 위치 찾기 (휴리스틱)
    return best_position

# DQN으로 베팅만 학습
```

#### 단계 3: 통합
```python
# 베팅 + 배치 모두 DQN
# 또는 Multi-agent: 각각 별도 DQN
```

---

## 11. 핵심 정리

### 11.1 DQN의 핵심 아이디어

1. **함수 근사**: Q-table → 신경망
2. **Experience Replay**: 경험 재사용, 상관관계 제거
3. **Target Network**: 안정적 학습 목표

### 11.2 DQN vs Q-Learning

| 항목 | Q-Learning | DQN |
|------|-----------|-----|
| Q 표현 | Table | Neural Network |
| 상태 공간 | 작음 | 큼 (무한대도 가능) |
| 일반화 | 불가능 | 가능 |
| 메모리 | O(S×A) | O(θ) (고정) |
| 학습 | 안정적 | 불안정 (해결 필요) |
| 경험 사용 | 1회 | 여러 번 (Replay) |

### 11.3 언제 DQN을 사용하는가?

#### DQN이 적합한 경우 ✓
- 상태 공간이 매우 큼 (예: 이미지)
- 연속적 상태 공간
- 비슷한 상태에 비슷한 행동 필요
- 행동 공간은 이산적이고 작음

#### DQN이 부적합한 경우 ✗
- 상태 공간이 작음 (Q-table으로 충분)
- 연속적 행동 공간 (Policy Gradient 사용)
- 실시간 빠른 결정 필요 (신경망은 느림)

### 11.4 DQN의 한계

1. **이산 행동만 가능**: 연속 행동 (예: 각도 0~360°) 불가능
2. **샘플 비효율적**: 많은 경험 필요
3. **과대평가**: max 연산으로 Q값 과대평가 경향
4. **하이퍼파라미터 민감**: 튜닝 필요

### 11.5 DQN 개선 기법들

DQN 이후 많은 개선 기법이 나왔습니다:

1. **Double DQN**: 과대평가 문제 해결
2. **Dueling DQN**: 네트워크 구조 개선
3. **Prioritized Experience Replay**: 중요한 경험 우선 학습
4. **Rainbow DQN**: 여러 기법 통합

### 11.6 실습 체크리스트

오늘 배운 내용을 실습해봅시다:

- [ ] PyTorch로 간단한 신경망 만들어보기
- [ ] Replay Buffer 구현하기
- [ ] CartPole DQN 전체 코드 실행하기
- [ ] 하이퍼파라미터 변경해보고 결과 비교하기
- [ ] Betris 상태/행동 공간 설계해보기

---

## 12. 다음 주 예고: Policy Gradient

### 12.1 DQN의 한계 재확인

DQN은 **이산 행동**만 가능합니다:
```python
# 가능
actions = [0, 1, 2]  # 왼쪽, 정지, 오른쪽

# 불가능
action = 0.73  # 연속값
action = (0.5, 0.3)  # 2차원 연속
```

### 12.2 연속 행동이 필요한 경우

#### 예시 1: 로봇 팔 제어
```python
# 관절 각도 (연속)
action = [30.5°, 45.2°, 60.8°]

# DQN으로 하려면?
action = [30°, 31°, 32°, ...]  # 이산화 필요
# → 너무 많은 행동 (360^3 = 4천만 개)
```

#### 예시 2: 자동차 운전
```python
# 핸들 각도: -180° ~ 180° (연속)
# 가속: 0 ~ 100% (연속)

# DQN으로는 부자연스러움
```

### 12.3 Policy Gradient 핵심 아이디어

DQN: Q(s,a) 학습 → argmax로 행동 선택
Policy Gradient: **정책 π(a|s) 직접 학습**

```python
# DQN
Q(s, a) → action = argmax_a Q(s, a)

# Policy Gradient
π(a|s; θ) → action ~ π(a|s; θ)  # 확률 분포에서 샘플링
```

### 12.4 다음 주 주제

1. **Policy Gradient 기초**
   - 정책 네트워크
   - REINFORCE 알고리즘

2. **Actor-Critic**
   - Value와 Policy를 함께 학습
   - Advantage 함수

3. **실습**
   - Pendulum (연속 행동)
   - 간단한 연속 제어 문제

### 12.5 준비 사항

- DQN 코드를 충분히 이해하기
- PyTorch 기본 문법 익히기
- 확률 분포 개념 복습 (정규분포 등)

---

## 부록: 참고 자료

### 논문
- Playing Atari with Deep Reinforcement Learning (DQN, 2013)
- Human-level control through deep reinforcement learning (Nature DQN, 2015)

### 온라인 자료
- OpenAI Spinning Up (spinningup.openai.com)
- PyTorch 공식 튜토리얼
- Gymnasium 문서 (gymnasium.farama.org)

### 연습 문제
1. Replay Buffer를 직접 구현해보세요
2. CartPole을 200점 이상 달성해보세요
3. 하이퍼파라미터를 바꿔가며 학습 곡선을 비교해보세요
4. Betris 상태 공간을 3가지 방법으로 설계해보세요

---

**다음 시간에는 CartPole DQN을 직접 구현하며 더 자세히 배웁니다!**
