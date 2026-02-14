# Week 6: Q-Learning과 SARSA - 모델 없이 학습하기

## 수업 목표
이번 주차에서는 환경의 전이확률을 모르는 상황에서 경험을 통해 학습하는 **Model-free 강화학습**을 배웁니다. Q-Learning과 SARSA 알고리즘을 이해하고 실제로 구현하여 에이전트가 스스로 학습하는 과정을 관찰합니다.

---

## 1. 복습: MDP와 벨만 방정식

### 1.1 지난 주 내용 되짚기

지난 주에는 **Value Iteration**과 **Policy Iteration**을 배웠습니다. 이 알고리즘들은 **환경의 모델을 알고 있을 때** 사용할 수 있는 방법이었습니다.

**환경의 모델이란?**
- **전이확률 P(s'|s,a)**: 상태 s에서 행동 a를 했을 때 다음 상태 s'로 갈 확률
- **보상함수 R(s,a,s')**: 각 전이에서 받는 보상

**벨만 최적 방정식 (복습)**:
```
V*(s) = max_a Σ_{s'} P(s'|s,a) [R(s,a,s') + γ·V*(s')]
```

Value Iteration은 이 벨만 방정식을 반복적으로 적용하여 최적 가치함수를 계산했습니다.

### 1.2 문제점: 현실 세계에서는 P를 알 수 없다

**현실의 문제들**:
- 바둑: 상대방이 어떤 수를 둘지 정확한 확률을 모름
- 로봇 제어: 모터가 정확히 어떻게 움직일지 물리 법칙을 완벽히 모름
- 게임: 적 AI의 행동 패턴을 정확히 모름
- 주식 거래: 시장의 다음 상태를 예측하는 확률 모델을 모름

**따라서 필요한 것**:
- 환경의 모델 없이 **경험을 통해 직접 학습**하는 방법
- 이것이 바로 **Model-free 강화학습**입니다

---

## 2. Model-based vs Model-free

### 2.1 비교표

| 구분 | Model-based | Model-free |
|------|-------------|------------|
| **환경 정보** | P(s'\|s,a), R(s,a) 필요 | 필요 없음 |
| **학습 방법** | 벨만 방정식 직접 계산 | 경험으로부터 학습 |
| **대표 알고리즘** | Value Iteration, Policy Iteration | Q-Learning, SARSA |
| **장점** | 빠르고 정확 (모델이 정확하면) | 모델 불필요, 범용적 |
| **단점** | 모델 필요, 복잡한 환경 어려움 | 많은 경험 필요, 수렴 느림 |
| **사용 사례** | GridWorld, FrozenLake (모델 제공) | 게임, 로봇, 실제 환경 |

### 2.2 Model-free의 핵심 아이디어

환경의 모델을 모르더라도, **실제로 행동하면서 얻는 경험**으로부터 학습할 수 있습니다.

**경험이란?**
```
(state, action, reward, next_state)
```

이 튜플을 여러 번 모으면, 평균적으로 어떤 행동이 좋은지 알 수 있습니다.

**예시**:
- 상태 A에서 행동 1을 10번 했더니 평균 보상이 5
- 상태 A에서 행동 2를 10번 했더니 평균 보상이 8
- → 행동 2가 더 좋다!

---

## 3. Q-Learning: 대표적인 Model-free 알고리즘

### 3.1 Q-함수란?

**Q-함수 (Action-Value Function)**:
```
Q(s, a) = 상태 s에서 행동 a를 했을 때 기대되는 총 보상
```

**가치함수 V(s)와의 차이**:
- V(s): 상태 s의 가치 (어떤 행동을 할지는 정해지지 않음)
- Q(s,a): 상태 s에서 특정 행동 a를 했을 때의 가치

**관계식**:
```
V*(s) = max_a Q*(s, a)
π*(s) = argmax_a Q*(s, a)
```

즉, Q-함수를 알면 최적 정책을 바로 구할 수 있습니다!

### 3.2 Q-Learning 업데이트 규칙

Q-Learning의 핵심은 **경험으로부터 Q-함수를 업데이트**하는 것입니다.

**업데이트 공식**:
```python
Q(s, a) ← Q(s, a) + α · [R + γ · max_a' Q(s', a') - Q(s, a)]
                          └──────────────────┬──────────────────┘
                                        TD Error (시간차 오차)
```

**각 항목의 의미**:
- **Q(s, a)**: 현재 추정값
- **α**: 학습률 (learning rate, 0 < α ≤ 1)
- **R**: 실제로 받은 보상
- **γ**: 할인율 (discount factor, 0 ≤ γ ≤ 1)
- **max_a' Q(s', a')**: 다음 상태에서 가능한 최대 Q값
- **TD Error**: 목표값과 현재값의 차이

### 3.3 TD Error 이해하기

**TD (Temporal Difference) Error**:
```
TD Error = [현재 경험으로 추정한 Q값] - [이전에 추정한 Q값]
         = [R + γ · max Q(s', a')] - Q(s, a)
```

**의미**:
- 양수면: 예상보다 좋았음 → Q값을 올림
- 음수면: 예상보다 나빴음 → Q값을 낮춤
- 0이면: 예상과 동일 → 변화 없음

**예시**:
```
현재 Q(s, a) = 10
실제 받은 보상 R = 5
다음 상태 최대 Q = max Q(s') = 8
γ = 0.9, α = 0.1

TD Error = [5 + 0.9 × 8] - 10 = 12.2 - 10 = 2.2
새로운 Q(s, a) = 10 + 0.1 × 2.2 = 10.22
```

### 3.4 Q-Learning 알고리즘 (의사코드)

```python
# 초기화
Q(s, a) = 0 for all s, a
α = 0.1  # 학습률
γ = 0.99 # 할인율
ε = 1.0  # 탐험률

for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # 1. ε-greedy로 행동 선택
        if random.random() < ε:
            action = random.choice(actions)  # 탐험
        else:
            action = argmax_a Q(state, a)    # 활용

        # 2. 행동 실행
        next_state, reward, done = env.step(action)

        # 3. Q-함수 업데이트
        best_next_q = max_a' Q(next_state, a')
        td_error = reward + γ * best_next_q - Q(state, action)
        Q(state, action) += α * td_error

        # 4. 상태 전이
        state = next_state

    # ε 감소 (탐험 줄이기)
    ε = max(0.01, ε * 0.995)
```

### 3.5 Off-policy 특성

Q-Learning은 **Off-policy** 알고리즘입니다.

**Off-policy란?**
- **행동 정책** (Behavior Policy): 실제로 행동을 선택하는 정책 (ε-greedy)
- **학습 정책** (Target Policy): 학습하려는 정책 (greedy, max Q)
- 두 정책이 **다를 수 있음**

**Q-Learning의 경우**:
- 행동은 ε-greedy로 선택 (탐험 포함)
- 업데이트는 max Q(s') 사용 (탐험 없는 greedy)

**장점**:
- 탐험을 하면서도 최적 정책을 학습할 수 있음
- 다른 정책의 경험을 재사용 가능

### 3.6 Q-table 구현

간단한 환경에서는 **Q-table**을 사용합니다.

**Q-table**: 모든 (상태, 행동) 쌍의 Q값을 저장하는 표

**예시 (4개 상태, 2개 행동)**:
```
       행동0  행동1
상태0   0.5    0.8
상태1   0.3    0.2
상태2   0.9    0.7
상태3   0.0    0.0
```

**Python 구현**:
```python
import numpy as np

# Q-table 초기화
num_states = 16
num_actions = 4
Q = np.zeros((num_states, num_actions))

# Q값 업데이트
state = 5
action = 2
reward = 1.0
next_state = 9
alpha = 0.1
gamma = 0.99

best_next_q = np.max(Q[next_state])
Q[state, action] += alpha * (reward + gamma * best_next_q - Q[state, action])
```

---

## 4. SARSA: On-policy 학습

### 4.1 SARSA란?

**SARSA** = State-Action-Reward-State-Action

Q-Learning과 비슷하지만 **On-policy** 방식입니다.

### 4.2 SARSA 업데이트 규칙

```python
Q(s, a) ← Q(s, a) + α · [R + γ · Q(s', a') - Q(s, a)]
```

**Q-Learning과의 차이**:
- Q-Learning: `max_a' Q(s', a')` 사용
- SARSA: `Q(s', a')` 사용 (실제로 선택한 행동)

### 4.3 On-policy 특성

**On-policy란?**
- 행동 정책과 학습 정책이 **같음**
- 실제로 한 행동으로 학습

**SARSA의 경우**:
- 행동도 ε-greedy로 선택
- 업데이트도 그 행동의 Q값 사용

**의미**:
- 더 보수적인 정책 학습
- 탐험 중에 받은 패널티도 학습에 반영

### 4.4 SARSA 알고리즘 (의사코드)

```python
for episode in range(num_episodes):
    state = env.reset()
    action = epsilon_greedy(Q, state, ε)  # 첫 행동 선택
    done = False

    while not done:
        # 1. 행동 실행
        next_state, reward, done = env.step(action)

        # 2. 다음 행동 선택 (ε-greedy)
        next_action = epsilon_greedy(Q, next_state, ε)

        # 3. Q-함수 업데이트 (next_action의 Q값 사용!)
        td_error = reward + γ * Q(next_state, next_action) - Q(state, action)
        Q(state, action) += α * td_error

        # 4. 상태 및 행동 전이
        state = next_state
        action = next_action  # 이미 선택한 행동을 사용
```

### 4.5 Q-Learning vs SARSA 비교

| 구분 | Q-Learning | SARSA |
|------|------------|-------|
| **업데이트** | max Q(s', a') | Q(s', a') |
| **정책 타입** | Off-policy | On-policy |
| **학습 대상** | 최적 정책 | 현재 정책 |
| **특성** | 공격적, 위험 감수 | 보수적, 안전 |
| **사용 사례** | 시뮬레이션, 위험 OK | 실제 로봇, 안전 중요 |

**예시 상황: 절벽 옆 경로**

```
S . . . . . . . . . G
. . . . . . . . . . .
C C C C C C C C C C C  (절벽)
```

- **Q-Learning**: 절벽 바로 위 최단경로 학습 (위험하지만 최적)
- **SARSA**: 절벽에서 멀리 떨어진 안전한 경로 학습

**이유**:
- Q-Learning: max Q 사용 → 실수(탐험)로 떨어져도 무시
- SARSA: 실제 행동 사용 → 실수로 떨어진 경험도 학습

---

## 5. ε-Greedy 탐험 전략

### 5.1 탐험-활용 딜레마

강화학습의 근본적인 문제:

**탐험 (Exploration)**:
- 새로운 행동을 시도하여 더 좋은 방법을 발견
- 단기적으로는 손해 볼 수 있음

**활용 (Exploitation)**:
- 현재 알고 있는 최선의 행동 선택
- 더 좋은 방법을 놓칠 수 있음

**예시: 식당 선택**
- 탐험: 새로운 식당 가보기 (맛없을 수도...)
- 활용: 아는 맛집 가기 (새로운 맛집 발견 못함)

### 5.2 ε-Greedy 전략

가장 간단하고 효과적인 탐험 전략입니다.

**알고리즘**:
```python
def epsilon_greedy(Q, state, epsilon):
    if random.random() < epsilon:
        # 탐험: 랜덤 행동
        return random.choice(available_actions)
    else:
        # 활용: 최선의 행동
        return argmax_a Q[state, a]
```

**파라미터**:
- **ε = 0.0**: 완전 활용 (탐험 없음)
- **ε = 1.0**: 완전 탐험 (랜덤)
- **ε = 0.1**: 10% 탐험, 90% 활용

### 5.3 ε-Decay: 탐험 점진적 감소

학습 초반에는 많이 탐험하고, 후반에는 활용하는 것이 효율적입니다.

**ε-Decay 전략**:

**1) 선형 감소**:
```python
epsilon = max(epsilon_min, epsilon - decay_rate)
```

**2) 지수 감소** (더 일반적):
```python
epsilon = max(epsilon_min, epsilon * decay_rate)
# decay_rate = 0.995
```

**예시**:
```python
epsilon = 1.0
epsilon_min = 0.01
decay_rate = 0.995

for episode in range(1000):
    # ... 학습 ...
    epsilon = max(epsilon_min, epsilon * decay_rate)

# 결과:
# Episode 0: ε = 1.0
# Episode 100: ε ≈ 0.61
# Episode 300: ε ≈ 0.22
# Episode 500: ε ≈ 0.08
# Episode 700+: ε = 0.01 (최소값 도달)
```

### 5.4 다른 탐험 전략들

**1) Softmax (Boltzmann Exploration)**:
```python
def softmax(Q, state, temperature):
    exp_Q = np.exp(Q[state] / temperature)
    probs = exp_Q / np.sum(exp_Q)
    return np.random.choice(actions, p=probs)
```
- 좋은 행동일수록 높은 확률
- 나쁜 행동도 작은 확률로 선택

**2) UCB (Upper Confidence Bound)**:
```python
Q_ucb[a] = Q[state, a] + c * sqrt(log(t) / N[state, a])
action = argmax_a Q_ucb[a]
```
- 적게 시도한 행동에 보너스
- Multi-armed Bandit에서 효과적

**3) ε-Soft**:
```python
# 모든 행동이 최소 확률 ε/|A|
# 최선의 행동이 1 - ε + ε/|A|
```

**일반적으로 ε-greedy가 가장 단순하고 효과적**입니다.

---

## 6. 하이퍼파라미터 튜닝

Q-Learning/SARSA의 성능은 하이퍼파라미터에 크게 좌우됩니다.

### 6.1 학습률 (Learning Rate, α)

**의미**: 새로운 정보를 얼마나 빠르게 반영할지

**범위**: 0 < α ≤ 1

**값에 따른 특성**:

**α = 0.01 (작은 값)**:
- 장점: 안정적, 노이즈에 강함
- 단점: 학습 매우 느림
- 사용: 안정적 환경, 노이즈 많은 환경

**α = 0.1 ~ 0.3 (중간 값)**:
- 장점: 균형 잡힌 학습 속도
- 단점: -
- 사용: **대부분의 경우 추천**

**α = 0.9 (큰 값)**:
- 장점: 빠른 학습
- 단점: 불안정, 노이즈에 민감, 진동
- 사용: 결정적 환경, 빠른 적응 필요

**학습 곡선 비교**:
```
보상
 │        α=0.5 (불안정)
 │       /\/\/\
 │      /       α=0.1 (안정적)
 │     /       /
 │    /    ___/
 │   /  __/     α=0.01 (느림)
 │  / _/
 │_/_____________________ 에피소드
```

**실무 팁**:
- 시작: α = 0.1
- 빠른 학습 필요: α = 0.3
- 안정성 필요: α = 0.05
- **α도 decay 가능**: α = α * 0.9999

### 6.2 할인율 (Discount Factor, γ)

**의미**: 미래 보상을 얼마나 중요하게 볼지

**범위**: 0 ≤ γ ≤ 1

**값에 따른 특성**:

**γ = 0.0**:
- 의미: 즉각 보상만 고려 (근시안적)
- 학습: 매우 빠름
- 정책: 단기적 최적화
- 사용: 드물음

**γ = 0.9**:
- 의미: 중간 미래까지 고려
- 학습: 적당한 속도
- 정책: 10스텝 정도 미래 계획
- 사용: 짧은 에피소드

**γ = 0.99**:
- 의미: 먼 미래도 중요하게 고려
- 학습: 느림 (많은 에피소드 필요)
- 정책: 100스텝 정도 미래 계획
- 사용: **대부분의 경우, 긴 에피소드**

**γ = 1.0**:
- 의미: 모든 미래 보상 동등하게 중요
- 문제: 에피소드가 무한히 길면 발산 가능
- 사용: 에피소드가 항상 종료되는 경우만

**미래 보상의 가치**:
```
γ = 0.9 일 때:
- 1 스텝 후: 1.0 * 0.9 = 0.90
- 2 스텝 후: 1.0 * 0.9² = 0.81
- 5 스텝 후: 1.0 * 0.9⁵ = 0.59
- 10 스텝 후: 1.0 * 0.9¹⁰ = 0.35

γ = 0.99 일 때:
- 10 스텝 후: 1.0 * 0.99¹⁰ = 0.90
- 50 스텝 후: 1.0 * 0.99⁵⁰ = 0.61
- 100 스텝 후: 1.0 * 0.99¹⁰⁰ = 0.37
```

**실무 팁**:
- 기본값: **γ = 0.99**
- 짧은 에피소드 (<50 스텝): γ = 0.9
- 긴 에피소드 (>200 스텝): γ = 0.995
- 무한 에피소드: γ < 1.0 필수

### 6.3 탐험률 (Exploration Rate, ε)

**초기값**: ε = 1.0 (완전 랜덤)

**최종값**: ε = 0.01 ~ 0.1 (약간의 탐험 유지)

**감소 속도**:

**빠른 감소** (decay_rate = 0.99):
- 100 에피소드에 ε ≈ 0.37
- 빠르게 수렴
- 위험: 조기 수렴 (로컬 최적)

**중간 감소** (decay_rate = 0.995):
- 500 에피소드에 ε ≈ 0.08
- **대부분의 경우 추천**

**느린 감소** (decay_rate = 0.999):
- 1000 에피소드에도 ε ≈ 0.37
- 충분한 탐험
- 위험: 학습 느림

**실무 팁**:
```python
# 설정 1: 표준
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

# 설정 2: 빠른 학습 (위험)
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99

# 설정 3: 안전한 학습 (느림)
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.999
```

### 6.4 하이퍼파라미터 조합 예시

**초보자 추천 (안정적)**:
```python
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
```

**빠른 프로토타이핑**:
```python
alpha = 0.3
gamma = 0.9
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99
```

**안전 중요 (로봇 등)**:
```python
alpha = 0.05
gamma = 0.99
epsilon = 0.5  # 처음부터 너무 랜덤하지 않게
epsilon_min = 0.05
epsilon_decay = 0.999
```

### 6.5 그리드 서치로 최적 값 찾기

```python
import itertools

alphas = [0.01, 0.1, 0.3]
gammas = [0.9, 0.95, 0.99]
epsilon_decays = [0.99, 0.995, 0.999]

best_reward = -float('inf')
best_params = None

for alpha, gamma, decay in itertools.product(alphas, gammas, epsilon_decays):
    avg_reward = train_and_evaluate(alpha, gamma, decay)
    if avg_reward > best_reward:
        best_reward = avg_reward
        best_params = (alpha, gamma, decay)

print(f"최적 파라미터: α={best_params[0]}, γ={best_params[1]}, decay={best_params[2]}")
```

---

## 7. FrozenLake 환경

### 7.1 환경 소개

**FrozenLake-v1**은 OpenAI Gymnasium의 고전적인 강화학습 환경입니다.

**배경 스토리**:
- 겨울에 얼어붙은 호수를 건너야 함
- 시작점(S)에서 목표(G)까지 안전하게 이동
- 얼음 구멍(H)에 빠지면 실패

**4x4 맵 예시**:
```
S F F F
F H F H
F F F H
H F F G
```

**요소**:
- **S (Start)**: 시작 위치
- **F (Frozen)**: 안전한 얼음
- **H (Hole)**: 구멍 (빠지면 게임 종료, 보상 0)
- **G (Goal)**: 목표 (도착하면 보상 1)

### 7.2 상태와 행동

**상태 공간**:
- 4×4 = 16개 상태
- 각 칸의 위치를 0~15로 인덱싱
```
 0  1  2  3
 4  5  6  7
 8  9 10 11
12 13 14 15
```

**행동 공간**:
- 0: 왼쪽 (LEFT)
- 1: 아래 (DOWN)
- 2: 오른쪽 (RIGHT)
- 3: 위 (UP)

### 7.3 보상 구조

- 목표(G) 도달: +1
- 구멍(H) 빠짐: 0
- 일반 이동: 0

**특징**: 희소 보상 (Sparse Reward)
- 목표에 도달하기 전까지 보상 없음
- 학습이 어려운 이유

### 7.4 is_slippery 옵션

**is_slippery=True (기본값)**:
- **확률적 환경** (Stochastic)
- 의도한 방향: 1/3 확률
- 수직 방향들: 각 1/3 확률

예시: 오른쪽(2) 행동
```
실제 이동:
- 오른쪽: 1/3
- 위: 1/3
- 아래: 1/3
```

**is_slippery=False**:
- **결정적 환경** (Deterministic)
- 의도한 방향으로 100% 이동
- 학습하기 더 쉬움

### 7.5 환경 사용 예시

```python
import gymnasium as gym

# 환경 생성
env = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')

# 환경 정보
print(f"상태 개수: {env.observation_space.n}")  # 16
print(f"행동 개수: {env.action_space.n}")      # 4

# 에피소드 실행
state, info = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # 랜덤 행동
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    state = next_state

env.close()
```

### 7.6 Q-Learning으로 FrozenLake 학습하기

**Q-table 초기화**:
```python
Q = np.zeros((16, 4))  # 16 states, 4 actions
```

**학습 과정**:
```python
for episode in range(10000):
    state, _ = env.reset()
    done = False

    while not done:
        # ε-greedy 행동 선택
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])

        # 환경 step
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Q-Learning 업데이트
        best_next = np.max(Q[next_state]) if not done else 0
        td_error = reward + gamma * best_next - Q[state, action]
        Q[state, action] += alpha * td_error

        state = next_state

    epsilon = max(0.01, epsilon * 0.995)
```

### 7.7 학습된 정책 시각화

```python
def print_policy(Q):
    """학습된 Q-table로부터 최적 정책을 화살표로 출력"""
    symbols = ['←', '↓', '→', '↑']
    policy = np.argmax(Q, axis=1)

    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(symbols[policy[state]], end=' ')
        print()

# 예시 출력:
# ↓ → ↓ ←
# ← ← ← ←
# → ↓ ← ←
# ← → → G
```

### 7.8 성공률 측정

```python
def evaluate_policy(env, Q, num_episodes=100):
    """학습된 정책의 성공률 측정 (탐험 없이 greedy만)"""
    success = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = np.argmax(Q[state])  # greedy (탐험 없음)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1:
                success += 1

    return success / num_episodes

# 사용
success_rate = evaluate_policy(env, Q)
print(f"성공률: {success_rate * 100:.1f}%")
```

---

## 8. 실전: 쥐를 잡자 (ALPHANO 문제 2)

### 8.1 문제 소개

**쥐를 잡자**는 ALPHANO 플랫폼의 비대칭 게임입니다.

**게임 개요**:
- 보드: 7×11 격자
- 플레이어: 고양이 vs 쥐
- 고양이 (선공): 쥐를 잡는 것이 목표
- 쥐 (후공): 고양이를 피하는 것이 목표

### 8.2 ALPHANO 프로토콜

**표준 입출력**:
```
1. READY FIRST 또는 READY SECOND
   → 출력: OK

2. TURN my_time opp_time
   → 출력: MOVE x y (이동할 좌표, 1-indexed)

3. OPP x y
   → 상대방의 이동 정보 업데이트

4. FINISH
   → 프로그램 종료
```

### 8.3 간단한 휴리스틱 전략

Q-Learning을 적용하기 전에, 먼저 간단한 규칙 기반 전략을 만들어봅시다.

**고양이 전략**:
- 쥐와의 Manhattan Distance를 계산
- 거리가 가장 가까워지는 이동 선택

**쥐 전략**:
- 고양이와의 Manhattan Distance를 계산
- 거리가 가장 멀어지는 이동 선택

**Manhattan Distance**:
```python
def manhattan_distance(x1, y1, x2, y2):
    return abs(x1 - x2) + abs(y1 - y2)
```

**휴리스틱 에이전트 의사코드**:
```python
def get_best_move(my_pos, opp_pos, is_cat):
    best_move = None
    best_distance = float('inf') if is_cat else -float('inf')

    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
        new_x, new_y = my_pos[0] + dx, my_pos[1] + dy

        # 범위 체크
        if not is_valid(new_x, new_y):
            continue

        dist = manhattan_distance(new_x, new_y, opp_pos[0], opp_pos[1])

        if is_cat:  # 고양이: 거리 최소화
            if dist < best_distance:
                best_distance = dist
                best_move = (new_x, new_y)
        else:  # 쥐: 거리 최대화
            if dist > best_distance:
                best_distance = dist
                best_move = (new_x, new_y)

    return best_move
```

### 8.4 Q-Learning 적용 고려사항

**상태 표현**:
- Option 1: (cat_x, cat_y, mouse_x, mouse_y) → 7×11×7×11 = 5929 상태
- Option 2: (relative_x, relative_y) → 상대적 위치만 고려

**행동 공간**:
- 4방향 이동: 위, 아래, 왼쪽, 오른쪽

**보상 설계**:
- 고양이: 쥐 잡으면 +100, 거리 줄이면 +1, 못 잡으면 -1
- 쥐: 잡히면 -100, 거리 늘리면 +1, 생존하면 +1

**학습 방법**:
- Self-play: 고양이와 쥐 정책을 동시에 학습
- 반복 학습으로 점점 강해지는 에이전트

### 8.5 실전 제출 시 주의사항

실제 ALPHANO 제출 시:
1. 문제 상세 규칙 확인 (이동 규칙, 승리 조건)
2. 시간 제한 고려 (턴당 1초 등)
3. 입출력 형식 정확히 맞추기
4. 예외 처리 (잘못된 입력, 범위 벗어남)

---

## 9. 핵심 정리

### 9.1 이번 주에 배운 것

1. **Model-free RL**: 환경 모델 없이 경험으로 학습
2. **Q-Learning**: Off-policy, max Q 사용
3. **SARSA**: On-policy, 실제 행동의 Q 사용
4. **ε-Greedy**: 탐험-활용 균형
5. **하이퍼파라미터**: α, γ, ε의 영향
6. **FrozenLake**: Q-Learning 실습 환경

### 9.2 Q-Learning vs SARSA 요약

| 특징 | Q-Learning | SARSA |
|------|------------|-------|
| 업데이트 | max Q(s') | Q(s', a') |
| 정책 | Off-policy | On-policy |
| 수렴 | 최적 정책 | 현재 정책 |
| 특성 | 공격적 | 보수적 |

### 9.3 실무 가이드라인

**언제 Q-Learning?**
- 시뮬레이션 환경
- 실패해도 괜찮음
- 최적 성능 필요

**언제 SARSA?**
- 실제 환경 (로봇 등)
- 안전이 중요
- 학습 중 사고 방지

**기본 하이퍼파라미터**:
```python
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
```

### 9.4 Q-Learning의 한계

1. **큰 상태공간**: Q-table이 너무 커짐 (바둑: 10^170)
2. **연속 상태**: 실수값 상태를 다룰 수 없음
3. **일반화 부족**: 비슷한 상태에 대한 학습 공유 안 됨

**해결책**: 다음 주에 배울 **DQN (Deep Q-Network)**
- Q-table 대신 신경망 사용
- 함수 근사로 큰 상태공간 처리
- 연속 상태 처리 가능

---

## 10. 다음 주 예고: DQN (Deep Q-Network)

### 10.1 왜 DQN이 필요한가?

**FrozenLake**: 16 상태 × 4 행동 = 64 값 → Q-table OK

**Atari 게임**: 210×160 픽셀, RGB → 10^6000 상태 → Q-table 불가능!

**해결책**: 신경망으로 Q-함수 근사
```
Q(s, a) ≈ Neural_Network(s, a; θ)
```

### 10.2 DQN의 핵심 아이디어

1. **Deep Neural Network**: Q-table → Q-network
2. **Experience Replay**: 과거 경험 재사용
3. **Target Network**: 안정적인 학습

### 10.3 DQN의 성과

- **2015년 Nature 논문**: Atari 게임 49개에서 인간 수준 달성
- Breakout, Pong, Space Invaders 등
- 픽셀 입력만으로 학습

### 10.4 배울 내용

- 신경망으로 Q-함수 근사
- Experience Replay Buffer
- Target Network
- PyTorch로 DQN 구현
- CartPole, Atari 게임 학습

**준비물**: PyTorch 설치
```bash
pip install torch gymnasium[atari] gymnasium[accept-rom-license]
```

---

## 연습 문제

### 문제 1: 이론 문제

1. Q-Learning과 SARSA의 업데이트 수식을 쓰고 차이점을 설명하세요.
2. ε-greedy에서 ε=0.2일 때, 4개의 행동 중 최선의 행동이 선택될 확률은?
3. γ=0.9일 때, 10스텝 후 받을 보상 1.0의 현재 가치는?

### 문제 2: FrozenLake 학습

`practice/q_learning_frozenlake.py`를 실행하고:
1. is_slippery=False로 학습 성공률 측정
2. is_slippery=True로 변경하여 학습 (더 어려움)
3. 하이퍼파라미터를 바꿔가며 최고 성공률 달성

### 문제 3: SARSA 비교

`practice/sarsa_frozenlake.py`를 실행하고:
1. Q-Learning과 학습 곡선 비교
2. 두 알고리즘의 최종 정책 시각화 비교
3. 어떤 알고리즘이 더 안전한 경로를 학습하는지 분석

### 문제 4: 하이퍼파라미터 실험

`practice/hyperparameter_experiment.py`를 실행하고:
1. α를 0.01, 0.1, 0.5로 바꿔가며 실험
2. γ를 0.9, 0.95, 0.99로 바꿔가며 실험
3. 각 파라미터의 영향을 그래프로 분석

### 문제 5: 쥐를 잡자 (도전 과제)

`alphano/catch_mouse_heuristic.py`를 개선하여:
1. 더 똑똑한 휴리스틱 전략 설계
2. (선택) Q-Learning을 적용한 에이전트 구현
3. (선택) Self-play로 학습시키기

---

## 참고 자료

### 책
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2nd ed.)
  - Chapter 6: Temporal-Difference Learning

### 논문
- Watkins & Dayan (1992), "Q-Learning"
- Rummery & Niranjan (1994), "SARSA"

### 온라인 자료
- OpenAI Spinning Up: https://spinningup.openai.com/
- Gymnasium Documentation: https://gymnasium.farama.org/
- ALPHANO Platform: https://alphano.kr/

### 코드 예제
- 이번 주 practice/ 폴더의 모든 코드
- Gymnasium examples: https://github.com/Farama-Foundation/Gymnasium

---

## 마치며

이번 주에는 **Model-free 강화학습의 기초**인 Q-Learning과 SARSA를 배웠습니다.

핵심은:
1. 환경 모델 없이 **경험으로 학습**
2. **Q-함수**를 반복적으로 업데이트
3. **ε-greedy**로 탐험-활용 균형
4. 하이퍼파라미터가 성능에 큰 영향

다음 주에는 이 개념을 확장하여, **신경망으로 Q-함수를 근사**하는 **DQN**을 배웁니다. Atari 게임을 직접 학습시켜보면서 딥러닝과 강화학습의 결합을 경험하게 될 것입니다.

**실습을 많이 해보세요!** 코드를 직접 실행하고, 파라미터를 바꿔가며 학습 과정을 관찰하는 것이 가장 중요합니다.

다음 주에 만나요!
