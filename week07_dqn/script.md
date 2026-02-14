# Week 7 수업 대본: Deep Q-Network (DQN)

**수업 시간**: 90분
**대상**: 고등학생
**준비물**: 노트북, Python 환경 (PyTorch, Gymnasium 설치)

---

## 도입 (5분)

### [0:00-0:30] 인사 및 지난 주 복습

**교사**: 안녕하세요 여러분! 지난 주에는 Q-Learning을 배웠죠? 누가 Q-Learning의 핵심 아이디어를 말해볼 수 있나요?

**학생**: Q-table을 만들어서 각 상태에서 어떤 행동이 좋은지 학습하는 거요!

**교사**: 맞아요! Q(s,a) 값을 저장해서, 상태 s에서 행동 a를 했을 때 얼마나 좋은지 기억하는 거죠. 그런데 지난 주 GridWorld는 5×5였죠? 상태가 25개밖에 안 됐어요. 그런데 만약 보드가 훨씬 크다면?

### [0:30-1:30] 동기부여: Q-Learning의 한계

**교사**: (칠판에 쓰면서) ATAXX 게임을 기억하시나요? 7×7 보드인데, 각 칸이 빈 칸, 흰색, 검은색 3가지 상태를 가질 수 있어요. 그러면 가능한 보드 상태가 몇 개일까요?

**학생**: 음... 3의 49제곱인가요?

**교사**: 정확해요! (계산기 보여주며) 3^49 = 약 2경입니다. 2 뒤에 0이 23개 붙어요. 이게 얼마나 큰 수인지 감이 오나요?

**학생**: 상상이 안 돼요...

**교사**: 1초에 1억 개 상태를 처리할 수 있다고 해도, 모든 상태를 처리하려면 **600억 년**이 걸려요. 우주의 나이가 138억 년인데 그것보다 훨씬 길죠. Q-table을 컴퓨터 메모리에 저장하는 것조차 불가능합니다.

### [1:30-2:30] 해결책 제시

**교사**: 그럼 어떻게 해야 할까요? 핵심은 이겁니다. (칠판에 크게 쓰며)

```
모든 상태를 외우지 말고,
패턴을 배우자!
```

**교사**: 실생활 예를 들어볼게요. 여러분이 덧셈을 배울 때, "2+3=5", "7+9=16", "123+456=579" 이런 걸 모두 외웠나요? 아니면 덧셈 **규칙**을 배웠나요?

**학생**: 규칙을 배웠어요!

**교사**: 맞아요! 규칙을 한 번 배우면, 본 적 없는 "9876 + 1234"도 계산할 수 있죠. 이것이 바로 **일반화**입니다. Q-Learning에도 이런 일반화가 필요해요. 그래서 오늘 배울 것이 바로 **Deep Q-Network, DQN**입니다!

### [2:30-5:00] 오늘의 목표

**교사**: (슬라이드 보여주며) 오늘 배울 내용입니다:

1. 신경망으로 Q 함수를 근사하는 방법
2. DQN의 두 가지 핵심 기법
   - Experience Replay
   - Target Network
3. PyTorch로 CartPole 문제 해결하기
4. Betris에 DQN 적용하는 방법

자, 시작해봅시다!

---

## 이론 1: 함수 근사와 신경망 기초 (15분)

### [5:00-7:00] 함수 근사 개념

**교사**: (칠판에 그리며) Q-Learning은 이런 표를 만들었죠:

```
상태1, 행동1 → 10.5
상태1, 행동2 → 8.3
상태2, 행동1 → 12.7
...
```

이제는 이 표 대신 **함수**를 사용합니다:

```
Q(상태, 행동) = 함수로 계산
```

**교사**: 이 함수를 어떻게 만들까요? 바로 **신경망**을 사용합니다. 신경망은 매우 복잡한 패턴을 학습할 수 있어요.

### [7:00-10:00] 신경망 기초

**교사**: (그림 그리며) 신경망의 기본 단위는 **뉴런**입니다.

```
입력: x₁, x₂, x₃
     ↓
  [ 뉴런 ]
     ↓
   출력: y
```

뉴런이 하는 일:
1. 입력을 받음
2. 각 입력에 **가중치**를 곱함
3. 다 더함
4. **활성화 함수** 적용

**교사**: 예를 들어볼게요. (칠판에 쓰며)

```python
입력 x = [0.5, 0.3, 0.8]
가중치 w = [0.2, -0.4, 0.6]
편향 b = 0.1

# 가중합 계산
z = 0.2×0.5 + (-0.4)×0.3 + 0.6×0.8 + 0.1
  = 0.1 - 0.12 + 0.48 + 0.1
  = 0.56

# ReLU 활성화 함수
출력 = max(0, z) = 0.56
```

**학생**: ReLU가 뭐예요?

**교사**: 좋은 질문이에요! ReLU는 Rectified Linear Unit의 약자인데, 아주 간단해요:

```python
ReLU(x) = max(0, x)

ReLU(-5) = 0
ReLU(0) = 0
ReLU(3) = 3
```

음수는 0으로 만들고, 양수는 그대로 두는 거예요. (그래프 그리며) 이렇게 생겼어요.

### [10:00-12:00] 왜 비선형 함수가 필요한가?

**교사**: 누군가 물어볼 수 있어요. "그냥 곱하고 더하기만 하면 안 돼요? 왜 ReLU 같은 걸 써야 하나요?"

(칠판에 수식 쓰며) 만약 선형 함수만 사용하면:

```
Layer 1: y = W₁·x + b₁
Layer 2: z = W₂·y + b₂
        = W₂·(W₁·x + b₁) + b₂
        = (W₂·W₁)·x + (W₂·b₁ + b₂)
        = W₃·x + b₃
```

결국 선형 함수가 돼요! 아무리 층을 많이 쌓아도 직선밖에 표현 못 합니다. 하지만 ReLU 같은 비선형 함수를 쓰면 복잡한 곡선, 패턴을 학습할 수 있어요.

### [12:00-15:00] 신경망 학습: 역전파

**교사**: 그럼 신경망은 어떻게 학습할까요? **역전파(Backpropagation)**라는 알고리즘을 사용해요.

**과정**:
1. 입력을 넣고 출력을 계산 (순전파)
2. 출력과 정답의 차이 계산 (손실)
3. 출력에서 입력 방향으로 거꾸로 가며 그래디언트 계산 (역전파)
4. 가중치를 조금씩 조정

(비유) 산에서 내려오는 것과 비슷해요. 어느 방향이 가장 가파르게 내려가는지(그래디언트) 확인하고, 그쪽으로 조금씩 이동하는 거죠.

**학생**: 그래디언트가 정확히 뭐예요?

**교사**: 그래디언트는 "각 가중치를 조금 바꾸면 손실이 얼마나 변하는가"를 나타내요. 만약 어떤 가중치의 그래디언트가 크면, 그 가중치가 손실에 큰 영향을 미친다는 뜻이에요. 그래서 그 가중치를 많이 조정해야 하죠.

### [15:00-17:00] 경사하강법

**교사**: (슬라이드 보여주며) 가중치를 조정하는 공식은 이렇게 간단해요:

```
새 가중치 = 옛 가중치 - 학습률 × 그래디언트

w_new = w_old - α × ∇L
```

**학습률 α**가 중요한데:
- 너무 크면: 왔다갔다 하며 발산
- 너무 작으면: 학습이 너무 느림
- 적절한 값: 보통 0.001, 0.0001 등

(그림 그리며) 산을 내려가는데, 발걸음을 너무 크게 하면 산을 넘어가버리고, 너무 작게 하면 평생 못 내려와요. 적당히 해야죠!

### [17:00-20:00] 간단한 질의응답

**교사**: 여기까지 질문 있나요?

**학생**: 신경망이 Q-table보다 왜 더 좋은 거예요?

**교사**: 아주 좋은 질문이에요! 세 가지 이유가 있어요:

1. **메모리**: Q-table은 10^23개 값을 저장해야 하지만, 신경망은 가중치 몇 만 개만 저장하면 돼요.

2. **일반화**: Q-table은 정확히 본 상태만 알 수 있지만, 신경망은 비슷한 상태를 추론할 수 있어요.

3. **학습 효율**: 신경망은 한 상태를 학습하면 비슷한 상태들도 함께 개선돼요.

**학생**: 그럼 신경망이 항상 더 좋은 거 아니에요?

**교사**: 아니요! 만약 상태가 적다면(예: 5×5 GridWorld) Q-table이 더 간단하고 빠르고 안정적이에요. 상태가 많을 때만 신경망을 사용하는 거죠. 도구는 상황에 맞게 선택해야 해요!

---

## 이론 2: DQN 핵심 기법 (15분)

### [20:00-22:00] PyTorch 간단 소개

**교사**: 신경망을 구현하려면 **PyTorch**라는 도구를 사용할 거예요. 간단한 예제를 보죠.

```python
import torch
import torch.nn as nn

# 신경망 정의
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 128)  # 입력 4, 출력 128
        self.fc2 = nn.Linear(128, 2)   # 입력 128, 출력 2

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```

**교사**: `nn.Linear`가 하는 일은:
```
output = input × weight + bias
```

`torch.relu`는 우리가 방금 배운 ReLU 함수예요!

### [22:00-27:00] 문제 1: 연속된 경험의 상관관계

**교사**: 이제 DQN의 핵심으로 넘어가볼게요. 신경망으로 Q 함수를 만들었다고 해봅시다. Q-Learning처럼 경험을 즉시 사용하면 어떻게 될까요?

```python
state1 → action1 → reward1 → state2  # 즉시 학습
state2 → action2 → reward2 → state3  # 즉시 학습
state3 → action3 → reward3 → state4  # 즉시 학습
```

**학생**: 뭐가 문제예요?

**교사**: (칠판에 예시 쓰며) CartPole 게임을 생각해봐요:

```
state_t   = [0.1, 0.5, 0.01, 0.1]
state_t+1 = [0.11, 0.48, 0.011, 0.09]
```

두 상태가 거의 똑같죠? 이렇게 비슷한 데이터로만 학습하면:
- 신경망이 특정 상황에만 과적합
- 다양한 상황을 일반화하지 못함
- 학습이 불안정

(비유) 시험공부를 할 때, 1번 문제만 100번 푸는 것과 비슷해요. 1번은 잘 풀겠지만, 2번, 3번은 못 풀겠죠?

### [27:00-30:00] 해결책 1: Experience Replay

**교사**: 해결책은 **Experience Replay**예요! (칠판에 그림 그리며)

```
Replay Buffer (저장소)
┌─────────────────────────┐
│ (s₁, a₁, r₁, s₁', done) │
│ (s₂, a₂, r₂, s₂', done) │
│ (s₃, a₃, r₃, s₃', done) │
│ ...                     │
│ (s₁₀₀₀, a, r, s', done)│
└─────────────────────────┘

랜덤으로 32개 샘플링 ↓

학습에 사용
```

**과정**:
1. 경험을 버퍼에 **저장**
2. 버퍼에서 **랜덤 샘플링**
3. 샘플링된 경험으로 **학습**

**교사**: 이렇게 하면 어떤 좋은 점이 있을까요?

**학생**: 다양한 경험을 섞어서 배울 수 있어요!

**교사**: 정확해요! 랜덤 샘플링하면:
- Episode 1의 경험
- Episode 5의 경험
- Episode 12의 경험

이렇게 섞여서 배치가 만들어져요. 상관관계가 사라지고, 학습이 안정적이 되죠!

### [30:00-32:00] Replay Buffer 구현

**교사**: (코드 보여주며) 구현은 아주 간단해요:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def size(self):
        return len(self.buffer)
```

**교사**: `deque`는 Double-Ended Queue의 약자인데, `maxlen`을 설정하면 자동으로 오래된 항목을 삭제해줘요. 10000개까지만 저장하는 거죠.

### [32:00-35:00] 문제 2: 움직이는 목표

**교사**: 두 번째 문제를 볼게요. Q-Learning 업데이트 식을 기억하나요?

```
Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]
                      └────────────┘
                         목표값
```

신경망 버전:
```
Loss = (Q(s,a;θ) - [r + γ·max Q(s',a';θ)])²
        └──────┘     └─────────────────┘
        예측값            목표값
```

**교사**: 뭐가 문제일까요? (학생들 생각할 시간 주기)

**학생**: 예측값과 목표값이 둘 다 θ를 사용해요!

**교사**: 정확해요! θ를 업데이트하면:
- 예측값이 변함 ✓ (이건 원하는 거)
- 목표값도 변함 ✗ (이건 문제!)

(비유) 과녁을 맞추려는데, 화살을 쏠 때마다 과녁이 움직이는 것과 같아요. 어떻게 맞추겠어요?

### [35:00-40:00] 해결책 2: Target Network

**교사**: 해결책은 **Target Network**예요! (그림 그리며)

```
메인 네트워크 (θ)
- 예측에 사용
- 매 스텝 업데이트
- 학습이 계속 진행

타겟 네트워크 (θ⁻)
- 목표값 계산에만 사용
- 천천히 업데이트 (예: 1000 step마다)
- 안정적인 목표 제공
```

**Loss 계산**:
```python
# 예측: 메인 네트워크
current_q = q_network(state, action)

# 목표: 타겟 네트워크
target_q = reward + gamma * target_network(next_state).max()

# Loss
loss = (current_q - target_q)²
```

**교사**: 타겟 네트워크는 1000 스텝마다 메인 네트워크를 복사해서 업데이트해요:

```python
if step % 1000 == 0:
    target_network = copy.deepcopy(q_network)
```

### [40:00-43:00] 왜 효과적인가?

**교사**: (그래프 그리며) 타겟 네트워크 없이 학습하면:

```
목표값: 10 → 15 → 8 → 20 → 12 → ...
(매 스텝 변함, 불안정)
```

타겟 네트워크 사용하면:

```
Step 1-1000:    목표 = 10 (고정)
Step 1001-2000: 목표 = 12 (고정)
Step 2001-3000: 목표 = 13 (고정)
(안정적)
```

고정된 목표를 향해 학습하니까 안정적이고, 주기적으로 업데이트하니까 발전도 해요!

### [43:00-45:00] 간단한 정리

**교사**: DQN의 두 가지 핵심 기법을 정리해볼게요:

**1. Experience Replay**
- 경험을 저장했다가 랜덤 샘플링
- 상관관계 제거
- 데이터 효율성 증가

**2. Target Network**
- 목표값 계산용 별도 네트워크
- 천천히 업데이트
- 학습 안정성 증가

이 두 가지가 없으면 신경망으로 Q-Learning을 하기 매우 어려워요!

---

## 실습 1: CartPole DQN 구현 (25분)

### [45:00-47:00] CartPole 환경 소개

**교사**: 이제 실습을 해봅시다! (화면 공유하며) CartPole이라는 환경을 사용할 거예요.

(CartPole 영상 재생)

**목표**: 막대가 쓰러지지 않도록 카트를 좌우로 움직이기

**상태**: 4개 값
- 카트 위치
- 카트 속도
- 막대 각도
- 막대 각속도

**행동**: 2개
- 0: 왼쪽
- 1: 오른쪽

**보상**: 매 타임스텝마다 +1

**성공 기준**: 200 타임스텝 이상 유지

### [47:00-50:00] 환경 테스트

**교사**: (실시간 코딩) 먼저 환경을 테스트해봅시다:

```python
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode='human')
state, info = env.reset()

print(f"초기 상태: {state}")
print(f"상태 크기: {len(state)}")
print(f"행동 수: {env.action_space.n}")

# 랜덤으로 플레이
total_reward = 0
for t in range(200):
    action = env.action_space.sample()  # 랜덤 행동
    state, reward, done, truncated, info = env.step(action)
    total_reward += reward

    if done or truncated:
        break

print(f"총 보상: {total_reward}")
env.close()
```

(실행하며) 랜덤으로 하니까 얼마 못 가죠? 보통 20~30 정도 받아요. 이제 DQN으로 학습시켜봅시다!

### [50:00-53:00] Q-Network 정의

**교사**: (코드 작성하며) 먼저 Q-Network를 만들어요:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 출력층은 활성화 함수 없음

# 테스트
q_net = QNetwork(state_size=4, action_size=2)
test_state = torch.tensor([0.1, -0.5, 0.2, 0.3], dtype=torch.float32)
q_values = q_net(test_state)
print(f"Q값: {q_values}")
print(f"최선의 행동: {q_values.argmax().item()}")
```

**학생**: 왜 마지막 층에는 활성화 함수가 없어요?

**교사**: 좋은 질문이에요! Q값은 양수일 수도, 음수일 수도 있어요. 예를 들어 나쁜 행동은 Q값이 -10일 수 있죠. 그래서 ReLU를 쓰면 안 돼요. 그냥 선형으로 출력해야 해요.

### [53:00-56:00] Replay Buffer 구현

**교사**: (코드 작성하며) 아까 본 Replay Buffer를 만들어요:

```python
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def size(self):
        return len(self.buffer)

# 테스트
buffer = ReplayBuffer(capacity=5)
buffer.store([1, 2, 3, 4], 0, 1.0, [1.1, 2.1, 3.1, 4.1], False)
buffer.store([2, 3, 4, 5], 1, 1.0, [2.1, 3.1, 4.1, 5.1], False)
print(f"버퍼 크기: {buffer.size()}")
```

### [56:00-62:00] DQN Agent 구현

**교사**: (코드 작성하며) 이제 DQN Agent를 만들어요. 조금 길지만 차근차근 따라오세요:

```python
import copy
import torch.optim as optim

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Q-Network와 Target Network
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = copy.deepcopy(self.q_network)

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        self.step_count = 0

    def select_action(self, state, epsilon):
        """ε-greedy 행동 선택"""
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()

    def learn(self, batch_size):
        """미니배치로 학습"""
        if self.replay_buffer.size() < batch_size:
            return

        # 미니배치 샘플링
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # 현재 Q값 (메인 네트워크)
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()

        # 목표 Q값 (타겟 네트워크)
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1)[0]
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Loss 계산 및 업데이트
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """타겟 네트워크 업데이트"""
        self.target_network.load_state_dict(self.q_network.state_dict())
```

**교사**: (중요 부분 설명하며)

**`select_action`**: ε-greedy로 행동을 선택해요. epsilon 확률로 랜덤, 나머지는 Q값이 높은 행동을 선택.

**`learn`**: 핵심이에요!
1. 버퍼에서 미니배치 샘플링
2. 메인 네트워크로 현재 Q값 계산
3. 타겟 네트워크로 목표 Q값 계산
4. Loss 계산 후 역전파

**`gather`**: 각 상태에서 실제로 선택한 행동의 Q값만 가져오는 거예요.

### [62:00-68:00] 학습 루프 구현

**교사**: (코드 작성하며) 이제 학습 루프를 만들어요:

```python
# 하이퍼파라미터
num_episodes = 500
max_steps = 500
batch_size = 32
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.995
target_update_freq = 1000

# 초기화
env = gym.make('CartPole-v1')
agent = DQNAgent(state_size=4, action_size=2)
epsilon = epsilon_start

# 학습 기록
episode_rewards = []
moving_avg_rewards = []

# 학습
for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0

    for t in range(max_steps):
        # 행동 선택
        action = agent.select_action(state, epsilon)

        # 환경 실행
        next_state, reward, done, truncated, _ = env.step(action)

        # Replay Buffer에 저장
        agent.replay_buffer.store(state, action, reward, next_state, done or truncated)

        # 학습
        loss = agent.learn(batch_size)

        # 타겟 네트워크 업데이트
        agent.step_count += 1
        if agent.step_count % target_update_freq == 0:
            agent.update_target_network()

        episode_reward += reward
        state = next_state

        if done or truncated:
            break

    # ε 감소
    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    # 기록
    episode_rewards.append(episode_reward)
    moving_avg = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
    moving_avg_rewards.append(moving_avg)

    # 진행 상황 출력
    if episode % 10 == 0:
        print(f"Episode {episode:3d} | Reward: {episode_reward:6.2f} | "
              f"Avg: {moving_avg:6.2f} | ε: {epsilon:.3f}")

print("학습 완료!")
```

### [68:00-70:00] 학습 실행

**교사**: (실행하며) 자, 이제 실행해봅시다!

(학습 진행 중 출력 관찰)

```
Episode   0 | Reward:  23.00 | Avg:  23.00 | ε: 1.000
Episode  10 | Reward:  18.00 | Avg:  25.45 | ε: 0.951
Episode  20 | Reward:  45.00 | Avg:  29.67 | ε: 0.904
...
Episode 100 | Reward: 125.00 | Avg:  68.34 | ε: 0.606
...
Episode 200 | Reward: 200.00 | Avg: 165.23 | ε: 0.367
...
Episode 300 | Reward: 200.00 | Avg: 195.67 | ε: 0.222
```

**교사**: 보세요! 처음에는 20~30 정도였는데, 점점 좋아지고 있죠? Episode 200쯤 되니까 거의 200에 가까워져요. 학습이 되고 있어요!

---

## 이론 3: DQN 하이퍼파라미터 (10분)

### [70:00-73:00] 하이퍼파라미터 소개

**교사**: 학습 결과를 보니까 궁금한 게 있죠? "왜 batch_size는 32인가요?" "왜 epsilon은 0.995씩 감소하나요?"

이런 값들을 **하이퍼파라미터**라고 해요. 모델의 구조나 학습 과정을 결정하는 값들이죠.

**주요 하이퍼파라미터**:
1. **Learning Rate (lr)**: 얼마나 크게 가중치를 변경할 것인가
2. **Gamma (γ)**: 미래 보상을 얼마나 중시할 것인가
3. **Epsilon**: 탐험을 얼마나 할 것인가
4. **Buffer Size**: 경험을 얼마나 저장할 것인가
5. **Batch Size**: 한 번에 몇 개 경험으로 학습할 것인가
6. **Target Update Frequency**: 타겟 네트워크를 얼마나 자주 업데이트할 것인가

### [73:00-76:00] 각 하이퍼파라미터의 영향

**교사**: (표 보여주며)

**1. Learning Rate**
```
lr = 0.01:   빠르지만 불안정
lr = 0.001:  적당 (권장)
lr = 0.0001: 느리지만 안정적
```

**2. Gamma (할인율)**
```
γ = 0.9:   단기 보상 중시
γ = 0.99:  장기 보상 중시 (권장)
γ = 0.999: 매우 장기적
```

**3. Epsilon Decay**
```
decay = 0.99:  빠르게 감소 → 탐험 부족 가능
decay = 0.995: 적당 (권장)
decay = 0.999: 느리게 감소 → 학습 느림
```

**4. Buffer Size**
```
1000:   작음 → 최근 경험만
10000:  적당 (권장)
100000: 큼 → 다양한 경험
```

**5. Batch Size**
```
16:  빠르지만 불안정
32:  적당 (권장)
64:  안정적이지만 느림
```

**6. Target Update Frequency**
```
100:   자주 업데이트 → 불안정
1000:  적당 (권장)
10000: 느리게 업데이트 → 학습 느림
```

### [76:00-80:00] 하이퍼파라미터 튜닝 전략

**교사**: 하이퍼파라미터를 어떻게 설정해야 할까요?

**전략 1: 검증된 기본값으로 시작**
```python
# CartPole 권장 설정
lr = 0.001
gamma = 0.99
epsilon_decay = 0.995
buffer_size = 10000
batch_size = 32
target_update_freq = 1000
```

**전략 2: 한 번에 하나씩 변경**
```python
# 잘못된 방법 ✗
실험 1: lr=0.0001, batch_size=64, decay=0.99 (동시 변경)
→ 어떤 게 영향 미쳤는지 모름

# 올바른 방법 ✓
실험 1: lr=0.0001 (나머지 기본값)
실험 2: lr=0.001  (나머지 기본값)
실험 3: lr=0.01   (나머지 기본값)
→ lr의 영향 명확히 파악
```

**전략 3: 학습 곡선 관찰**
```
좋은 곡선: 점점 증가 ✓
나쁜 곡선 1: 요동침 → lr 너무 큼 또는 batch 너무 작음
나쁜 곡선 2: 평평함 → lr 너무 작음 또는 탐험 부족
```

---

## 실습 2: 하이퍼파라미터 실험 (15분)

### [80:00-82:00] 실험 설명

**교사**: 이제 직접 하이퍼파라미터를 바꿔보며 실험해봅시다! 세 가지 실험을 할 거예요:

1. Learning Rate 비교 (0.0001 vs 0.001 vs 0.01)
2. Target Update Frequency 비교 (100 vs 1000 vs 10000)
3. Batch Size 비교 (16 vs 32 vs 64)

### [82:00-87:00] 실험 코드 작성

**교사**: (코드 작성하며)

```python
import matplotlib.pyplot as plt

def train_dqn(lr=0.001, target_update_freq=1000, batch_size=32,
              num_episodes=300, label="Default"):
    """DQN 학습 함수"""
    env = gym.make('CartPole-v1')
    agent = DQNAgent(state_size=4, action_size=2, lr=lr)
    epsilon = 1.0

    episode_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(500):
            action = agent.select_action(state, epsilon)
            next_state, reward, done, truncated, _ = env.step(action)
            agent.replay_buffer.store(state, action, reward, next_state, done or truncated)
            agent.learn(batch_size)

            agent.step_count += 1
            if agent.step_count % target_update_freq == 0:
                agent.update_target_network()

            episode_reward += reward
            state = next_state
            if done or truncated:
                break

        epsilon = max(0.01, epsilon * 0.995)
        episode_rewards.append(episode_reward)

    # 이동평균 계산
    moving_avg = []
    for i in range(len(episode_rewards)):
        avg = sum(episode_rewards[max(0, i-99):i+1]) / min(i+1, 100)
        moving_avg.append(avg)

    return moving_avg

# 실험 1: Learning Rate
print("실험 1: Learning Rate 비교")
results_lr = {}
for lr in [0.0001, 0.001, 0.01]:
    print(f"  lr = {lr} 학습 중...")
    results_lr[lr] = train_dqn(lr=lr, label=f"lr={lr}")

# 그래프 그리기
plt.figure(figsize=(10, 6))
for lr, rewards in results_lr.items():
    plt.plot(rewards, label=f"lr={lr}")
plt.xlabel('Episode')
plt.ylabel('Moving Average Reward')
plt.title('Learning Rate Comparison')
plt.legend()
plt.grid()
plt.savefig('lr_comparison.png')
plt.show()
```

### [87:00-90:00] 결과 분석

**교사**: (그래프 보여주며) 결과를 볼까요?

**Learning Rate 비교**:
- lr=0.01: 초반에 빠르지만 불안정하게 요동침
- lr=0.001: 안정적으로 증가 (최고!)
- lr=0.0001: 너무 느림, 300 에피소드로 부족

**Target Update Frequency 비교**:
- 100: 약간 불안정
- 1000: 가장 안정적
- 10000: 학습이 느림

**Batch Size 비교**:
- 16: 불안정
- 32: 적당
- 64: 안정적이지만 약간 느림

**교사**: 보시다시피, 하이퍼파라미터에 따라 학습 속도와 안정성이 크게 달라져요. 문제마다 최적의 값이 다르니까, 실험을 통해 찾아야 해요!

---

## 정리 및 다음 주 예고 (5분)

### [90:00-92:00] 핵심 정리

**교사**: 오늘 배운 내용을 정리해볼게요!

**DQN의 3가지 핵심**:
1. **신경망으로 Q 함수 근사** → 큰 상태 공간 처리
2. **Experience Replay** → 상관관계 제거, 안정적 학습
3. **Target Network** → 고정된 목표, 안정적 수렴

**언제 DQN을 사용하나?**
- 상태 공간이 큼 (Q-table 불가능)
- 행동 공간은 이산적
- 예: Atari, ATAXX, Betris

**하이퍼파라미터**:
- 기본값으로 시작
- 한 번에 하나씩 변경
- 학습 곡선으로 평가

### [92:00-94:00] Betris 적용 힌트

**교사**: ALPHANO Betris 문제에 DQN을 적용하려면:

1. **상태 표현**: 5×5 보드 + 점수 + 코인 + 현재 블록
2. **행동 공간**: 베팅 + 배치 (계층적 접근 권장)
3. **보상 설계**: 줄 제거 보너스 + 실제 점수
4. **네트워크**: CNN으로 보드 패턴 인식

**단계적 개발**:
- 먼저 베팅 고정, 배치만 학습
- 그다음 베팅 추가
- 충분한 에피소드 학습 (10000+)

### [94:00-95:00] 다음 주 예고

**교사**: 다음 주에는 **Policy Gradient**를 배워요!

DQN의 한계:
- 이산 행동만 가능
- 연속 행동 (예: 각도 0~360°) 불가능

Policy Gradient:
- 정책을 직접 학습
- 연속 행동 가능
- REINFORCE, Actor-Critic

**준비 사항**:
- DQN 코드를 충분히 이해하기
- PyTorch 기본 문법 복습
- 확률 분포 개념 (정규분포)

### [95:00] 질의응답 및 마무리

**교사**: 질문 있나요?

(학생 질문 받고 답변)

**교사**: 오늘 고생 많았어요! 집에 가서 CartPole 코드를 다시 실행해보고, 하이퍼파라미터를 바꿔가며 실험해보세요. Betris 상태 공간도 설계해보면 좋겠어요. 다음 주에 봐요!

---

## 보충 자료: 학생 질문 예상 답변

### Q1: "왜 신경망이 일반화를 할 수 있나요?"

**답변**: 신경망은 데이터의 **패턴**을 학습하기 때문이에요. 예를 들어, [0.1, 0.5, 0.2, 0.3]이라는 상태를 학습했다면, [0.11, 0.48, 0.21, 0.29]라는 비슷한 상태에 대해서도 비슷한 Q값을 출력해요. 왜냐하면 신경망의 가중치가 **부드러운(smooth) 함수**를 만들기 때문이죠. 입력이 조금 바뀌면 출력도 조금만 바뀌어요.

### Q2: "Replay Buffer가 가득 차면 어떻게 되나요?"

**답변**: `deque`의 `maxlen` 기능 덕분에 자동으로 가장 오래된 경험이 삭제돼요. 예를 들어 capacity=10000인데 10001번째 경험을 저장하면, 1번째 경험이 자동으로 사라지고 10001번째가 들어가요. 선입선출(FIFO)이라고 생각하면 돼요!

### Q3: "타겟 네트워크를 왜 복사하나요? 공유하면 안 되나요?"

**답변**: 공유하면 안 돼요! 메인 네트워크와 타겟 네트워크가 같은 객체를 가리키면, 메인 네트워크가 업데이트될 때 타겟 네트워크도 함께 바뀌어요. 그럼 "고정된 목표"가 아니게 되죠. `copy.deepcopy()`를 사용하면 **완전히 별개의 네트워크**가 만들어져서, 메인 네트워크가 바뀌어도 타겟 네트워크는 그대로 유지돼요.

### Q4: "DQN으로 바둑이나 체스를 할 수 있나요?"

**답변**: 이론적으로는 가능하지만 현실적으로 매우 어려워요. 바둑은 행동 공간이 너무 크고 (19×19 = 361개 위치), 게임이 너무 길어요. AlphaGo는 DQN이 아니라 **Monte Carlo Tree Search + Policy Network + Value Network**를 결합한 훨씬 복잡한 방법을 사용했어요. 체스도 비슷하고요. DQN은 Atari 정도 복잡도의 게임에 적합해요.

### Q5: "왜 batch_size를 사용하나요? 하나씩 학습하면 안 되나요?"

**답변**: 미니배치를 사용하는 이유는:
1. **안정성**: 여러 경험의 평균 그래디언트를 사용하면 노이즈가 줄어들어요
2. **효율성**: GPU는 병렬 처리에 최적화돼 있어서, 32개를 동시에 처리하는 게 32번 처리하는 것보다 훨씬 빨라요
3. **일반화**: 다양한 경험을 동시에 보면 특정 경험에 과적합되는 것을 방지해요

하나씩 학습하면 학습 곡선이 매우 불안정해져요!

### Q6: "epsilon이 0.01까지만 감소하는 이유는?"

**답변**: 완전히 0으로 만들지 않는 이유는 **탐험(exploration)**을 계속 유지하기 위해서예요. 만약 epsilon=0이 되면, 에이전트는 항상 현재 Q값이 높은 행동만 선택해요. 그런데 Q값이 잘못 학습됐을 수도 있잖아요? 1% 정도는 랜덤으로 행동해서 새로운 경험을 계속 얻는 게 중요해요. 이걸 "영구적 탐험"이라고 해요.

### Q7: "학습이 잘 안 될 때 어떻게 디버깅하나요?"

**답변**: 체크리스트:
1. **보상 확인**: 보상이 제대로 설정됐나? (너무 sparse하지 않나?)
2. **네트워크 크기**: 너무 작으면 표현력 부족, 너무 크면 과적합
3. **학습률**: 너무 크면 발산, 너무 작으면 학습 안 됨
4. **Replay Buffer**: 충분히 찼나? (최소 batch_size 이상)
5. **탐험**: epsilon이 너무 빨리 감소하지 않나?
6. **타겟 네트워크**: 업데이트 주기가 적절한가?

하나씩 확인하며 실험해보세요!

---

**수업 준비 노트**:
- CartPole 학습은 시간이 걸리므로, 미리 학습된 결과를 준비해두기
- 학생들이 코드를 따라 칠 시간을 충분히 주기
- 실습 중 에러가 나면 침착하게 디버깅하며 설명
- 그래프를 보여줄 때 학습 곡선의 의미를 강조
- Betris 적용은 개념만 설명, 다음 주 과제로 남기기
