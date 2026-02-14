# Week 7: Deep Q-Network (DQN)

## 개요

이 주차에서는 **Deep Q-Network (DQN)**을 학습합니다. DQN은 Q-Learning의 한계를 극복하고 큰 상태 공간에서도 작동하는 강화학습 알고리즘입니다.

## 디렉토리 구조

```
week07_dqn/
├── README.md                           # 이 파일
├── lecture.md                          # 수업 자료 (500+ lines)
├── script.md                           # 수업 대본 (90분)
├── practice/
│   ├── dqn_cartpole.py                # CartPole DQN 완전 구현
│   └── dqn_hyperparameter_exp.py      # 하이퍼파라미터 실험
└── alphano/
    └── betris_dqn_agent.py            # Betris ALPHANO 에이전트
```

## 학습 목표

1. **Q-Learning의 한계 이해**: 큰 상태 공간에서 Q-table이 작동하지 않는 이유
2. **함수 근사**: 신경망으로 Q 함수를 근사하는 방법
3. **DQN 핵심 기법**: Experience Replay와 Target Network
4. **PyTorch 기초**: 신경망 구현 및 학습
5. **실전 적용**: CartPole 해결 및 Betris 적용 방법

## 주요 내용

### 1. Q-Learning의 한계

- **작은 상태 공간**: Q-table 사용 가능 (예: GridWorld 5×5)
- **큰 상태 공간**: Q-table 불가능 (예: ATAXX 7×7 → 3^49 상태)
- **해결책**: 신경망으로 Q 함수 근사

### 2. 함수 근사

```python
# Q-table 방식
Q(s, a) = table[s][a]  # 정확히 저장된 값

# 함수 근사 방식
Q(s, a; θ) = neural_network(s, a)  # 신경망으로 계산
```

**장점**:
- 메모리 효율적 (파라미터 θ만 저장)
- 일반화 가능 (비슷한 상태에 비슷한 Q값)
- 본 적 없는 상태도 추론 가능

### 3. DQN 핵심 기법

#### Experience Replay
```python
# 경험 저장
replay_buffer.store(state, action, reward, next_state, done)

# 랜덤 샘플링
batch = replay_buffer.sample(batch_size=32)

# 학습
for (s, a, r, s', done) in batch:
    Q_update(s, a, r, s')
```

**효과**:
- 상관관계 제거 (연속된 경험의 문제 해결)
- 데이터 효율성 (경험 재사용)
- 학습 안정성 증가

#### Target Network
```python
# 메인 네트워크: 예측 및 학습
current_q = q_network(state, action)

# 타겟 네트워크: 목표값 계산만
target_q = reward + gamma * target_network(next_state).max()

# Loss
loss = (current_q - target_q)^2
```

**효과**:
- 고정된 학습 목표 제공
- 학습 안정성 증가
- 발산 방지

### 4. DQN 알고리즘

```
초기화:
  Replay Buffer D
  Q-Network θ
  Target Network θ⁻ = θ

for episode in episodes:
    state = env.reset()
    for t in steps:
        # 행동 선택 (ε-greedy)
        action = ε-greedy(Q(state; θ))

        # 환경 실행
        next_state, reward, done = env.step(action)

        # 경험 저장
        D.store(state, action, reward, next_state, done)

        # 학습
        batch = D.sample(batch_size)
        target = reward + γ·max Q(next_state; θ⁻)
        loss = (Q(state, action; θ) - target)²
        θ ← θ - α·∇loss

        # 타겟 네트워크 업데이트
        if t % C == 0:
            θ⁻ ← θ
```

## 실습

### 1. CartPole DQN

**목표**: 막대를 쓰러뜨리지 않고 200 타임스텝 이상 유지

```bash
cd practice
python dqn_cartpole.py
```

**기대 결과**:
- Episode 100: 평균 50~70
- Episode 200: 평균 150~170
- Episode 300: 평균 195+ (목표 달성!)

**학습 곡선**:
```
Episode   0 | Reward:  23.00 | Avg:  23.00 | ε: 1.000
Episode  50 | Reward:  45.00 | Avg:  38.50 | ε: 0.778
Episode 100 | Reward: 125.00 | Avg:  68.34 | ε: 0.606
Episode 150 | Reward: 180.00 | Avg: 132.45 | ε: 0.471
Episode 200 | Reward: 200.00 | Avg: 165.23 | ε: 0.367
Episode 250 | Reward: 200.00 | Avg: 185.67 | ε: 0.285
Episode 300 | Reward: 200.00 | Avg: 195.89 | ε: 0.222
```

### 2. 하이퍼파라미터 실험

```bash
cd practice
python dqn_hyperparameter_exp.py
```

**실험 항목**:
1. Learning Rate (0.0001, 0.001, 0.01)
2. Target Update Frequency (100, 1000, 10000)
3. Batch Size (16, 32, 64, 128)
4. Replay Buffer Size (1000, 5000, 10000)

**개별 실험 실행**:
```bash
python dqn_hyperparameter_exp.py lr      # Learning Rate만
python dqn_hyperparameter_exp.py freq    # Target Update Frequency만
python dqn_hyperparameter_exp.py batch   # Batch Size만
python dqn_hyperparameter_exp.py buffer  # Buffer Size만
```

### 3. Betris ALPHANO 에이전트

```bash
cd alphano
python betris_dqn_agent.py
```

**주의사항**:
- 이 코드는 Betris 프로토콜을 정확히 알지 못한 상태의 개념 구현입니다
- 실제 제출 전에 ALPHANO Betris 프로토콜을 확인해야 합니다
- DQN 실전 적용은 사전 학습이 필요합니다 (여기서는 휴리스틱 사용)

## 하이퍼파라미터 가이드

### CartPole 권장 설정

```python
lr = 0.001                  # 학습률
gamma = 0.99                # 할인율
epsilon_start = 1.0         # 초기 탐험 확률
epsilon_end = 0.01          # 최소 탐험 확률
epsilon_decay = 0.995       # 탐험 확률 감소율
buffer_size = 10000         # Replay Buffer 크기
batch_size = 32             # 배치 크기
target_update_freq = 1000   # Target Network 업데이트 주기
```

### 하이퍼파라미터 의미

| 파라미터 | 너무 작으면 | 적당 | 너무 크면 |
|----------|-------------|------|-----------|
| Learning Rate | 학습 느림 | 0.001 | 발산, 불안정 |
| Gamma | 단기적 | 0.99 | 수렴 느림 |
| Buffer Size | 다양성 부족 | 10000 | 메모리 부족 |
| Batch Size | 불안정 | 32 | 느림 |
| Target Update Freq | 불안정 | 1000 | 학습 느림 |

## 문제 해결 가이드

### 학습이 안 될 때

**증상**: 보상이 계속 낮음

**원인 및 해결**:
- Learning Rate 너무 낮음 → 0.001로 증가
- Epsilon이 너무 낮음 → Decay 느리게 (0.995)
- Network 너무 작음 → Hidden Layer 크기 증가

### 학습이 불안정할 때

**증상**: 보상이 요동침

**원인 및 해결**:
- Learning Rate 너무 높음 → 0.0001로 감소
- Batch Size 너무 작음 → 32 이상
- Target Update 너무 빈번 → 1000 이상

### 과적합

**증상**: 특정 상황에서만 잘함

**원인 및 해결**:
- Buffer Size 너무 작음 → 10000 이상
- Epsilon이 너무 빨리 감소 → Decay 느리게

## DQN vs Q-Learning

| 항목 | Q-Learning | DQN |
|------|-----------|-----|
| Q 표현 | Table | Neural Network |
| 상태 공간 | 작음 | 큼 (무한대 가능) |
| 일반화 | 불가능 | 가능 |
| 메모리 | O(S×A) | O(θ) |
| 학습 안정성 | 높음 | 낮음 (기법 필요) |
| 경험 사용 | 1회 | 여러 번 |

## 언제 DQN을 사용하는가?

### DQN이 적합한 경우 ✓

- 상태 공간이 매우 큼 (예: 이미지)
- 연속적 상태 공간
- 비슷한 상태에 비슷한 행동 필요
- 행동 공간은 이산적이고 작음

### DQN이 부적합한 경우 ✗

- 상태 공간이 작음 (Q-table으로 충분)
- 연속적 행동 공간 (Policy Gradient 사용)
- 실시간 빠른 결정 필요
- 학습 데이터가 매우 부족

## 다음 단계

### Week 8 예고: Policy Gradient

DQN의 한계:
- **이산 행동만 가능**: 연속 행동 불가능
- 예: 로봇 팔 각도, 자동차 핸들 등

Policy Gradient:
- **정책을 직접 학습**: π(a|s)
- **연속 행동 가능**: 정규분포 등 사용
- **REINFORCE**, **Actor-Critic** 알고리즘

## 참고 자료

### 논문
- [Playing Atari with Deep Reinforcement Learning (DQN, 2013)](https://arxiv.org/abs/1312.5602)
- [Human-level control through deep reinforcement learning (Nature DQN, 2015)](https://www.nature.com/articles/nature14236)

### 온라인 자료
- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [PyTorch 공식 튜토리얼](https://pytorch.org/tutorials/)
- [Gymnasium 문서](https://gymnasium.farama.org/)

### 개선 기법
- **Double DQN**: 과대평가 문제 해결
- **Dueling DQN**: 네트워크 구조 개선
- **Prioritized Experience Replay**: 중요한 경험 우선 학습
- **Rainbow DQN**: 여러 기법 통합

## 연습 문제

1. Replay Buffer를 직접 구현하고 테스트해보세요
2. CartPole을 200점 이상 달성해보세요
3. 하이퍼파라미터를 바꿔가며 학습 곡선을 비교해보세요
4. Betris 상태 공간을 3가지 방법으로 설계해보세요
5. Double DQN을 구현해보세요 (도전!)

## 체크리스트

오늘 배운 내용을 확인해보세요:

- [ ] Q-Learning의 한계를 이해했다
- [ ] 함수 근사의 개념을 이해했다
- [ ] PyTorch로 간단한 신경망을 만들 수 있다
- [ ] Experience Replay의 원리를 이해했다
- [ ] Target Network의 필요성을 이해했다
- [ ] DQN 전체 알고리즘을 이해했다
- [ ] CartPole DQN 코드를 실행했다
- [ ] 하이퍼파라미터의 영향을 확인했다
- [ ] Betris에 DQN을 적용하는 방법을 생각해봤다

## 라이선스

이 자료는 교육 목적으로 제작되었습니다.

---

**다음 주차**: [Week 8 - Policy Gradient](../week08_policy_gradient/)

**이전 주차**: [Week 6 - Q-Learning](../week06_qlearning/)
