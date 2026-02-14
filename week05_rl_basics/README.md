# Week 5: 강화학습 기초 - MDP, Value Iteration, Policy Iteration

## 개요

이번 주차에서는 강화학습의 기초 개념을 학습하고, Markov Decision Process(MDP)를 이해하며, Value Iteration과 Policy Iteration 알고리즘을 직접 구현합니다.

## 학습 목표

- 강화학습의 기본 개념과 게임 탐색과의 차이점 이해
- MDP(Markov Decision Process)의 정의와 구성 요소 학습
- 가치 함수와 벨만 방정식의 의미 파악
- Value Iteration과 Policy Iteration 알고리즘 구현
- GridWorld 환경에서 최적 정책 도출

## 주요 개념

### 1. 강화학습 (Reinforcement Learning)

에이전트가 환경과 상호작용하며 **보상(Reward)**을 최대화하는 방법을 학습하는 기계학습 분야

```
┌─────────┐                    ┌─────────────┐
│         │   Action (a_t)     │             │
│  Agent  │──────────────────> │ Environment │
│         │                    │             │
│         │ <──────────────────│             │
└─────────┘  State (s_t)       └─────────────┘
              Reward (r_t)
```

**핵심 특징:**
- Trial and Error (시행착오)
- Delayed Reward (지연된 보상)
- Exploration vs Exploitation (탐색 vs 활용)

### 2. MDP (Markov Decision Process)

강화학습 문제를 수학적으로 모델링하는 표준 방법

**구성 요소: (S, A, P, R, γ)**
- **S**: 상태 공간 (State Space)
- **A**: 행동 공간 (Action Space)
- **P**: 전이 확률 (Transition Probability)
- **R**: 보상 함수 (Reward Function)
- **γ**: 감가율 (Discount Factor, 0 ≤ γ < 1)

**Markov Property**: 미래는 현재 상태에만 의존하며, 과거 이력과는 무관

### 3. 가치 함수 (Value Function)

**상태 가치 함수:**
```
V^π(s) = E_π[G_t | S_t = s]
       = "상태 s에서 정책 π를 따를 때의 기대 수익"
```

**행동 가치 함수:**
```
Q^π(s, a) = E_π[G_t | S_t = s, A_t = a]
          = "상태 s에서 행동 a를 한 후 정책 π를 따를 때의 기대 수익"
```

### 4. 벨만 방정식 (Bellman Equation)

**벨만 기대 방정식:**
```
V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R(s,a,s') + γV^π(s')]
```

**벨만 최적 방정식:**
```
V*(s) = max_a Σ_{s'} P(s'|s,a)[R(s,a,s') + γV*(s')]
```

## 파일 구조

```
week05_rl_basics/
├── README.md                    # 이 파일
├── lecture.md                   # 강의 자료 (500+ lines)
├── script.md                    # 수업 대본 (600+ lines)
└── practice/
    ├── gridworld.py             # GridWorld 환경 구현
    ├── value_iteration.py       # Value Iteration 알고리즘
    └── policy_iteration.py      # Policy Iteration 알고리즘
```

## 실습 가이드

### 1. GridWorld 환경

4×4 격자 세계에서 에이전트가 목표를 찾아가는 환경

```python
from practice.gridworld import GridWorld

# 환경 생성
env = GridWorld(grid_size=4)

# 초기 상태
state = env.reset()  # (0, 0)

# 행동 수행
next_state, reward, done = env.step(None, GridWorld.ACTION_RIGHT)

# 시각화
env.render()
```

**환경 설정:**
```
┌─────┬─────┬─────┬─────┐
│  S  │     │     │     │  S: 시작 (Start)
├─────┼─────┼─────┼─────┤  G: 목표 (Goal)
│     │  X  │     │     │  X: 장애물
├─────┼─────┼─────┼─────┤
│     │     │  X  │     │
├─────┼─────┼─────┼─────┤
│     │     │     │  G  │
└─────┴─────┴─────┴─────┘
```

**보상:**
- 목표 도달: +1.0
- 장애물: -1.0
- 일반 이동: -0.04

### 2. Value Iteration

벨만 최적 방정식을 반복 적용하여 최적 가치 함수와 정책을 찾는 알고리즘

```python
from practice.value_iteration import value_iteration
from practice.gridworld import GridWorld

env = GridWorld()
V, policy = value_iteration(env, gamma=0.9, theta=0.001)

# 결과 출력
env.render_policy(policy)
env.render_values(V)
```

**알고리즘 구조:**
```
1. V(s) = 0으로 초기화
2. 반복:
   for each state s:
     V(s) = max_a [R + γV(s')]
3. 수렴할 때까지
4. 정책 추출: π(s) = argmax_a Q(s,a)
```

**실행 예시:**
```bash
python practice/value_iteration.py
```

### 3. Policy Iteration

정책 평가와 정책 개선을 번갈아 수행하여 최적 정책을 찾는 알고리즘

```python
from practice.policy_iteration import policy_iteration
from practice.gridworld import GridWorld

env = GridWorld()
policy, V = policy_iteration(env, gamma=0.9, theta=0.001)

# 결과 출력
env.render_policy(policy)
env.render_values(V)
```

**알고리즘 구조:**
```
1. π를 임의로 초기화
2. 반복:
   a. 정책 평가: V^π 계산 (수렴까지)
   b. 정책 개선: π' ← greedy(V^π)
   c. if π' == π: break
3. return π
```

**실행 예시:**
```bash
python practice/policy_iteration.py
```

## 실습 과제

### 기초 과제

**1. GridWorld 환경 탐색**
- GridWorld 환경을 생성하고 수동으로 몇 번 이동해보기
- 목표 도달, 장애물 충돌, 벽 부딪히기 테스트
- 각 경우의 보상 확인

**2. Value Iteration 실행**
- 기본 설정(γ=0.9)으로 Value Iteration 실행
- 수렴 과정 관찰
- 최종 정책과 가치 함수 분석

**3. Policy Iteration 실행**
- 기본 설정으로 Policy Iteration 실행
- Value Iteration과 결과 비교
- 수렴 속도 비교

### 중급 과제

**4. Gamma 값 실험**
- γ = 0.5, 0.7, 0.9, 0.99로 Value Iteration 실행
- 각 γ 값에 따른 정책 변화 관찰
- 왜 정책이 달라지는지 분석

**5. 환경 수정**
- 장애물 위치 변경 또는 추가
- 보상 값 조정 (목표 보상, 이동 패널티)
- 변경된 환경에서 알고리즘 재실행

**6. 알고리즘 비교 분석**
- Value Iteration vs Policy Iteration
- 반복 횟수, 실행 시간, 최종 정책 비교
- 각 알고리즘의 장단점 정리

### 도전 과제

**7. 큰 격자 세계**
- 6×6 또는 8×8 GridWorld 생성
- 장애물 무작위 배치
- 수렴 시간과 정책 복잡도 분석

**8. 확률적 환경**
- GridWorld를 확률적으로 수정:
  - 의도한 방향 80%
  - 좌우 각 10%
- Value Iteration 재실행
- 결정적 환경과 정책 비교

**9. 쥐를 잡자 간소화 버전**
- 5×5 보드에서 쥐를 잡자 게임 구현
- 상태: (고양이 위치, 쥐 위치)
- 쥐는 무작위로 이동
- Value Iteration으로 최적 정책 도출

## 주요 개념 정리

### Value Iteration vs Policy Iteration

| 특성 | Value Iteration | Policy Iteration |
|------|------------------|-----------------|
| **업데이트** | 벨만 최적 방정식 | 벨만 기대 방정식 |
| **정책 평가** | 한 번만 | 수렴까지 반복 |
| **반복 횟수** | 많음 (수십~수백) | 적음 (5-10) |
| **반복당 시간** | 빠름 | 느림 |
| **전체 시간** | 빠름 (일반적) | 중간 |
| **적용** | 큰 상태 공간 | 작은 상태 공간 |
| **구현** | 간단 | 복잡 |

### 감가율(γ)의 영향

| γ 값 | 특성 | 행동 경향 | 적용 분야 |
|------|------|-----------|-----------|
| 0.0 ~ 0.5 | 단기 근시안적 | 즉각 보상 추구 | 즉각 반응 시스템 |
| 0.5 ~ 0.9 | 균형잡힌 | 중기 계획 | 대부분의 게임 |
| 0.9 ~ 0.99 | 장기 전략적 | 장기 보상 추구 | 바둑, 체스 |
| 0.99 ~ 1.0 | 매우 장기적 | 먼 미래 중시 | 금융, 자원 관리 |

## 다음 주 예고: Q-Learning

### 한계점

**Value/Policy Iteration의 문제:**
- 모든 상태를 방문해야 함
- 전이 확률 P와 보상 R을 정확히 알아야 함
- 상태 공간이 크면 계산 불가능

### 해결책: Model-Free RL

**Q-Learning (다음 주)**
- 환경 모델(P, R) 없이 학습
- 경험(샘플)으로부터 직접 학습
- 큰 상태 공간에 대응 가능

**주요 내용:**
- Temporal Difference (TD) 학습
- Q-Learning 알고리즘
- ε-greedy 탐색 전략
- 쥐를 잡자 게임에 Q-Learning 적용

## 참고 자료

### 교재
- **Sutton & Barto - Reinforcement Learning: An Introduction**
  - Chapter 3: MDP
  - Chapter 4: Dynamic Programming
  - 무료 온라인: http://incompleteideas.net/book/the-book-2nd.html

### 온라인 강의
- **David Silver's RL Course** (YouTube)
  - Lecture 2: Markov Decision Processes
  - Lecture 3: Planning by Dynamic Programming

### 추가 학습 자료
- OpenAI Spinning Up: https://spinningup.openai.com/
- Reinforcement Learning Korea: https://www.facebook.com/groups/ReinforcementLearningKR/

## 문제 해결

### 자주 발생하는 오류

**1. Value Iteration이 수렴하지 않음**
- theta 값을 키워보기 (0.001 → 0.01)
- max_iterations 값 확인
- 감가율 γ < 1인지 확인

**2. Policy Iteration이 너무 느림**
- 정책 평가의 theta 값 조정
- 작은 격자(2×2, 3×3)에서 먼저 테스트

**3. 정책이 이상함**
- 보상 설계 확인
- 장애물 위치 확인
- γ 값이 적절한지 확인

### 디버깅 팁

1. **작은 예제부터**: 2×2 격자에서 시작
2. **시각화 활용**: render_policy(), render_values() 자주 호출
3. **손 계산 비교**: 간단한 경우 손으로 계산해서 코드와 비교
4. **단계별 확인**: 벨만 방정식을 한 상태에 대해 직접 계산

## 라이선스

이 자료는 교육 목적으로 제작되었습니다.

## 문의

질문이나 버그 리포트는 수업 시간에 문의해주세요.

---

**수고하셨습니다! 강화학습의 기초를 마스터했습니다!** 🎉
