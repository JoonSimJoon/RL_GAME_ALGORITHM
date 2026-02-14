# Week 4: Monte Carlo Tree Search (MCTS)

## 목차
1. [복습: 탐색 기반 접근의 한계](#1-복습-탐색-기반-접근의-한계)
2. [MCTS 개요](#2-mcts-개요)
3. [MCTS 4단계 상세 설명](#3-mcts-4단계-상세-설명)
4. [UCB1 공식 상세](#4-ucb1-공식-상세)
5. [MCTS 의사코드](#5-mcts-의사코드)
6. [MCTS vs Alpha-Beta 비교](#6-mcts-vs-alpha-beta-비교)
7. [MCTS 개선 기법](#7-mcts-개선-기법)
8. [핵심 정리 및 다음 주 예고](#8-핵심-정리-및-다음-주-예고)

---

## 1. 복습: 탐색 기반 접근의 한계

### 1.1 지금까지 배운 것

지난 3주 동안 우리는 게임 AI를 위한 강력한 탐색 알고리즘들을 배웠습니다:

1. **Week 1: Minimax**
   - 게임 트리를 완전히 탐색
   - 최적의 수를 보장 (시간이 충분하다면)
   - 하지만 지수적 시간 복잡도: O(b^d)

2. **Week 2: Alpha-Beta Pruning**
   - 불필요한 가지를 가지치기
   - 최악의 경우 Minimax와 동일하지만, 최선의 경우 O(b^(d/2))
   - Principal Variation Search로 더 개선

3. **Week 3: PVS + 평가 함수**
   - 실전에서 사용 가능한 성능
   - 좋은 평가 함수로 제한된 깊이에서도 강력

### 1.2 Alpha-Beta의 강점

Alpha-Beta + 좋은 평가 함수는 매우 강력합니다:

```
장점:
✓ 완전 정보 게임에서 정확한 판단
✓ 깊이 제한으로 시간 조절 가능
✓ 평가 함수가 좋으면 얕은 깊이에서도 강력
✓ 구현이 직관적
```

실제로 체스 엔진(Stockfish), 오셀로 프로그램 등 많은 보드게임 AI가 Alpha-Beta 기반입니다.

### 1.3 Alpha-Beta의 한계

하지만 모든 게임에 Alpha-Beta가 완벽한 것은 아닙니다:

#### 문제 1: 평가 함수 설계의 어려움

좋은 평가 함수를 만들기 어려운 게임들이 있습니다:

**바둑 (Go)**
- 19×19 = 361개의 교점
- 국면 평가가 매우 복잡
- "이 돌의 가치는?"을 수치화하기 어려움
- 전문가도 직관에 의존

**복잡한 전략 게임**
- 여러 요소가 복합적으로 작용
- 단순 합산으로는 국면 평가 불가능

예시: ATAXX에서도 완벽한 평가 함수는 어렵습니다
```python
def evaluate(board):
    # 이것만으로 충분할까?
    my_count = count_my_pieces(board)
    opp_count = count_opponent_pieces(board)
    return my_count - opp_count

    # 아니면 이런 것도 고려해야 할까?
    # - 중앙 위치의 가치
    # - 이동 가능한 수의 개수
    # - 고립된 돌의 패널티
    # - 코너 위치의 가치
    # 어떻게 가중치를 정해야 할까?
```

#### 문제 2: 수평선 효과 (Horizon Effect)

깊이 제한 때문에 발생하는 문제:

```
현재 깊이 6에서 탐색 중...

깊이 6: "좋아 보이는 국면" (평가값 +5)
  └─ 하지만 깊이 7에서: "치명적인 패배" (평가값 -100)

깊이 제한 때문에 깊이 7을 못 보고 나쁜 수를 선택!
```

#### 문제 3: 평가 함수의 부정확성

평가 함수는 어디까지나 "추정치"입니다:

```
실제 게임 결과:  승 / 패 / 무승부
평가 함수 출력: +3.7 / -2.1 / +0.8 ???

평가값과 실제 결과 사이에는 항상 괴리가 있습니다.
```

### 1.4 새로운 접근: 시뮬레이션 기반

그렇다면 **평가 함수 없이도** 좋은 수를 찾을 수 있을까요?

**핵심 아이디어:**
> "평가 함수로 추정하지 말고, 실제로 게임을 끝까지 해보자!"

예시:
```
수 A를 두면?
  → 게임 끝까지 랜덤으로 진행 → 승리! (1점)
  → 다시 랜덤으로 진행 → 패배... (0점)
  → 다시 랜덤으로 진행 → 승리! (1점)
  → ... 1000번 반복
  → 승률: 620/1000 = 62%

수 B를 두면?
  → ... 1000번 반복
  → 승률: 450/1000 = 45%

결론: 수 A가 더 좋다!
```

이것이 바로 **Monte Carlo Tree Search (MCTS)**의 기본 아이디어입니다.

---

## 2. MCTS 개요

### 2.1 Monte Carlo 방법이란?

**Monte Carlo 방법**: 무작위 샘플링을 반복해서 결과를 추정하는 기법

일상 예시:
```
질문: "이 가방에 빨간 구슬이 몇 %일까?"

전통적 방법: 모든 구슬을 꺼내서 센다
Monte Carlo: 무작위로 100개 뽑아서 비율 추정
  → 빨간 구슬 37개 발견
  → 추정: 약 37%
```

게임 AI에 적용:
```
질문: "이 수가 좋은 수일까?"

전통적 방법 (Alpha-Beta): 평가 함수로 추정
Monte Carlo: 실제로 여러 번 게임을 끝까지 해봄
  → 1000번 시뮬레이션
  → 620번 승리
  → 추정: 62% 승률, 좋은 수!
```

### 2.2 MCTS의 핵심 개념

MCTS는 **시뮬레이션 기반 트리 탐색**입니다:

**주요 특징:**
1. **평가 함수 불필요**: 실제 게임 결과(승/패)를 사용
2. **점진적 확장**: 중요한 부분만 집중 탐색
3. **확률적 접근**: 랜덤 시뮬레이션으로 가능성 평가
4. **시간 활용**: 주어진 시간만큼 계속 개선

### 2.3 기본 작동 원리

간단한 예시로 이해해봅시다:

```
초기 상태: 루트 노드만 존재
           [루트]
            /  \
         수A   수B

1단계: 수A 선택 → 게임 끝까지 랜덤 플레이 → 승리!
      [루트: 1승/1경기]
        /
     [A: 1승/1경기]

2단계: 수B도 탐색 → 랜덤 플레이 → 패배
      [루트: 1승/2경기]
        /              \
     [A: 1승/1경기]   [B: 0승/1경기]

3단계: A가 승률 높으니 A를 더 탐색
      [루트: 2승/3경기]
        /              \
     [A: 2승/2경기]   [B: 0승/1경기]
        /
     [A1: 1승/1경기]

... 반복하면서 트리 확장 및 통계 누적
```

수천 번 반복 후, **가장 많이 방문된 수**를 선택합니다.

### 2.4 왜 MCTS가 효과적인가?

**1. 평가 함수의 부정확성 제거**
- 추정치(평가값) 대신 실제 결과(승/패) 사용
- 평가 함수 설계 불필요

**2. 중요한 변화 집중 탐색**
- 좋은 수는 자주 시뮬레이션
- 나쁜 수는 빨리 포기
- 제한된 시간을 효율적으로 사용

**3. 언제든지 중단 가능 (Anytime Algorithm)**
- 1초 주어지면 1초만큼 탐색
- 10초 주어지면 10초만큼 탐색
- 더 오래 탐색할수록 정확도 향상

**4. 병렬화 용이**
- 시뮬레이션들이 독립적
- 멀티코어 활용 가능

### 2.5 MCTS의 성공 사례

**바둑 (AlphaGo)**
- 2016년, AlphaGo가 이세돌 9단을 4:1로 격파
- MCTS + 딥러닝의 조합
- 평가 함수 설계가 어려운 바둑에서 혁명적 성과

**기타 게임**
- Hex
- General Game Playing 대회
- 불완전 정보 게임 (포커 등)에도 응용

---

## 3. MCTS 4단계 상세 설명

MCTS는 4개의 단계를 **반복**합니다:

```
┌─────────────────────────────────────┐
│  MCTS 1회 Iteration                 │
│                                     │
│  1. Selection (선택)                │
│  2. Expansion (확장)                │
│  3. Simulation (시뮬레이션)         │
│  4. Backpropagation (역전파)        │
└─────────────────────────────────────┘
         ↓
   수천~수만 번 반복
         ↓
   최종 수 결정
```

각 단계를 자세히 알아봅시다.

### 3.1 단계 1: Selection (선택)

**목적**: 트리에서 가장 유망한 노드를 찾아 내려간다

**방법**: UCB1 (Upper Confidence Bound 1) 공식 사용

```
루트에서 시작:
  └─ 자식 노드들 중 UCB1 값이 가장 큰 노드 선택
     └─ 다시 그 노드의 자식들 중 UCB1 최대 선택
        └─ 리프 노드에 도달할 때까지 반복
```

**UCB1 공식** (다음 섹션에서 상세 설명):
```
UCB1 = (승리 횟수 / 방문 횟수) + C × √(ln(부모 방문 횟수) / 방문 횟수)
       \_____________________/   \___________________________________/
              활용(Exploitation)              탐험(Exploration)
```

**시각화**:
```
        [루트: 10승/20경기]
         /       |        \
        /        |         \
    [A:5/10]  [B:3/5]   [C:2/5]

    UCB1 계산 (C=1.414):
    A: 5/10  + 1.414×√(ln(20)/10) = 0.50 + 0.53 = 1.03
    B: 3/5   + 1.414×√(ln(20)/5)  = 0.60 + 0.75 = 1.35 ← 최대!
    C: 2/5   + 1.414×√(ln(20)/5)  = 0.40 + 0.75 = 1.15

    → B를 선택!
```

**핵심**: UCB1은 승률 높은 수(활용)와 덜 탐색된 수(탐험)의 균형을 맞춥니다.

### 3.2 단계 2: Expansion (확장)

**목적**: 선택된 리프 노드에서 새로운 자식 노드를 추가

**방법**:
1. Selection으로 도달한 리프 노드 확인
2. 아직 시도하지 않은 수(untried moves) 중 하나 선택
3. 그 수에 해당하는 새 자식 노드 생성

**시각화**:
```
Selection으로 노드 B에 도달:
    [B: 3승/5경기]
    미탐색 수: [B1, B2, B3]

Expansion:
    [B: 3승/5경기]
     └─ [B1] ← 새로 생성! (B1 수를 선택)
    미탐색 수: [B2, B3]
```

**주의사항**:
- 터미널 노드(게임 종료)는 확장하지 않음
- 모든 자식이 이미 존재하면 확장하지 않고 Selection 계속

**확장 전략**:
- 일반적으로 **미탐색 수 중 무작위 선택**
- 또는 휴리스틱으로 우선순위 지정 가능 (고급)

### 3.3 단계 3: Simulation (시뮬레이션, Rollout)

**목적**: 확장된 노드에서 게임 끝까지 빠르게 플레이해서 결과 확인

**방법**: 게임이 끝날 때까지 **무작위로** 수를 선택

```python
def simulate(state):
    """게임 끝까지 랜덤 플레이"""
    current = state.copy()
    while not current.is_terminal():
        legal_moves = current.get_legal_moves()
        move = random.choice(legal_moves)  # 무작위 선택
        current.apply_move(move)
    return current.get_result()  # 승(1) / 패(0) / 무(0.5)
```

**시각화**:
```
    [B1: 새로 생성됨]
     │
     │ Simulation 시작 (랜덤 플레이)
     ├─→ B1의 상대 차례: 랜덤 수 선택
     ├─→ B1의 차례: 랜덤 수 선택
     ├─→ 상대 차례: 랜덤 수 선택
     ├─→ ...
     └─→ 게임 종료: 승리! (결과 = 1)
```

**왜 랜덤인가?**
- **속도**: 빠르게 많은 시뮬레이션 수행
- **단순성**: 복잡한 전략 불필요
- **통계적 유효성**: 많이 하면 평균적 경향 파악 가능

**개선 가능성**:
- 완전 랜덤 대신 간단한 휴리스틱 사용 (Heavy Rollout)
- 예: "중앙 선호", "상대 돌 많은 곳 피하기" 등

### 3.4 단계 4: Backpropagation (역전파)

**목적**: 시뮬레이션 결과를 루트까지 전파하여 통계 업데이트

**방법**:
1. 시뮬레이션 결과(승=1, 패=0, 무=0.5)를 가져옴
2. 확장된 노드부터 루트까지 경로상 모든 노드 업데이트
3. 각 노드의 방문 횟수(visits) +1
4. 각 노드의 승리 횟수(wins)에 결과 추가

**관점 전환 (중요!)**:
- 내 차례 노드: 내가 승리 → +1
- 상대 차례 노드: 내가 승리 → 상대 패배 → +0
- 즉, 부모-자식 간 결과를 뒤집어야 함: `result = 1 - result`

**시각화**:
```
초기 상태:
        [루트: 10승/20경기]
              |
        [B: 3승/5경기]
              |
        [B1: 0승/0경기] ← Simulation 결과: 승리(1)

Backpropagation:
1. B1 업데이트 (자신의 관점):
   [B1: 1승/1경기] (0→1, 0→1)

2. B 업데이트 (상대 관점, 결과 뒤집기 1→0):
   [B: 3승/6경기] (3→3, 5→6)

3. 루트 업데이트 (자신의 관점, 결과 다시 뒤집기 0→1):
   [루트: 11승/21경기] (10→11, 20→21)
```

**의사코드**:
```python
def backpropagate(node, result):
    """결과를 루트까지 역전파"""
    while node is not None:
        node.visits += 1
        node.wins += result
        result = 1 - result  # 관점 전환
        node = node.parent
```

### 3.5 4단계 전체 흐름 시각화

하나의 MCTS iteration 전체 과정:

```
트리 초기 상태:
                [루트: 10승/20경기]
                 /              \
            [A: 5승/10경기]   [B: 3승/5경기]
            /      \           미탐색: [B1,B2,B3]
        [A1]      [A2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1️⃣ SELECTION (UCB1으로 선택)
   루트 → B 선택 (UCB1 최대)

                [루트]
                   ↓ (UCB1=1.35)
                 [B] ← 리프 도달

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

2️⃣ EXPANSION (새 자식 추가)
   B1 수를 선택해서 노드 생성

                 [B]
                  |
                [B1] ← 새로 생성

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

3️⃣ SIMULATION (랜덤 플레이아웃)
   B1 상태에서 게임 끝까지 랜덤 진행

   [B1] → 랜덤 → 랜덤 → ... → 게임 종료
                                   ↓
                              결과: 승리(1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

4️⃣ BACKPROPAGATION (결과 역전파)
   승리(1) 결과를 루트까지 전파

                [루트: 11/21] ← +1승, +1경기
                   ↑ (result=1)
                 [B: 3/6]     ← +0승, +1경기
                   ↑ (result=0)
                [B1: 1/1]     ← +1승, +1경기
                     (result=1)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

트리 최종 상태:
                [루트: 11승/21경기]
                 /              \
            [A: 5승/10경기]   [B: 3승/6경기]
            /      \              |
        [A1]      [A2]          [B1: 1승/1경기]
```

이 과정을 **수천~수만 번** 반복합니다!

### 3.6 최종 수 선택

모든 iteration이 끝난 후, 어떤 수를 선택할까?

**일반적 방법**: 가장 많이 방문된 자식 선택

```
최종 트리 (10,000 iterations 후):
        [루트: 5200승/10000경기]
         /              |              \
    [A: 3800/7000]  [B: 900/2000]  [C: 500/1000]

선택: A (방문 횟수 7000으로 최다)
```

**왜 승률이 아니라 방문 횟수?**
- 방문 횟수가 많다 = 시뮬레이션을 많이 했다 = 더 신뢰할 수 있다
- 승률이 높아도 방문 횟수 적으면 우연일 수 있음
- 예: C는 승률 50%이지만 1000번만 방문 → 불확실

---

## 4. UCB1 공식 상세

UCB1은 MCTS의 핵심입니다. 자세히 알아봅시다.

### 4.1 UCB1 공식

```
UCB1(노드) = w_i/n_i + C × √(ln(N)/n_i)

여기서:
- w_i: 이 노드의 승리 횟수 (wins)
- n_i: 이 노드의 방문 횟수 (visits)
- N: 부모 노드의 방문 횟수
- C: 탐험 상수 (exploration constant), 보통 √2 ≈ 1.414
```

### 4.2 두 항의 의미

**첫 번째 항: w_i / n_i (활용, Exploitation)**
- 이 노드의 평균 승률
- 높을수록 좋은 노드
- "이미 알려진 좋은 수를 선택하자"

**두 번째 항: C × √(ln(N)/n_i) (탐험, Exploration)**
- 불확실성 보너스
- n_i가 작을수록 (덜 방문할수록) 커짐
- "아직 잘 모르는 수도 탐색해보자"

### 4.3 탐험 vs 활용 균형

**문제 상황**:
```
[A: 80승/100경기]  승률 80%, 하지만 이미 많이 탐색
[B: 5승/10경기]    승률 50%, 하지만 덜 탐색됨

어떤 것을 선택해야 할까?
```

**UCB1의 해답** (C=1.414, 부모 방문 200회 가정):

```
A의 UCB1:
  = 80/100 + 1.414 × √(ln(200)/100)
  = 0.80 + 1.414 × √(5.30/100)
  = 0.80 + 1.414 × 0.23
  = 0.80 + 0.33
  = 1.13

B의 UCB1:
  = 5/10 + 1.414 × √(ln(200)/10)
  = 0.50 + 1.414 × √(5.30/10)
  = 0.50 + 1.414 × 0.73
  = 0.50 + 1.03
  = 1.53 ← 더 큼!
```

→ B를 선택! (승률은 낮지만 불확실성 때문에)

### 4.4 C 값의 영향

탐험 상수 C는 탐험-활용 균형을 조절합니다:

**C가 클 때 (예: C=2.0)**
```
✓ 탐험 항이 커짐
✓ 덜 방문된 노드를 더 자주 선택
✓ 트리가 넓게 확장됨
✗ 좋은 수에 집중하지 못할 수 있음
```

**C가 작을 때 (예: C=0.5)**
```
✓ 활용 항이 지배적
✓ 승률 높은 노드에 집중
✓ 깊이 우선 탐색 경향
✗ 다른 좋은 수를 놓칠 수 있음
```

**C=0 (극단적 활용)**
```
항상 승률이 가장 높은 노드만 선택
→ Greedy 전략
→ 좋지 않은 전략 (지역 최적에 빠짐)
```

**C=∞ (극단적 탐험)**
```
항상 가장 덜 방문된 노드 선택
→ 균등 탐색
→ 시간 낭비
```

**적절한 C 값**:
- 이론적 최적값: √2 ≈ 1.414
- 실전에서는 게임에 따라 조정 (보통 0.7 ~ 2.0)

### 4.5 UCB1의 수학적 배경 (간단히)

UCB1은 **Multi-Armed Bandit** 문제에서 유래했습니다.

**Multi-Armed Bandit 비유**:
```
카지노에 슬롯머신 10대가 있음
각 머신은 서로 다른 확률로 돈을 줌
하지만 확률을 모름!

목표: 총 수익 최대화
전략: 어떤 머신을 당길까?
```

**딜레마**:
- 당겨본 머신: 수익률 알지만, 더 좋은 머신 놓칠 수 있음
- 안 당긴 머신: 더 좋을 수도 있지만 불확실

**UCB1의 보장**:
- 이론적으로 **regret**(후회, 최적 대비 손실)을 최소화
- 모든 팔을 충분히 당기면서도 좋은 팔에 집중

MCTS에서:
- 각 자식 노드 = 슬롯머신
- 승률 = 수익률
- UCB1로 선택 → 최적 수에 수렴

### 4.6 UCB1 변형들

**UCB1-Tuned**
- 분산도 고려
- 더 정확하지만 계산 복잡

**UCT (UCB for Trees)**
- MCTS에 UCB1을 적용한 것
- 일반적으로 UCT = MCTS와 혼용

**Progressive Bias**
- 도메인 지식을 UCB1에 추가
- 예: 체스에서 중앙 수에 보너스

---

## 5. MCTS 의사코드

이제 실제로 구현할 수 있는 코드를 봅시다.

### 5.1 MCTSNode 클래스

```python
import math
import random

class MCTSNode:
    """MCTS 트리의 노드"""

    def __init__(self, state, parent=None, move=None):
        """
        Args:
            state: 게임 상태
            parent: 부모 노드
            move: 이 노드로 오게 한 수
        """
        self.state = state              # 게임 상태
        self.parent = parent            # 부모 노드
        self.move = move                # 부모→자신으로 온 수
        self.children = []              # 자식 노드들
        self.wins = 0                   # 승리 횟수
        self.visits = 0                 # 방문 횟수
        self.untried_moves = state.get_legal_moves()  # 미탐색 수들

    def is_fully_expanded(self):
        """모든 자식이 확장되었는가?"""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """터미널 노드인가?"""
        return self.state.is_terminal()

    def ucb1(self, c=1.414):
        """
        UCB1 값 계산

        Args:
            c: 탐험 상수

        Returns:
            UCB1 값
        """
        if self.visits == 0:
            return float('inf')  # 미방문 노드는 최우선

        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration

    def select_child(self):
        """UCB1 값이 가장 큰 자식 선택"""
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        """
        미탐색 수 중 하나를 선택해서 자식 노드 추가

        Returns:
            새로 추가된 자식 노드
        """
        move = self.untried_moves.pop()  # 미탐색 수 하나 선택
        next_state = self.state.apply_move(move)  # 수 적용
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def rollout(self):
        """
        현재 상태에서 게임 끝까지 랜덤 플레이

        Returns:
            게임 결과 (1=승, 0=패, 0.5=무)
        """
        state = self.state.copy()
        while not state.is_terminal():
            legal_moves = state.get_legal_moves()
            move = random.choice(legal_moves)
            state.apply_move(move)
        return state.get_result()  # 현재 플레이어 관점 결과

    def backpropagate(self, result):
        """
        결과를 루트까지 역전파

        Args:
            result: 시뮬레이션 결과
        """
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1 - result  # 관점 전환 (부모는 상대편)
            node = node.parent

    def best_child(self, c=0):
        """
        최선의 자식 선택 (최종 수 결정용)

        Args:
            c: 0이면 순수 활용, 0 아니면 UCB1 사용

        Returns:
            최선의 자식 노드
        """
        if c == 0:
            # 방문 횟수가 가장 많은 자식
            return max(self.children, key=lambda child: child.visits)
        else:
            # UCB1 값이 가장 큰 자식
            return max(self.children, key=lambda child: child.ucb1(c))
```

### 5.2 MCTS 메인 함수

```python
def mcts(root_state, iterations=1000, c=1.414):
    """
    Monte Carlo Tree Search

    Args:
        root_state: 초기 게임 상태
        iterations: 시뮬레이션 반복 횟수
        c: UCB1 탐험 상수

    Returns:
        최선의 수 (move)
    """
    root = MCTSNode(root_state)

    for _ in range(iterations):
        node = root

        # 1. Selection: 리프 노드까지 내려가기
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child()

        # 2. Expansion: 가능하면 확장
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation: 게임 끝까지 랜덤 플레이
        result = node.rollout()

        # 4. Backpropagation: 결과 역전파
        node.backpropagate(result)

    # 최종 수 선택: 가장 많이 방문된 자식
    best = root.best_child(c=0)
    return best.move
```

### 5.3 사용 예시

```python
# 게임 상태 (ATAXX 예시)
current_state = AtaxxState()

# MCTS로 최선의 수 찾기
best_move = mcts(current_state, iterations=10000)

# 수 적용
current_state.apply_move(best_move)
print(f"MCTS가 선택한 수: {best_move}")
```

### 5.4 시간 기반 MCTS

실전에서는 횟수 대신 시간 제한을 사용합니다:

```python
import time

def mcts_with_time_limit(root_state, time_limit_ms=1000, c=1.414):
    """
    시간 제한이 있는 MCTS

    Args:
        root_state: 초기 게임 상태
        time_limit_ms: 시간 제한 (밀리초)
        c: UCB1 탐험 상수

    Returns:
        최선의 수 (move)
    """
    root = MCTSNode(root_state)
    start_time = time.time()
    end_time = start_time + time_limit_ms / 1000.0

    iterations = 0
    while time.time() < end_time:
        node = root

        # 1. Selection
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child()

        # 2. Expansion
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation
        result = node.rollout()

        # 4. Backpropagation
        node.backpropagate(result)

        iterations += 1

    print(f"MCTS: {iterations} iterations in {time_limit_ms}ms")

    # 최종 수 선택
    best = root.best_child(c=0)
    return best.move
```

### 5.5 게임 상태 인터페이스

MCTS가 작동하려면 게임 상태 클래스가 다음 메서드를 제공해야 합니다:

```python
class GameState:
    """MCTS를 위한 게임 상태 인터페이스"""

    def get_legal_moves(self):
        """
        Returns:
            가능한 수들의 리스트
        """
        pass

    def apply_move(self, move):
        """
        수를 적용한 새 상태 반환

        Args:
            move: 적용할 수

        Returns:
            새 게임 상태 (원본은 변경 안 함)
        """
        pass

    def is_terminal(self):
        """
        Returns:
            게임이 종료되었는가?
        """
        pass

    def get_result(self):
        """
        현재 플레이어 관점의 게임 결과

        Returns:
            1.0 (승) / 0.0 (패) / 0.5 (무승부)
        """
        pass

    def copy(self):
        """
        Returns:
            상태의 복사본
        """
        pass
```

---

## 6. MCTS vs Alpha-Beta 비교

두 알고리즘을 다양한 측면에서 비교해봅시다.

### 6.1 핵심 차이

| 측면 | Alpha-Beta Pruning | MCTS |
|------|-------------------|------|
| **접근 방식** | 결정적 탐색 | 확률적 탐색 |
| **평가 방법** | 평가 함수 | 시뮬레이션 (실제 게임) |
| **탐색 전략** | 깊이 우선, 전체 폭 | 선택적, 유망한 가지 집중 |
| **시간 제어** | 최대 깊이로 제한 | 시뮬레이션 횟수/시간으로 제한 |
| **평가 함수** | 필수 | 불필요 |
| **구현 난이도** | 중간 (평가 함수 설계 어려움) | 중간 (구조는 단순) |

### 6.2 장단점 비교

**Alpha-Beta의 장점**:
```
✓ 평가 함수가 좋으면 매우 정확
✓ 완전 정보 게임에서 이론적으로 최적
✓ 깊이 제한으로 예측 가능한 계산 시간
✓ 구현이 직관적
✓ 메모리 효율적 (트리 저장 불필요)
```

**Alpha-Beta의 단점**:
```
✗ 좋은 평가 함수 설계가 어려움
✗ 평가 함수에 따라 성능 크게 좌우
✗ 수평선 효과 (horizon effect)
✗ 분기 계수 높으면 얕은 깊이만 탐색 가능
```

**MCTS의 장점**:
```
✓ 평가 함수 불필요
✓ 도메인 지식 없이도 작동
✓ 중요한 변화에 자동으로 집중
✓ Anytime algorithm (언제든 중단 가능)
✓ 시간이 많을수록 성능 향상
✓ 병렬화 용이
```

**MCTS의 단점**:
```
✗ 수렴이 느릴 수 있음 (특히 초반)
✗ 메모리 사용량 많음 (트리 저장)
✗ 랜덤 시뮬레이션은 비효율적일 수 있음
✗ 전술적 수 (tactical move) 놓치기 쉬움
```

### 6.3 게임 유형별 적합성

**Alpha-Beta가 유리한 게임**:
```
체스 (Chess)
  - 평가 함수 설계 잘 되어 있음
  - 깊이 탐색이 중요
  - 전술적 계산 필요

오셀로 (Othello/Reversi)
  - 평가 함수가 효과적
  - 분기 계수 적당

체커 (Checkers)
  - 완전히 해결된 게임
  - Alpha-Beta로 완벽한 플레이 가능
```

**MCTS가 유리한 게임**:
```
바둑 (Go)
  - 평가 함수 설계 매우 어려움
  - 분기 계수 매우 큼 (~250)
  - 국면 평가가 복잡

Hex
  - 평가 함수 어려움
  - MCTS의 성공 사례

복잡한 전략 게임
  - 다양한 요소가 얽혀 있음
  - 평가 함수 설계 어려움
```

**ATAXX는?**
```
중간 정도
  - 평가 함수 설계 가능 (돌 개수, 이동성 등)
  - 하지만 완벽하지는 않음
  - 분기 계수: 중간 정도

→ 둘 다 사용 가능!
→ 학습하기 좋은 예제
```

### 6.4 성능 비교 (ATAXX 예상)

**초반** (게임 시작):
```
Alpha-Beta (깊이 6):
  - 빠른 판단
  - 평가 함수에 의존
  - 안정적인 플레이

MCTS (1000 iterations):
  - 가능성 넓게 탐색
  - 예상 밖의 수도 시도
  - 초반에는 약간 불안정
```

**중반**:
```
Alpha-Beta:
  - 평가 함수가 정확하면 강력
  - 전술적 수 잘 찾음

MCTS:
  - 복잡한 국면에서 강점
  - 장기적 전략 더 나을 수 있음
```

**종반**:
```
Alpha-Beta:
  - 완전 탐색 가능하면 최강
  - 정확한 승부 계산

MCTS:
  - 시뮬레이션이 빠르게 끝남
  - 많은 경우의 수 탐색 가능
```

### 6.5 실전 조합

실제로는 두 방법을 **결합**하는 경우도 많습니다:

**1. MCTS + 평가 함수**
```python
def rollout_with_heuristic(state):
    """휴리스틱을 사용한 롤아웃"""
    while not state.is_terminal():
        moves = state.get_legal_moves()
        # 완전 랜덤 대신 간단한 평가로 선택
        move = max(moves, key=lambda m: simple_evaluate(state, m))
        state.apply_move(move)
    return state.get_result()
```

**2. Alpha-Beta + MCTS**
```
- 초반: MCTS로 광범위 탐색
- 중반: Alpha-Beta로 정확한 계산
- 종반: Alpha-Beta로 완전 탐색
```

**3. MCTS + 도메인 지식**
```python
def ucb1_with_prior(node):
    """사전 확률을 포함한 UCB1"""
    prior = get_domain_knowledge(node.state, node.move)
    return node.wins/node.visits + c*sqrt(...) + prior
```

---

## 7. MCTS 개선 기법

기본 MCTS는 시작점일 뿐, 다양한 개선 기법이 있습니다.

### 7.1 Heavy Rollout

**기본 MCTS의 문제**:
```
완전 랜덤 시뮬레이션은 비효율적
→ 명백히 나쁜 수도 선택
→ 수렴이 느림
```

**해결책: 간단한 휴리스틱 사용**

```python
def heavy_rollout(state):
    """휴리스틱 기반 롤아웃"""
    while not state.is_terminal():
        moves = state.get_legal_moves()

        # 완전 랜덤 대신 간단한 규칙 적용
        scored_moves = []
        for move in moves:
            score = 0
            # 규칙 1: 중앙 선호
            if is_center(move):
                score += 2
            # 규칙 2: 상대 많은 곳 선호 (감염)
            score += count_adjacent_opponents(state, move)
            # 규칙 3: Jump보다 Split 선호 (확장)
            if is_split(move):
                score += 1

            scored_moves.append((move, score))

        # 확률적 선택 (높은 점수일수록 선택 확률 높음)
        move = weighted_random_choice(scored_moves)
        state.apply_move(move)

    return state.get_result()
```

**장점**:
- 더 현실적인 시뮬레이션
- 빠른 수렴

**단점**:
- 휴리스틱 설계 필요 (도메인 지식)
- 잘못된 휴리스틱은 역효과

### 7.2 RAVE (Rapid Action Value Estimation)

**아이디어**: 같은 수는 어느 시점에 두어도 비슷한 가치

```
예시:
  경로 A: X→Y→Z → 승리
  경로 B: Y→X→Z → ?

RAVE: Y와 X의 순서가 바뀌어도 Z는 좋은 수일 가능성 높음
→ Z의 가치를 더 빨리 학습
```

**구현 개념**:
```python
class RAVENode(MCTSNode):
    def __init__(self, ...):
        super().__init__(...)
        self.rave_wins = 0    # RAVE 승리 횟수
        self.rave_visits = 0  # RAVE 방문 횟수

    def backpropagate_rave(self, result, moves_played):
        """RAVE 역전파: 경로에 있는 모든 수 업데이트"""
        for move in moves_played:
            for child in self.children:
                if child.move == move:
                    child.rave_visits += 1
                    child.rave_wins += result

    def ucb1_rave(self, beta):
        """RAVE를 결합한 UCB1"""
        ucb1_value = self.wins / self.visits + c * sqrt(...)
        rave_value = self.rave_wins / self.rave_visits if self.rave_visits > 0 else 0
        return (1 - beta) * ucb1_value + beta * rave_value
```

**효과**: 초기 수렴 속도 크게 향상

### 7.3 시뮬레이션 횟수와 성능 관계

MCTS는 시뮬레이션을 많이 할수록 강해집니다:

```
Iterations  |  승률 (vs 랜덤)  |  평균 수 계산 시간
------------|------------------|------------------
100         |  60%             |  10ms
500         |  75%             |  50ms
1,000       |  82%             |  100ms
5,000       |  91%             |  500ms
10,000      |  94%             |  1000ms
50,000      |  97%             |  5000ms
```

**수확 체감 법칙**:
- 처음엔 빠르게 개선
- 나중엔 개선 속도 감소

**실전 권장**:
- 빠른 대전: 1000~5000 iterations
- 느긴 대전: 10000~50000 iterations
- 시간 제한: 100~500ms per move

### 7.4 병렬 MCTS

MCTS는 병렬화가 용이합니다:

**방법 1: Root Parallelization**
```
여러 트리를 동시에 성장시킴
→ 마지막에 통계 합산
```

**방법 2: Leaf Parallelization**
```
Selection/Expansion은 순차
→ Simulation만 병렬로 여러 개
```

**방법 3: Tree Parallelization**
```
하나의 트리를 여러 쓰레드가 공유
→ 락(lock)으로 동기화
```

### 7.5 Early Termination

**시뮬레이션 조기 종료**:
```python
def rollout_with_early_termination(state, max_depth=50):
    """일정 깊이 후 평가 함수 사용"""
    depth = 0
    while not state.is_terminal() and depth < max_depth:
        move = random.choice(state.get_legal_moves())
        state.apply_move(move)
        depth += 1

    if state.is_terminal():
        return state.get_result()
    else:
        # 평가 함수로 추정
        return evaluate(state)
```

**효과**: 긴 게임에서 시간 절약

### 7.6 Progressive Widening

**문제**: 분기 계수가 매우 크면 모든 자식 탐색이 비효율적

**해결**: 점진적으로 자식 추가
```python
def should_add_child(node):
    """새 자식을 추가할지 결정"""
    # 방문 횟수에 따라 자식 수 제한
    max_children = int(node.visits ** 0.5)
    return len(node.children) < max_children
```

---

## 8. 핵심 정리 및 다음 주 예고

### 8.1 MCTS 핵심 요약

**MCTS란?**
- 시뮬레이션 기반 게임 트리 탐색
- 평가 함수 없이도 작동
- UCB1로 탐험-활용 균형

**4단계**:
1. **Selection**: UCB1으로 유망한 노드 선택
2. **Expansion**: 새 자식 추가
3. **Simulation**: 게임 끝까지 랜덤 플레이
4. **Backpropagation**: 결과 역전파

**장점**:
- 도메인 지식 불필요
- 복잡한 게임에 효과적
- 언제든 중단 가능

**단점**:
- 수렴이 느릴 수 있음
- 메모리 사용량 많음

### 8.2 구현 체크리스트

MCTS를 구현할 때 확인사항:

```
□ MCTSNode 클래스
  □ wins, visits 통계
  □ children, parent 연결
  □ untried_moves 관리

□ UCB1 계산
  □ exploitation + exploration
  □ 미방문 노드는 무한대
  □ C 값 설정 (보통 1.414)

□ Selection
  □ UCB1 최대 자식 선택
  □ 리프까지 반복

□ Expansion
  □ 미탐색 수 선택
  □ 새 자식 노드 생성

□ Simulation
  □ 랜덤 또는 휴리스틱
  □ 터미널까지 진행
  □ 결과 반환

□ Backpropagation
  □ 루트까지 역전파
  □ 관점 전환 (1-result)

□ 최종 수 선택
  □ 방문 횟수 최대 자식
```

### 8.3 실습 과제

**과제 1: 틱택토 MCTS**
- 간단한 게임으로 MCTS 구현
- 완벽한 플레이 학습 확인

**과제 2: ATAXX MCTS**
- ALPHANO 프로토콜 준수
- 시간 제한 고려
- 성능 측정

**과제 3: 파라미터 실험**
- C 값 변화에 따른 성능
- Iterations 수에 따른 승률
- Heavy rollout 효과

### 8.4 MCTS의 역사와 영향

**발전 과정**:
```
2006: MCTS 논문 발표 (Coulom, Kocsis & Szepesvári)
2007: MoGo (바둑 프로그램)가 프로 기사를 격파 (9×9)
2015: AlphaGo (MCTS + Deep Learning)
2016: AlphaGo vs 이세돌 9단 (4:1)
2017: AlphaGo Zero (완전 자가 학습)
```

**게임 AI의 패러다임 전환**:
- 평가 함수 설계 → 시뮬레이션 기반
- 도메인 지식 → 데이터 기반 학습
- 결정적 탐색 → 확률적 탐색

### 8.5 다음 주 예고: 강화학습 (Reinforcement Learning)

MCTS는 강화학습의 일종입니다. 다음 주에는:

**강화학습 기초**:
- Agent, Environment, Reward
- Policy, Value Function
- Exploration vs Exploitation

**Q-Learning**:
- Q-Table 학습
- Bellman Equation
- Epsilon-Greedy

**ATAXX + Q-Learning**:
- 상태 표현
- 보상 설계
- 학습 과정

**MCTS vs RL 비교**:
- MCTS: 시뮬레이션으로 즉시 학습
- RL: 경험을 통해 점진적 학습

강화학습은 게임 AI를 넘어 로봇, 자율주행, 추천 시스템 등 다양한 분야에 적용됩니다!

---

## 부록: 참고 자료

### A. 추천 논문

1. **Kocsis & Szepesvári (2006)**: "Bandit based Monte-Carlo Planning"
   - UCT (UCB for Trees) 제안
   - MCTS의 이론적 기초

2. **Coulom (2006)**: "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"
   - MCTS의 체계화
   - 바둑 적용

3. **Silver et al. (2016)**: "Mastering the game of Go with deep neural networks and tree search"
   - AlphaGo 논문
   - MCTS + Deep Learning

### B. 추천 자료

**온라인 강의**:
- Stanford CS234: Reinforcement Learning
- DeepMind x UCL RL Lecture Series

**책**:
- "Artificial Intelligence: A Modern Approach" (Russell & Norvig)
  - Chapter 5: Adversarial Search

**웹사이트**:
- https://www.moderndescartes.com/essays/deep_dive_mcts/
- https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/

### C. MCTS 구현 예제

다양한 언어로 구현된 MCTS:
- Python: https://github.com/pbsinclair42/MCTS
- C++: https://github.com/memo/ofxMSAmcts
- Java: https://github.com/eugenp/tutorials/tree/master/algorithms-modules/algorithms-miscellaneous-2

### D. ALPHANO 대회 팁

**시간 관리**:
- my_time > 1000ms: 여유 있게 탐색 (150ms)
- my_time < 1000ms: 빠르게 탐색 (10ms)

**메모리**:
- 트리가 너무 커지지 않도록 주의
- 오래된 노드 정리 고려

**디버깅**:
- 시뮬레이션 횟수 출력
- UCB1 값 확인
- 승률 모니터링

**최적화**:
- 빠른 보드 복사
- 효율적인 합법 수 생성
- Heavy rollout 고려

---

**Week 4 수업 준비 완료!**

학생들이 MCTS의 핵심 아이디어를 이해하고, 실제로 구현할 수 있도록 준비되었습니다.

다음 단계:
1. 강의 자료 리뷰
2. 실습 코드 준비
3. ALPHANO 제출 코드 테스트
4. 학생 질문 예상 및 답변 준비

행운을 빕니다!
