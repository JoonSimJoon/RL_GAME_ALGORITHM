# Week 4 수업 대본: Monte Carlo Tree Search (MCTS)

**수업 시간**: 90분
**대상**: 고등학생 게임 AI 수업
**주제**: 시뮬레이션 기반 게임 트리 탐색

---

## 수업 개요

| 시간 | 내용 | 방식 |
|------|------|------|
| 0-5분 | 도입 및 동기부여 | 강의 |
| 5-25분 | 이론 1: MCTS 4단계 + UCB1 | 강의 + 시연 |
| 25-45분 | 실습 1: 틱택토 MCTS 구현 | 실습 |
| 45-55분 | 이론 2: MCTS vs Alpha-Beta | 강의 + 토론 |
| 55-75분 | 실습 2: ATAXX MCTS + 실험 | 실습 |
| 75-85분 | 정리 및 과제 안내 | 강의 |
| 85-90분 | 질의응답 | 토론 |

---

## 도입 (0-5분)

### 인사 및 출석 (1분)

**교사**: 안녕하세요, 여러분! 오늘은 Week 4, Monte Carlo Tree Search를 배울 거예요. 지난주 과제는 잘 제출했나요?

🎯 **학생 반응 확인**

### 동기부여: Alpha-Beta의 한계 (4분)

**교사**: 지난 3주 동안 우리는 Minimax, Alpha-Beta, PVS를 배웠습니다. 이들은 매우 강력한 알고리즘이죠. 하지만 한 가지 질문을 해볼게요.

**[화면에 표시]**
```
질문: Alpha-Beta가 완벽한 게임 AI 알고리즘일까?
```

🎯 **학생들에게 질문**: "Alpha-Beta의 문제점이 뭐가 있을까요?"

**예상 답변**:
- "깊이 제한이 있어요"
- "평가 함수가 필요해요"
- "시간이 오래 걸려요"

**교사**: 좋은 답변들이네요! 특히 **평가 함수**가 핵심입니다.

**[슬라이드: 평가 함수의 어려움]**

```python
# ATAXX 평가 함수 - 이게 최선일까?
def evaluate(board):
    my_count = count_my_pieces(board)
    opp_count = count_opponent_pieces(board)
    return my_count - opp_count

# 이것만으로 충분할까?
# - 중앙 위치의 가치는?
# - 이동 가능한 수의 개수는?
# - 고립된 돌은 얼마나 나쁠까?
#
# 가중치를 어떻게 정해야 할까? 🤔
```

**교사**: 더 큰 문제는 **바둑** 같은 게임입니다.

**[슬라이드: 바둑 사진]**

19×19 바둑판에서 "이 국면이 좋은가?"를 숫자로 나타내기 매우 어렵습니다. 전문가도 직관에 의존하죠.

**교사**: 그렇다면 이런 생각을 해볼 수 있습니다:

**[화면에 크게 표시]**
```
💡 "평가 함수로 추정하지 말고,
    실제로 게임을 끝까지 해보면 어떨까?"
```

**교사**: 이것이 바로 오늘 배울 **Monte Carlo Tree Search**의 핵심 아이디어입니다!

---

## 이론 1: MCTS 4단계 + UCB1 (5-25분)

### Monte Carlo 방법 소개 (3분)

**교사**: 먼저 "Monte Carlo"가 무슨 뜻인지 알아봅시다.

**[슬라이드: 카지노 사진]**

Monte Carlo는 모나코의 유명한 카지노입니다. 이 방법은 **무작위(random)**를 사용해서 답을 구하는 기법이에요.

**일상 예시**:

```
문제: 이 가방에 빨간 구슬이 몇 %일까?

방법 1 (정확한 방법):
  → 모든 구슬을 꺼내서 센다
  → 시간이 오래 걸림

방법 2 (Monte Carlo):
  → 무작위로 100개만 뽑는다
  → 빨간 구슬 37개 발견
  → 추정: 약 37%
  → 빠르고 충분히 정확!
```

**교사**: 게임 AI에도 같은 아이디어를 적용할 수 있습니다.

```
질문: "수 A가 좋은 수일까?"

전통적 방법 (Alpha-Beta):
  → 평가 함수로 점수 계산
  → 예: +3.7점

Monte Carlo 방법:
  → 수 A를 두고 게임을 1000번 끝까지 해봄
  → 620번 승리
  → 승률: 62%
  → "좋은 수다!"
```

### MCTS 기본 개념 (4분)

**교사**: MCTS는 **시뮬레이션 기반 트리 탐색**입니다.

**[화면에 표시]**
```
MCTS = Monte Carlo + Tree Search

Monte Carlo: 무작위 시뮬레이션
Tree Search: 게임 트리 탐색
```

**핵심 특징**:

1. **평가 함수 불필요** ✓
   - 실제 게임 결과(승/패)를 사용

2. **점진적 확장** ✓
   - 중요한 부분만 집중 탐색
   - 시간을 효율적으로 사용

3. **언제든 중단 가능** ✓
   - 1초 주면 1초만큼 탐색
   - 10초 주면 10초만큼 탐색

**교사**: 간단한 예시를 봅시다.

**[화면: 애니메이션]**
```
초기:
    [루트]
     /  \
   수A  수B

1회: 수A 시도 → 랜덤 플레이 → 승리!
    [루트: 1승/1경기]
      /
   [A: 1승/1경기]

2회: 수B 시도 → 랜덤 플레이 → 패배
    [루트: 1승/2경기]
      /              \
   [A: 1승/1경기]   [B: 0승/1경기]

3회: A가 더 좋아 보이니 A를 더 탐색
    [루트: 2승/3경기]
      /              \
   [A: 2승/2경기]   [B: 0승/1경기]

... 수천 번 반복!
```

### MCTS 4단계 상세 설명 (10분)

**교사**: MCTS는 4개의 단계를 반복합니다. 각 단계를 자세히 봅시다.

**[슬라이드: 4단계 개요]**
```
┌─────────────────────┐
│ MCTS 1회 Iteration  │
│                     │
│ 1. Selection        │
│ 2. Expansion        │
│ 3. Simulation       │
│ 4. Backpropagation  │
└─────────────────────┘
```

#### 단계 1: Selection (선택)

**교사**: 첫 번째 단계는 Selection입니다.

**[화면에 표시]**
```
목적: 트리에서 가장 유망한 노드를 찾아 내려간다
방법: UCB1 공식 사용
```

**교사**: UCB1은 뭘까요? 잠시 후에 자세히 볼 텐데, 먼저 개념만 이해해봅시다.

```
        [루트: 10승/20경기]
         /       |        \
    [A:5/10]  [B:3/5]   [C:2/5]

어떤 노드를 선택해야 할까?
```

🎯 **학생들에게 질문**: "어떤 노드를 선택하는 게 좋을까요?"

**예상 답변**:
- "A요! 승률이 50%로 가장 높아요"
- "B요! 승률 60%예요"
- "C는 아직 덜 탐색했으니 한 번 더 봐야 해요"

**교사**: 모두 일리 있는 답변입니다! 바로 이것이 **탐험 vs 활용** 딜레마입니다.

**[슬라이드]**
```
탐험 (Exploration):
  → 아직 잘 모르는 수 탐색
  → "C를 더 봐야 해!"

활용 (Exploitation):
  → 이미 좋다고 알려진 수 선택
  → "B가 승률 60%니까 B!"

둘 다 중요! 균형이 필요!
```

**교사**: UCB1은 이 균형을 자동으로 맞춰줍니다.

#### 단계 2: Expansion (확장)

**교사**: Selection으로 리프 노드에 도달하면, 새 자식을 추가합니다.

```
[B: 3승/5경기] ← Selection으로 도달
미탐색 수: [B1, B2, B3]

↓ Expansion

[B: 3승/5경기]
 └─ [B1] ← 새로 생성!
미탐색 수: [B2, B3]
```

**교사**: 간단하죠? 미탐색 수 중 하나를 골라서 노드를 만듭니다.

#### 단계 3: Simulation (시뮬레이션)

**교사**: 이제 가장 중요한 부분입니다! 실제로 게임을 끝까지 해봅니다.

**[코드 시연]**
```python
def simulate(state):
    """게임 끝까지 랜덤 플레이"""
    current = state.copy()
    while not current.is_terminal():
        legal_moves = current.get_legal_moves()
        move = random.choice(legal_moves)  # 무작위!
        current.apply_move(move)
    return current.get_result()  # 승(1) / 패(0)
```

**교사**: **완전히 무작위**로 게임을 끝까지 진행합니다.

```
[B1: 새로 생성]
 │
 ├→ 상대 차례: 랜덤 수
 ├→ 내 차례: 랜덤 수
 ├→ 상대 차례: 랜덤 수
 ├→ ...
 └→ 게임 종료: 승리! (결과=1)
```

🎯 **학생 질문 예상**: "왜 랜덤이에요? 좋은 수를 선택하면 더 좋지 않나요?"

**교사**: 좋은 질문이에요! 랜덤을 사용하는 이유:

1. **속도**: 빠르게 많은 시뮬레이션 가능
2. **단순성**: 복잡한 로직 불필요
3. **통계적 유효성**: 많이 하면 평균 경향 파악 가능

나중에 개선할 수도 있습니다 (Heavy Rollout).

#### 단계 4: Backpropagation (역전파)

**교사**: 시뮬레이션 결과를 루트까지 전파합니다.

```
[B1: 0승/0경기] ← 시뮬레이션 결과: 승리(1)

Backpropagation:

1. [B1: 1승/1경기] ← 자신 업데이트
          ↑ (result=1)

2. [B: 3승/6경기] ← 부모 업데이트 (상대 관점!)
          ↑ (result=0)

3. [루트: 11승/21경기] ← 루트 업데이트
```

**교사**: 중요한 점! **관점 전환**을 해야 합니다.

**[화면에 강조]**
```python
def backpropagate(node, result):
    while node is not None:
        node.visits += 1
        node.wins += result
        result = 1 - result  # 관점 전환!
        node = node.parent
```

**교사**: 내가 이겼다는 것은 부모(상대 차례)에서는 졌다는 의미입니다.

### UCB1 공식 상세 (3분)

**교사**: 이제 UCB1 공식을 자세히 봅시다.

**[슬라이드: UCB1 공식]**
```
UCB1 = w_i/n_i + C × √(ln(N)/n_i)
       \_______/   \_______________/
        활용          탐험

여기서:
- w_i: 승리 횟수
- n_i: 방문 횟수
- N: 부모 방문 횟수
- C: 탐험 상수 (보통 √2 ≈ 1.414)
```

**교사**: 실제 계산을 해봅시다.

**[화면: 계산 예시]**
```
    [루트: 10승/20경기]
     /       |        \
  [A:5/10] [B:3/5]  [C:2/5]

UCB1 계산 (C=1.414, 부모=20):

A의 UCB1:
  = 5/10 + 1.414 × √(ln(20)/10)
  = 0.50 + 1.414 × √(2.996/10)
  = 0.50 + 1.414 × 0.547
  = 0.50 + 0.53
  = 1.03

B의 UCB1:
  = 3/5 + 1.414 × √(ln(20)/5)
  = 0.60 + 1.414 × √(2.996/5)
  = 0.60 + 1.414 × 0.774
  = 0.60 + 0.75
  = 1.35 ← 최대!

C의 UCB1:
  = 2/5 + 1.414 × √(ln(20)/5)
  = 0.40 + 0.75
  = 1.15

→ B를 선택!
```

**교사**: B의 승률(60%)이 A(50%)보다 높고, 탐험 보너스도 크기 때문에 선택됩니다.

🎯 **학생들에게 질문**: "만약 B를 10번 더 방문하면 어떻게 될까요?"

**교사**: 좋은 질문이에요! B의 n_i가 커지면 탐험 항이 작아집니다. 그러면 다른 노드(A나 C)가 선택될 기회가 생기죠. 이렇게 균형을 맞춥니다!

---

## 실습 1: 틱택토 MCTS 구현 (25-45분)

### 실습 소개 (2분)

**교사**: 이제 직접 MCTS를 구현해봅시다! 먼저 간단한 게임인 틱택토로 시작합니다.

**[화면: 틱택토 보드]**
```
X | O | X
---------
O | X |
---------
  | O |
```

**교사**: 틱택토는 간단하지만 MCTS를 배우기 완벽한 게임입니다.

### 코드 설명 (8분)

**교사**: 먼저 MCTSNode 클래스부터 봅시다.

**[화면: 코드]**
```python
import math
import random

class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = state.get_legal_moves()
```

**교사**: 각 변수의 의미:
- `state`: 게임 상태
- `parent`: 부모 노드 (역전파를 위해 필요)
- `move`: 이 노드로 오게 한 수
- `children`: 자식 노드들
- `wins`, `visits`: 통계
- `untried_moves`: 아직 탐색 안 한 수들

**[UCB1 메서드]**
```python
def ucb1(self, c=1.414):
    if self.visits == 0:
        return float('inf')  # 미방문은 최우선

    exploitation = self.wins / self.visits
    exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
    return exploitation + exploration
```

🎯 **학생 질문 예상**: "왜 visits가 0이면 무한대를 반환하나요?"

**교사**: 아직 한 번도 방문하지 않은 노드는 무조건 먼저 탐색해야 합니다. UCB1 공식은 visits > 0일 때만 의미가 있어요.

**[Selection 메서드]**
```python
def select_child(self):
    """UCB1 최대인 자식 선택"""
    return max(self.children, key=lambda c: c.ucb1())
```

**[Expansion 메서드]**
```python
def expand(self):
    """새 자식 추가"""
    move = self.untried_moves.pop()
    next_state = self.state.apply_move(move)
    child = MCTSNode(next_state, parent=self, move=move)
    self.children.append(child)
    return child
```

**[Rollout 메서드]**
```python
def rollout(self):
    """게임 끝까지 랜덤 플레이"""
    state = self.state.copy()
    while not state.is_terminal():
        move = random.choice(state.get_legal_moves())
        state.apply_move(move)
    return state.get_result()
```

**[Backpropagation 메서드]**
```python
def backpropagate(self, result):
    """결과를 루트까지 역전파"""
    node = self
    while node is not None:
        node.visits += 1
        node.wins += result
        result = 1 - result  # 관점 전환
        node = node.parent
```

**[MCTS 메인 함수]**
```python
def mcts(root_state, iterations=1000):
    root = MCTSNode(root_state)

    for _ in range(iterations):
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

    # 최종 수 선택
    best = max(root.children, key=lambda c: c.visits)
    return best.move
```

### 실습 진행 (10분)

**교사**: 이제 여러분이 직접 구현해봅시다. 제공된 틱택토 코드를 완성하세요.

**[실습 파일 제공]**
- `tictactoe.py`: 틱택토 게임 상태 (완성됨)
- `mcts_node.py`: MCTSNode 클래스 (빈칸 채우기)
- `test_mcts.py`: 테스트 코드

**실습 과제**:
```
1. MCTSNode 클래스 완성
   - ucb1() 메서드 구현
   - expand() 메서드 구현
   - backpropagate() 메서드 구현

2. mcts() 함수 완성
   - 4단계 구현

3. 테스트
   - MCTS vs 랜덤 대결
   - 승률 확인
```

**교사**: 10분 드릴게요. 질문 있으면 손들어주세요!

**[학생들 실습 진행]**

### 실습 결과 공유 (5분)

**교사**: 시간 됐습니다! 결과를 봅시다.

🎯 **학생들에게 질문**: "MCTS가 랜덤 플레이어를 이겼나요?"

**예상 결과**:
```
MCTS (1000 iterations) vs Random
승: 95개
패: 3개
무: 2개
승률: 95%
```

**교사**: 거의 완벽한 플레이죠! 틱택토는 간단해서 MCTS가 최적 전략을 학습할 수 있습니다.

🎯 **실험 제안**: "iterations를 100, 500, 1000, 5000으로 바꾸면 어떻게 될까요?"

**예상 결과**:
```
100 iterations:   75% 승률
500 iterations:   88% 승률
1000 iterations:  95% 승률
5000 iterations:  98% 승률
```

**교사**: 시뮬레이션을 많이 할수록 더 강해집니다!

---

## 이론 2: MCTS vs Alpha-Beta (45-55분)

### 비교 표 (5분)

**교사**: 이제 MCTS와 Alpha-Beta를 비교해봅시다.

**[슬라이드: 비교 표]**
| 항목 | Alpha-Beta | MCTS |
|------|-----------|------|
| **접근 방식** | 결정적 탐색 | 확률적 탐색 |
| **평가 방법** | 평가 함수 | 시뮬레이션 |
| **평가 함수** | 필수 | 불필요 |
| **시간 제어** | 깊이 제한 | 시뮬레이션 횟수 |
| **장점** | 정확, 빠름 | 도메인 지식 불필요 |
| **단점** | 평가 함수 필요 | 수렴 느림 |

### 장단점 토론 (5분)

**교사**: 각각의 장단점을 더 자세히 봅시다.

**Alpha-Beta의 장점**:
```
✓ 평가 함수가 좋으면 매우 정확
✓ 깊이 제한으로 시간 예측 가능
✓ 메모리 효율적
✓ 완전 정보 게임에서 이론적 최적
```

**Alpha-Beta의 단점**:
```
✗ 좋은 평가 함수 설계 어려움
✗ 수평선 효과
✗ 분기 계수 높으면 얕은 탐색만 가능
```

**MCTS의 장점**:
```
✓ 평가 함수 불필요
✓ 복잡한 게임에 효과적
✓ Anytime algorithm
✓ 중요한 변화에 자동 집중
```

**MCTS의 단점**:
```
✗ 수렴 느림
✗ 메모리 사용 많음
✗ 전술적 수 놓치기 쉬움
```

🎯 **학생들에게 질문**: "어떤 게임에는 Alpha-Beta가 좋고, 어떤 게임에는 MCTS가 좋을까요?"

**교사**: 좋은 질문이에요! 게임 유형에 따라 다릅니다.

**[슬라이드: 게임 유형별 적합성]**

**Alpha-Beta가 유리**:
- 체스: 평가 함수 잘 되어 있음
- 오셀로: 평가 함수 효과적
- 체커: 완전히 해결됨

**MCTS가 유리**:
- 바둑: 평가 함수 매우 어려움, 분기 계수 큼
- Hex: 평가 함수 어려움
- 복잡한 전략 게임

**교사**: ATAXX는 중간 정도입니다. 둘 다 사용 가능해요!

---

## 실습 2: ATAXX MCTS + 실험 (55-75분)

### ALPHANO 프로토콜 복습 (3분)

**교사**: 이제 ATAXX에 MCTS를 적용해봅시다. 먼저 ALPHANO 프로토콜을 복습합니다.

**[화면: 프로토콜]**
```
입력:
  READY FIRST/SECOND → "OK" 출력
  TURN my_time opp_time → "MOVE x1 y1 x2 y2" 출력 (1-indexed)
  OPP x1 y1 x2 y2 → 보드 업데이트
  FINISH → 종료

ATAXX 규칙:
  - 7×7 보드
  - Split (거리 1): 복제
  - Jump (거리 2): 이동
  - 인접 8방향 감염
```

### 코드 설명 (7분)

**교사**: ATAXX MCTS 에이전트를 봅시다.

**[화면: 코드 구조]**
```python
class AtaxxBoard:
    """ATAXX 보드 상태"""

    def get_legal_moves(self):
        """가능한 모든 수 반환"""
        pass

    def apply_move(self, move):
        """수를 적용한 새 보드 반환"""
        pass

    def is_terminal(self):
        """게임 종료 확인"""
        pass

    def get_result(self):
        """현재 플레이어 관점 결과"""
        pass
```

**시간 기반 MCTS**:
```python
def mcts_with_time_limit(board, time_limit_ms):
    root = MCTSNode(board)
    start = time.time()
    end = start + time_limit_ms / 1000.0

    iterations = 0
    while time.time() < end:
        # MCTS 1 iteration
        ...
        iterations += 1

    print(f"Iterations: {iterations}", file=sys.stderr)
    return best_move
```

**시간 관리 전략**:
```python
if my_time > 1000:
    time_limit = 150  # 여유 있음
else:
    time_limit = 10   # 시간 부족!
```

### 실습 진행 (10분)

**교사**: 이제 ATAXX MCTS 에이전트를 완성해봅시다.

**[실습 파일 제공]**
- `alphano/ataxx_board.py`: 보드 클래스 (완성)
- `alphano/mcts_agent.py`: MCTS 에이전트 (빈칸 채우기)

**실습 과제**:
```
1. AtaxxBoard 클래스 확인
   - get_legal_moves() 이해
   - apply_move() 이해
   - get_result() 이해

2. MCTS 에이전트 완성
   - MCTSNode 클래스 (틱택토와 동일)
   - mcts_with_time_limit() 구현
   - 시간 관리 추가

3. 로컬 테스트
   - 랜덤 vs MCTS 대결
```

**교사**: 10분 드릴게요!

**[학생들 실습 진행]**

### 성능 실험 (5분)

**교사**: 이제 재미있는 실험을 해봅시다!

**실험 1: 시뮬레이션 횟수 vs 승률**

```python
# 테스트 코드
for iterations in [100, 500, 1000, 5000]:
    win_rate = test_mcts(iterations, opponent='random', games=20)
    print(f"{iterations} iterations: {win_rate}% 승률")
```

**예상 결과**:
```
100 iterations:   55% 승률
500 iterations:   68% 승률
1000 iterations:  75% 승률
5000 iterations:  85% 승률
```

**교사**: 시뮬레이션을 많이 할수록 강해지지만, 수확 체감 법칙이 적용됩니다.

**실험 2: MCTS vs Alpha-Beta**

```python
# MCTS (1000 iterations) vs Alpha-Beta (깊이 4)
result = play_game(mcts_agent, alphabeta_agent)
```

🎯 **학생들에게 질문**: "누가 이길 것 같나요?"

**교사**: 실제로 해보면 비슷하거나 MCTS가 약간 유리할 수 있습니다. 시간을 많이 주면 MCTS가 더 강해지죠.

---

## 정리 및 과제 안내 (75-85분)

### 핵심 내용 정리 (5분)

**교사**: 오늘 배운 내용을 정리해봅시다.

**[슬라이드: 핵심 요약]**

**MCTS란?**
```
- 시뮬레이션 기반 게임 트리 탐색
- 평가 함수 없이도 작동
- UCB1로 탐험-활용 균형
```

**4단계**:
```
1. Selection: UCB1으로 유망한 노드 선택
2. Expansion: 새 자식 추가
3. Simulation: 게임 끝까지 랜덤 플레이
4. Backpropagation: 결과 역전파
```

**장점**:
```
✓ 도메인 지식 불필요
✓ 복잡한 게임에 효과적
✓ 언제든 중단 가능
```

**단점**:
```
✗ 수렴 느림
✗ 메모리 많이 사용
```

### 과제 안내 (3분)

**교사**: 과제는 3가지입니다.

**[화면: 과제]**

**과제 1: ATAXX MCTS 에이전트 완성 및 제출**
```
- alphano/mcts_agent.py 완성
- ALPHANO 프로토콜 준수
- 로컬 테스트 후 제출
- 제출 기한: 다음 주 일요일 23:59
```

**과제 2: 성능 실험 보고서**
```
다음 실험을 수행하고 결과 정리:

1. 시뮬레이션 횟수 vs 승률
   - 100, 500, 1000, 5000, 10000 iterations
   - 각각 랜덤 상대로 20게임

2. C 값 변화에 따른 성능
   - C = 0.5, 1.0, 1.414, 2.0, 3.0
   - 각각 랜덤 상대로 20게임

3. MCTS vs Alpha-Beta
   - MCTS (1000 iterations)
   - Alpha-Beta (깊이 4)
   - 10게임 대결

결과를 표와 그래프로 정리
```

**과제 3: 개선 시도 (선택)**
```
다음 중 하나를 구현:

1. Heavy Rollout
   - 랜덤 대신 간단한 휴리스틱
   - 예: 중앙 선호, 상대 많은 곳 선호

2. 조기 종료
   - 시뮬레이션 깊이 제한
   - 평가 함수로 추정

3. 시간 관리 개선
   - 상황에 따라 시간 조절
   - 초반/중반/종반 다르게

개선 전후 성능 비교
```

### 다음 주 예고 (2분)

**교사**: 다음 주는 강화학습입니다!

**[슬라이드: Week 5 Preview]**

**강화학습 (Reinforcement Learning)**:
```
- Agent가 환경과 상호작용하며 학습
- Reward를 최대화하는 Policy 학습
- Q-Learning 알고리즘
- ATAXX Q-Learning 에이전트 구현
```

**MCTS와의 관계**:
```
MCTS: 시뮬레이션으로 즉시 학습
RL:   경험을 통해 점진적 학습

둘 다 시행착오(trial-and-error)를 통해 학습!
```

**교사**: 강화학습은 게임 AI뿐만 아니라 로봇, 자율주행 등 다양한 분야에 적용됩니다. 기대해주세요!

---

## 질의응답 (85-90분)

**교사**: 질문 있나요?

### 예상 질문 및 답변

🎯 **질문 1**: "UCB1의 C 값은 어떻게 정하나요?"

**답변**: 이론적 최적값은 √2 ≈ 1.414입니다. 하지만 실전에서는 게임에 따라 조정합니다. 보통 0.7~2.0 사이에서 실험해보세요. C가 크면 탐험을 많이 하고, 작으면 좋은 수에 집중합니다.

🎯 **질문 2**: "MCTS는 항상 최적의 수를 찾나요?"

**답변**: 아니요. MCTS는 시간이 무한하면 최적에 수렴하지만, 실전에서는 시간 제한이 있습니다. 충분한 시뮬레이션을 하면 매우 좋은 수를 찾지만, 완벽하지는 않아요. 특히 전술적인 수(tactical move)를 놓칠 수 있습니다.

🎯 **질문 3**: "AlphaGo도 MCTS를 사용하나요?"

**답변**: 네! AlphaGo는 MCTS + 딥러닝을 결합했습니다. 랜덤 시뮬레이션 대신 신경망으로 국면을 평가하고, UCB1에 사전 확률(prior)을 추가했어요. MCTS의 구조를 그대로 사용하면서 성능을 크게 향상시켰죠.

🎯 **질문 4**: "Heavy Rollout은 어떻게 만드나요?"

**답변**: 간단한 휴리스틱을 추가하면 됩니다. 예를 들어:
```python
def heavy_rollout(state):
    while not state.is_terminal():
        moves = state.get_legal_moves()
        # 완전 랜덤 대신
        scored = [(m, score_move(state, m)) for m in moves]
        move = weighted_choice(scored)
        state.apply_move(move)
    return state.get_result()
```
ATAXX에서는 "상대 돌 많은 곳", "중앙 위치" 등에 보너스를 줄 수 있어요.

🎯 **질문 5**: "메모리가 부족하면 어떻게 하나요?"

**답변**: 트리가 너무 커지면 메모리 문제가 생길 수 있습니다. 해결 방법:
1. 오래된 노드 삭제
2. 방문 횟수 적은 노드 정리
3. 트리 재사용 (같은 국면 재방문 시)
4. 메모리 풀(pool) 사용

하지만 일반적인 게임에서는 크게 걱정 안 해도 됩니다.

🎯 **질문 6**: "MCTS가 항상 Alpha-Beta보다 느린가요?"

**답변**: 상황에 따라 다릅니다.
- 짧은 시간: Alpha-Beta가 더 빠르게 합리적인 수 선택
- 긴 시간: MCTS가 더 나은 수 찾을 수 있음
- 복잡한 게임: MCTS가 평가 함수 없어서 유리
- 간단한 게임: Alpha-Beta가 완전 탐색 가능해서 유리

둘을 결합하는 것도 좋은 방법입니다!

---

## 수업 마무리

**교사**: 오늘 수고 많았습니다! MCTS는 현대 게임 AI의 중요한 기법입니다. 과제 열심히 하고, 다음 주에 만나요!

**[수업 종료]**

---

## 교사 노트

### 수업 준비 사항

**필요한 자료**:
- [ ] 강의 슬라이드 (PPT/PDF)
- [ ] 틱택토 실습 코드
- [ ] ATAXX 보드 클래스
- [ ] MCTS 템플릿 코드
- [ ] 테스트 스크립트

**준비 사항**:
- [ ] 프로젝터 및 화면 공유
- [ ] 학생 코딩 환경 확인
- [ ] 예제 코드 동작 테스트
- [ ] 실습 시간 타이머

### 난이도 조절

**쉽게 만들기**:
- 틱택토부터 충분히 연습
- UCB1 공식 계산은 넘어가기
- 완성된 코드 제공하고 이해만 시키기

**어렵게 만들기**:
- UCB1 수학적 배경 설명
- RAVE 등 고급 기법 소개
- 최적화 기법 논의

### 자주 하는 실수

**학생들이 자주 틀리는 부분**:
1. **관점 전환 빠뜨림**
   ```python
   # 잘못
   def backpropagate(self, result):
       node = self
       while node:
           node.wins += result  # 관점 전환 안 함!
           node = node.parent

   # 올바름
   def backpropagate(self, result):
       node = self
       while node:
           node.wins += result
           result = 1 - result  # 관점 전환!
           node = node.parent
   ```

2. **UCB1에서 visits=0 처리 안 함**
   ```python
   # 잘못
   def ucb1(self):
       return self.wins / self.visits + ...  # ZeroDivisionError!

   # 올바름
   def ucb1(self):
       if self.visits == 0:
           return float('inf')
       return self.wins / self.visits + ...
   ```

3. **보드 복사 안 함**
   ```python
   # 잘못
   def rollout(self):
       state = self.state  # 참조만 복사!
       # state를 변경하면 원본도 변경됨

   # 올바름
   def rollout(self):
       state = self.state.copy()  # 깊은 복사
   ```

### 시간 관리 팁

- **이론이 길어지면**: 실습 2를 축소하거나 과제로 전환
- **실습이 늦어지면**: 이론 2를 간략히
- **질문이 많으면**: 시간 조절하되 중요한 질문은 충분히 답변

### 추가 자료

**심화 학습**:
- MCTS Survey Paper
- AlphaGo 논문
- UCB 이론 자료

**코드 저장소**:
- GitHub에 예제 코드 업로드
- 학생들이 참고할 수 있도록

---

**수업 준비 완료! 화이팅!**
