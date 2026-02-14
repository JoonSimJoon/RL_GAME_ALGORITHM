# Week 4: Monte Carlo Tree Search (MCTS)

## 파일 구조

```
week04_mcts/
├── README.md                  # 이 파일
├── lecture.md                 # 수업 자료 (500+ lines)
├── script.md                  # 수업 대본 (600+ lines)
└── alphano/
    └── mcts_agent.py         # ALPHANO 제출 코드
```

## 개요

이번 주차에서는 **Monte Carlo Tree Search (MCTS)**를 학습합니다.

### 주요 내용

1. **평가 함수의 한계**: Alpha-Beta의 문제점
2. **시뮬레이션 기반 접근**: 실제 게임 결과로 평가
3. **MCTS 4단계**:
   - Selection (UCB1)
   - Expansion
   - Simulation (Rollout)
   - Backpropagation
4. **UCB1 공식**: 탐험-활용 균형
5. **MCTS vs Alpha-Beta 비교**

## 수업 자료

### lecture.md

500줄 이상의 강의 자료:
- MCTS 개념 및 동기
- 4단계 상세 설명 (ASCII art 포함)
- UCB1 공식 및 수학적 배경
- 완전한 의사코드
- Alpha-Beta 비교
- 개선 기법 (Heavy Rollout, RAVE 등)
- 핵심 정리 및 다음 주 예고

### script.md

90분 수업 대본:
- 도입 (5분): 동기부여
- 이론 1 (20분): MCTS 4단계 + UCB1
- 실습 1 (20분): 틱택토 MCTS
- 이론 2 (10분): MCTS vs Alpha-Beta
- 실습 2 (20분): ATAXX MCTS
- 정리 (10분): 과제 안내
- 질의응답 (5분)

## ALPHANO 제출 코드

### alphano/mcts_agent.py

완전한 MCTS 에이전트 구현:

**주요 기능**:
- `AtaxxBoard`: 7×7 ATAXX 게임 보드
  - `get_legal_moves()`: 가능한 수 생성
  - `apply_move()`: 수 적용 (불변)
  - `is_terminal()`: 게임 종료 확인
  - `get_result()`: 게임 결과 반환

- `MCTSNode`: MCTS 트리 노드
  - `ucb1()`: UCB1 값 계산
  - `select_child()`: 최선의 자식 선택
  - `expand()`: 새 자식 추가
  - `rollout()`: 게임 끝까지 시뮬레이션
  - `backpropagate()`: 결과 역전파

- `mcts_search()`: 시간 제한이 있는 MCTS
  - 4단계 반복
  - 시간 관리
  - 최종 수 선택

**시간 관리**:
```python
if my_time > 1000:
    time_limit = 150  # 여유 있음
else:
    time_limit = 10   # 시간 부족
```

## 로컬 테스트

### 실행 방법

```bash
cd week04_mcts/alphano
python3 mcts_agent.py
```

### 입력 예시

```
READY FIRST
TURN 5000 5000
OPP 1 7 1 5
TURN 4850 4850
FINISH
```

### 출력 예시

```
OK
MOVE 1 1 2 2
OK
```

## 실습 과제

### 과제 1: 기본 MCTS 구현

틱택토로 MCTS 연습:

```python
# 1. MCTSNode 클래스 완성
# 2. mcts() 함수 구현
# 3. 테스트: MCTS vs 랜덤
```

### 과제 2: ATAXX MCTS 완성

ALPHANO 제출용 코드:

```python
# 1. AtaxxBoard 이해
# 2. MCTS 적용
# 3. 시간 관리 추가
# 4. 제출
```

### 과제 3: 성능 실험

다음 실험 수행:

**실험 1**: 시뮬레이션 횟수 vs 승률
```
100, 500, 1000, 5000, 10000 iterations
각각 랜덤 상대로 20게임
```

**실험 2**: C 값 변화에 따른 성능
```
C = 0.5, 1.0, 1.414, 2.0, 3.0
각각 랜덤 상대로 20게임
```

**실험 3**: MCTS vs Alpha-Beta
```
MCTS (1000 iterations) vs Alpha-Beta (깊이 4)
10게임 대결
```

### 과제 4: 개선 시도 (선택)

다음 중 하나 구현:

**1. Heavy Rollout**
```python
def heavy_rollout(state):
    """휴리스틱 기반 롤아웃"""
    while not state.is_terminal():
        moves = state.get_legal_moves()
        # 간단한 평가로 수 선택
        scored = [(m, evaluate_move(m)) for m in moves]
        move = weighted_choice(scored)
        state.apply_move(move)
    return state.get_result()
```

**2. 조기 종료**
```python
def rollout_with_limit(state, max_depth=50):
    """깊이 제한 후 평가 함수 사용"""
    depth = 0
    while not state.is_terminal() and depth < max_depth:
        move = random.choice(state.get_legal_moves())
        state.apply_move(move)
        depth += 1

    if state.is_terminal():
        return state.get_result()
    else:
        return evaluate(state)
```

**3. 시간 관리 개선**
```python
def adaptive_time_limit(my_time, turn_number):
    """상황에 따른 시간 조절"""
    if turn_number < 10:
        return 50  # 초반: 빠르게
    elif turn_number < 30:
        return 150  # 중반: 충분히
    else:
        return 100  # 종반: 적당히
```

## 핵심 개념

### MCTS 4단계

```
1. Selection (선택)
   → UCB1으로 유망한 노드 선택
   → 리프까지 내려감

2. Expansion (확장)
   → 새 자식 노드 추가
   → 미탐색 수 중 하나 선택

3. Simulation (시뮬레이션)
   → 게임 끝까지 랜덤 플레이
   → 실제 결과 확인 (승/패/무)

4. Backpropagation (역전파)
   → 결과를 루트까지 전파
   → 모든 노드의 통계 업데이트
   → 관점 전환 (result = 1 - result)
```

### UCB1 공식

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

### MCTS vs Alpha-Beta

| 항목 | Alpha-Beta | MCTS |
|------|-----------|------|
| **평가 방법** | 평가 함수 | 시뮬레이션 |
| **평가 함수** | 필수 | 불필요 |
| **시간 제어** | 깊이 제한 | 시뮬레이션 횟수 |
| **장점** | 정확, 빠름 | 도메인 지식 불필요 |
| **단점** | 평가 함수 필요 | 수렴 느림 |

## 디버깅 팁

### 자주 하는 실수

**1. 관점 전환 빠뜨림**
```python
# 잘못
def backpropagate(self, result):
    node = self
    while node:
        node.wins += result  # 항상 같은 result
        node = node.parent

# 올바름
def backpropagate(self, result):
    node = self
    while node:
        node.wins += result
        result = 1 - result  # 관점 전환!
        node = node.parent
```

**2. visits=0 처리 안 함**
```python
# 잘못
def ucb1(self):
    return self.wins / self.visits + ...  # ZeroDivisionError

# 올바름
def ucb1(self):
    if self.visits == 0:
        return float('inf')
    return self.wins / self.visits + ...
```

**3. 보드 복사 안 함**
```python
# 잘못
def rollout(self):
    state = self.state  # 참조만 복사
    # state 변경하면 원본도 변경됨

# 올바름
def rollout(self):
    state = self.state.copy()  # 깊은 복사
```

### 디버깅 출력

```python
# stderr로 출력하면 ALPHANO가 무시
print(f"Iterations: {iterations}", file=sys.stderr)
print(f"Best move: {move}, visits: {visits}", file=sys.stderr)
print(f"Winrate: {wins/visits:.2%}", file=sys.stderr)
```

## 성능 최적화

### 빠른 보드 복사

```python
from copy import deepcopy

def copy(self):
    """보드 복사 최적화"""
    new_board = AtaxxBoard()
    # 리스트 컴프리헨션으로 복사
    new_board.board = [row[:] for row in self.board]
    new_board.current_player = self.current_player
    return new_board
```

### 합법 수 캐싱

```python
def get_legal_moves(self):
    """합법 수 캐싱"""
    if hasattr(self, '_cached_moves'):
        return self._cached_moves

    moves = self._compute_legal_moves()
    self._cached_moves = moves
    return moves
```

### 롤아웃 조기 종료

```python
def rollout(self, max_depth=50):
    """일정 깊이 후 평가 함수 사용"""
    state = self.state.copy()
    depth = 0

    while not state.is_terminal() and depth < max_depth:
        move = random.choice(state.get_legal_moves())
        state.apply_move(move)
        depth += 1

    if state.is_terminal():
        return state.get_result()
    else:
        # 평가 함수로 추정
        return self._evaluate(state)
```

## 참고 자료

### 논문

1. **Kocsis & Szepesvári (2006)**: "Bandit based Monte-Carlo Planning"
   - UCT 제안
   - 이론적 기초

2. **Coulom (2006)**: "Efficient Selectivity and Backup Operators in Monte-Carlo Tree Search"
   - MCTS 체계화

3. **Silver et al. (2016)**: "Mastering the game of Go with deep neural networks and tree search"
   - AlphaGo
   - MCTS + Deep Learning

### 온라인 자료

- [MCTS Deep Dive](https://www.moderndescartes.com/essays/deep_dive_mcts/)
- [Intro to MCTS](https://jeffbradberry.com/posts/2015/09/intro-to-monte-carlo-tree-search/)
- [Wikipedia: Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search)

### 코드 예제

- Python MCTS: https://github.com/pbsinclair42/MCTS
- C++ MCTS: https://github.com/memo/ofxMSAmcts

## FAQ

### Q1: UCB1의 C 값은 어떻게 정하나요?

**A**: 이론적 최적값은 √2 ≈ 1.414입니다. 실전에서는 0.7~2.0 사이에서 실험하세요.
- C 크면: 탐험 많이 (트리 넓게)
- C 작으면: 활용 많이 (좋은 수에 집중)

### Q2: MCTS는 항상 최적의 수를 찾나요?

**A**: 아니요. 시간이 무한하면 최적에 수렴하지만, 실전에서는 시간 제한이 있습니다. 충분한 시뮬레이션을 하면 매우 좋은 수를 찾지만 완벽하지는 않습니다.

### Q3: Heavy Rollout은 어떻게 만드나요?

**A**: 간단한 휴리스틱을 추가하면 됩니다:
```python
def score_move(state, move):
    """수의 점수 평가"""
    score = 0
    if is_center(move):
        score += 2  # 중앙 선호
    score += count_adjacent_opponents(state, move)  # 감염 선호
    if is_split(move):
        score += 1  # Split 선호
    return score
```

### Q4: 메모리가 부족하면?

**A**: 트리 정리 전략:
1. 방문 횟수 적은 노드 삭제
2. 오래된 노드 제거
3. 트리 재사용 (같은 국면 재방문)
4. 메모리 풀 사용

### Q5: MCTS가 Alpha-Beta보다 느린가요?

**A**: 상황에 따라 다릅니다:
- 짧은 시간: Alpha-Beta 유리
- 긴 시간: MCTS 유리
- 복잡한 게임: MCTS 유리 (평가 함수 불필요)
- 간단한 게임: Alpha-Beta 유리 (완전 탐색 가능)

## 제출 방법

1. `alphano/mcts_agent.py` 완성
2. 로컬 테스트
3. ALPHANO 플랫폼에 제출
4. 리더보드 확인

## 라이선스

교육 목적으로 자유롭게 사용 가능합니다.

---

**Week 4: MCTS 학습을 즐기세요!**
