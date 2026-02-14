# Week 4 MCTS 빠른 시작 가이드

## 5분 안에 시작하기

### 1단계: 파일 확인 (30초)

```bash
cd week04_mcts
ls -l
```

다음 파일들이 있어야 합니다:
- `lecture.md` - 수업 자료
- `script.md` - 수업 대본
- `tictactoe_example.py` - 틱택토 예제
- `alphano/mcts_agent.py` - ALPHANO 제출 코드
- `alphano/test_mcts.py` - 테스트 코드

### 2단계: 틱택토 예제 실행 (2분)

```bash
python3 tictactoe_example.py
```

선택지:
- `1`: MCTS vs 랜덤 (1게임 시연)
- `2`: MCTS vs MCTS (1게임 시연)
- `3`: 성능 테스트
- `4`: 모두 실행

**추천**: 먼저 `1`을 선택해서 MCTS가 어떻게 동작하는지 봅시다!

### 3단계: ATAXX MCTS 테스트 (2분)

```bash
cd alphano
python3 test_mcts.py
```

모든 테스트가 통과하면 성공!

### 4단계: ALPHANO 제출 준비 (30초)

```bash
# mcts_agent.py가 준비되었습니다!
python3 mcts_agent.py
```

입력 예시:
```
READY FIRST
TURN 5000 5000
```

출력 예시:
```
OK
MOVE 1 1 2 2
```

---

## 학습 로드맵

### 초급 (1-2시간)

**목표**: MCTS 개념 이해

1. **강의 자료 읽기** (30분)
   ```bash
   # lecture.md의 1-3장 읽기
   # - MCTS 개요
   # - 4단계 설명
   # - UCB1 공식
   ```

2. **틱택토 예제 실습** (30분)
   ```bash
   python3 tictactoe_example.py
   # 1, 2, 3번 모두 실행해보기
   ```

3. **코드 이해** (30분)
   ```python
   # tictactoe_example.py의 MCTSNode 클래스 읽기
   # - ucb1() 메서드
   # - expand() 메서드
   # - rollout() 메서드
   # - backpropagate() 메서드
   ```

### 중급 (2-3시간)

**목표**: ATAXX MCTS 구현

1. **ATAXX 보드 이해** (30분)
   ```python
   # alphano/mcts_agent.py의 AtaxxBoard 읽기
   # - get_legal_moves()
   # - apply_move()
   # - is_terminal()
   ```

2. **MCTS 적용** (1시간)
   ```python
   # MCTSNode 클래스 완성
   # mcts_search() 함수 완성
   ```

3. **테스트** (30분)
   ```bash
   python3 test_mcts.py
   # 모든 테스트 통과 확인
   ```

4. **시간 관리 추가** (30분)
   ```python
   # 남은 시간에 따라 탐색 시간 조절
   if my_time > 1000:
       time_limit = 150
   else:
       time_limit = 10
   ```

### 고급 (3-5시간)

**목표**: 성능 개선

1. **Heavy Rollout** (1시간)
   ```python
   def heavy_rollout(state):
       """휴리스틱 기반 롤아웃"""
       while not state.is_terminal():
           moves = state.get_legal_moves()
           # 간단한 평가로 수 선택
           scored = [(m, score_move(m)) for m in moves]
           move = weighted_choice(scored)
           state.apply_move(move)
       return state.get_result()
   ```

2. **성능 실험** (1-2시간)
   - 시뮬레이션 횟수 vs 승률
   - C 값 변화에 따른 성능
   - MCTS vs Alpha-Beta

3. **최적화** (1-2시간)
   - 빠른 보드 복사
   - 합법 수 캐싱
   - 조기 종료

---

## 핵심 개념 치트시트

### MCTS 4단계

```
1. Selection (선택)
   UCB1 = w/n + C√(ln(N)/n)
   → 가장 큰 값을 가진 자식 선택
   → 리프 노드까지 반복

2. Expansion (확장)
   → 미탐색 수 중 하나 선택
   → 새 자식 노드 생성

3. Simulation (시뮬레이션)
   → 게임 끝까지 랜덤 플레이
   → 결과 확인 (승/패/무)

4. Backpropagation (역전파)
   → 루트까지 거슬러 올라감
   → visits += 1, wins += result
   → result = 1 - result (관점 전환!)
```

### UCB1 공식

```python
if visits == 0:
    return float('inf')  # 미방문 노드 최우선

exploitation = wins / visits  # 평균 승률
exploration = C * sqrt(log(parent.visits) / visits)  # 불확실성 보너스

return exploitation + exploration
```

### 자주 하는 실수

**1. 관점 전환 빠뜨림**
```python
# ✗ 잘못
def backpropagate(self, result):
    node = self
    while node:
        node.wins += result  # 항상 같은 result!
        node = node.parent

# ✓ 올바름
def backpropagate(self, result):
    node = self
    while node:
        node.wins += result
        result = 1 - result  # 관점 전환!
        node = node.parent
```

**2. visits=0 처리 안 함**
```python
# ✗ 잘못
def ucb1(self):
    return self.wins / self.visits + ...  # ZeroDivisionError!

# ✓ 올바름
def ucb1(self):
    if self.visits == 0:
        return float('inf')
    return self.wins / self.visits + ...
```

**3. 보드 복사 안 함**
```python
# ✗ 잘못
def rollout(self):
    state = self.state  # 참조만 복사!

# ✓ 올바름
def rollout(self):
    state = self.state.copy()  # 깊은 복사!
```

---

## 자주 묻는 질문 (FAQ)

### Q1: UCB1의 C 값은 어떻게 정하나요?

**A**: 이론적 최적값은 √2 ≈ 1.414입니다. 실전에서는:
- **C=0.5**: 활용 중심 (좋은 수에 집중)
- **C=1.414**: 균형 (기본값)
- **C=2.0**: 탐험 중심 (넓게 탐색)

실험해보고 게임에 맞는 값을 찾으세요!

### Q2: 시뮬레이션을 몇 번 해야 하나요?

**A**: 시간에 따라 다릅니다:
- **10ms**: ~100 iterations
- **50ms**: ~500 iterations
- **100ms**: ~1000 iterations
- **500ms**: ~5000 iterations

많을수록 강하지만, 수확 체감이 있습니다.

### Q3: MCTS가 Alpha-Beta보다 느린가요?

**A**: 상황에 따라 다릅니다:
- **짧은 시간**: Alpha-Beta 유리
- **긴 시간**: MCTS 유리
- **복잡한 게임**: MCTS 유리 (평가 함수 불필요)

### Q4: PASS는 어떻게 처리하나요?

**A**: 가능한 수가 없으면 None 반환:
```python
moves = board.get_legal_moves()
if not moves:
    moves = [None]
```

### Q5: 테스트가 실패하면?

**A**: 다음을 확인하세요:
1. 관점 전환 (1 - result)
2. visits=0 처리
3. 보드 복사 (copy())
4. 게임 규칙 (Split/Jump)

---

## 디버깅 팁

### stderr로 로그 출력

```python
import sys

# ALPHANO는 stderr를 무시하므로 자유롭게 출력
print(f"Iterations: {iterations}", file=sys.stderr)
print(f"Best move: {move}, visits: {visits}", file=sys.stderr)
print(f"Winrate: {wins/visits:.2%}", file=sys.stderr)
```

### 보드 상태 출력

```python
def print_board(board):
    """디버깅용 보드 출력"""
    symbols = {0: '.', 1: 'X', 2: 'O'}
    for i, row in enumerate(board.board):
        print(f"{i+1} {' '.join(symbols[c] for c in row)}", file=sys.stderr)
```

### UCB1 값 확인

```python
for child in node.children:
    ucb1_value = child.ucb1()
    print(f"Move {child.move}: UCB1={ucb1_value:.3f}, "
          f"visits={child.visits}, winrate={child.wins/child.visits:.2%}",
          file=sys.stderr)
```

---

## 성능 벤치마크

### 틱택토 (1000 iterations)

```
MCTS vs 랜덤: 95% 승률
MCTS vs MCTS: 50% 승률 (무승부 많음)
시간: ~100ms per move
```

### ATAXX (1000 iterations)

```
MCTS vs 랜덤: 75-85% 승률
MCTS vs Alpha-Beta(깊이4): 비슷
시간: ~100ms per move
```

---

## 다음 단계

### Week 5 예고: 강화학습

```python
# MCTS: 시뮬레이션으로 즉시 학습
move = mcts_search(board, 1000)

# RL: 경험을 통해 점진적 학습
q_table = train_q_learning(episodes=10000)
move = get_best_action(q_table, state)
```

두 방법 모두 **시행착오(trial-and-error)**를 통해 학습합니다!

---

## 리소스

### 파일별 용도

| 파일 | 용도 | 우선순위 |
|------|------|----------|
| `lecture.md` | 이론 학습 | ⭐⭐⭐ |
| `tictactoe_example.py` | 간단한 실습 | ⭐⭐⭐ |
| `alphano/mcts_agent.py` | 제출 코드 | ⭐⭐⭐ |
| `alphano/test_mcts.py` | 테스트 | ⭐⭐ |
| `script.md` | 수업 참고 | ⭐ |
| `README.md` | 종합 가이드 | ⭐⭐ |

### 추천 학습 순서

```
1. lecture.md (1-3장) 읽기
2. tictactoe_example.py 실행
3. tictactoe_example.py 코드 읽기
4. alphano/mcts_agent.py 읽기
5. alphano/test_mcts.py 실행
6. 성능 실험
7. ALPHANO 제출
```

---

## 도움이 필요하면?

### 체크리스트

- [ ] 틱택토 예제가 실행되나요?
- [ ] MCTS 4단계를 이해했나요?
- [ ] UCB1 공식을 이해했나요?
- [ ] 테스트가 모두 통과하나요?
- [ ] ALPHANO 프로토콜을 이해했나요?

### 문제 해결

1. **코드 오류**: `test_mcts.py` 실행
2. **개념 이해**: `lecture.md` 다시 읽기
3. **구현 막힘**: `tictactoe_example.py` 참고
4. **성능 개선**: `README.md`의 최적화 섹션

---

**시작하세요! MCTS의 세계로!**

Monte Carlo Tree Search는 평가 함수 없이도 강력한 게임 AI를 만들 수 있는 혁신적인 알고리즘입니다. AlphaGo의 핵심 기술이기도 하죠!

행운을 빕니다! 🎮🤖
