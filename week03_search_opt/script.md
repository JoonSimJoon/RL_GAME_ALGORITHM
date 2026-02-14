# Week 3 수업 대본: 탐색 최적화

**수업 시간:** 90분
**대상:** 고등학생
**주제:** Iterative Deepening, Transposition Table, PVS

---

## 수업 전 준비 (강사용)

- [ ] 프로젝터 및 코드 에디터 준비
- [ ] ATAXX 게임 데모 환경 준비
- [ ] 지난 주 Alpha-Beta 코드 확인
- [ ] 성능 측정 스크립트 준비
- [ ] 학생들 개발 환경 점검

---

## 1. 도입 (5분)

### [00:00-00:02] 인사 및 출석

"안녕하세요, 여러분! 오늘은 Week 3 수업입니다. 모두 지난 주 Alpha-Beta Pruning 과제는 완성하셨나요?"

*학생 반응 확인*

"좋습니다. 오늘은 지난 주에 배운 Alpha-Beta를 더욱 강력하게 만드는 세 가지 핵심 기법을 배웁니다."

### [00:02-00:05] 지난 주 복습

"먼저 간단히 복습하겠습니다. Alpha-Beta Pruning의 핵심은 무엇이었죠?"

🎯 **학생 질문:** "가지치기요! 불필요한 노드를 탐색하지 않는 거요."

"맞습니다! Alpha-Beta는 Minimax보다 약 5~10배 빠릅니다. 최선의 경우 b^d 노드를 b^(d/2)로 줄일 수 있죠."

*화면에 표시:*
```
Minimax: b^d (예: 35^6 = 1.8억 노드)
Alpha-Beta: b^(d/2) (예: 35^3 = 43,000 노드)
→ 약 4000배 차이!
```

"하지만 여전히 문제가 있습니다. 뭘까요?"

🎯 **학생 질문:** "깊이를 얼마로 설정해야 할지 모른다?"

"정확합니다! 오늘 첫 번째 주제가 바로 그 문제를 해결하는 Iterative Deepening입니다."

---

## 2. 이론 1: Iterative Deepening (10분)

### [00:05-00:08] 문제 상황 제시

"여러분이 ATAXX 대회에 참가한다고 생각해봅시다. 턴당 시간 제한은 2초입니다."

*화면에 두 가지 시나리오 표시:*

```
시나리오 1: depth=4로 설정
- 탐색 시간: 0.1초
- 남은 시간: 1.9초 낭비!
- 문제: 더 깊이 탐색할 수 있었는데...

시나리오 2: depth=10으로 설정
- 탐색 시간: 5분 (예상)
- 시간 제한: 2초
- 문제: 시간 초과로 실격!
```

"어떻게 해야 할까요?"

🎯 **학생 질문:** "딱 맞는 깊이를 찾는 공식이 있나요?"

"아쉽게도 없습니다. 보드 상태마다 복잡도가 다르기 때문이죠. 초반에는 수가 많고, 후반에는 적습니다."

### [00:08-00:12] Iterative Deepening 해결책

"해결책은 바로 **Iterative Deepening**입니다. 깊이 1부터 시작해서 시간이 허락하는 한 계속 깊이를 늘리는 거죠."

*화면에 코드 표시:*

```python
def iterative_deepening(board, time_limit):
    best_move = None
    start_time = current_time()

    for depth in range(1, MAX_DEPTH):
        # 시간 체크
        if elapsed_time(start_time) > time_limit * 0.9:
            break  # 90% 도달하면 중단

        # 현재 깊이 탐색
        score, move = alpha_beta(board, depth, -INF, INF)
        best_move = move  # 더 깊은 결과로 갱신

    return best_move
```

"이렇게 하면 어떤 장점이 있을까요?"

🎯 **학생 질문:** "시간이 부족하면 얕게, 충분하면 깊게 탐색할 수 있어요!"

"정확합니다! 그런데 의문이 들 수 있습니다. 같은 보드를 여러 번 탐색하면 시간 낭비 아닌가요?"

### [00:12-00:15] 시간 복잡도 분석

"실제로 계산해봅시다."

*화면에 표 표시:*

```
분기 인수 b=35, 목표 깊이 d=6

depth=1: 35^0.5 ≈ 6 노드
depth=2: 35^1 = 35 노드
depth=3: 35^1.5 ≈ 207 노드
depth=4: 35^2 = 1,225 노드
depth=5: 35^2.5 ≈ 7,218 노드
depth=6: 35^3 = 42,875 노드
───────────────────────────────
합계: 51,566 노드

단일 탐색 (depth=6): 42,875 노드
오버헤드: (51,566 - 42,875) / 42,875 = 20%
```

"오버헤드가 약 20%입니다. 하지만 실전에서는 다른 장점들이 이를 상쇄하고도 남습니다. 무엇일까요?"

🎯 **학생 질문:** "시간 관리 자동화?"

"맞습니다! 그리고 하나 더 있습니다. 이전 깊이에서 찾은 최선의 수를 다음 깊이에서 먼저 탐색하면 어떻게 될까요?"

🎯 **학생 질문:** "Alpha-Beta 가지치기가 더 잘 되겠네요!"

"정확합니다! 이것이 **Move Ordering**인데, 잠시 후 자세히 다룹니다. 결과적으로 Iterative Deepening은 **+50 Elo** 향상을 가져옵니다."

---

## 3. 실습 1: Iterative Deepening 구현 (10분)

### [00:15-00:20] 코드 작성

"자, 이제 직접 구현해봅시다. 지난 주 Alpha-Beta 코드를 엽니다."

*실습 파일: `id_agent.py`*

```python
import time

INF = float('inf')

class ATAXXBoard:
    # (지난 주 코드 사용)
    pass

def alpha_beta(board, depth, alpha, beta):
    # (지난 주 구현)
    pass

def iterative_deepening(board, time_limit_ms):
    """Iterative Deepening 구현"""
    start_time = time.time()
    best_move = None

    for depth in range(1, 50):  # 최대 깊이 50
        # 시간 체크 (밀리초 단위)
        elapsed = (time.time() - start_time) * 1000
        if elapsed > time_limit_ms * 0.9:
            print(f"Time limit reached at depth {depth-1}")
            break

        # 현재 깊이 탐색
        score, move = alpha_beta(board, depth, -INF, INF)

        if move:
            best_move = move
            print(f"Depth {depth}: score={score}, move={move}")

    return best_move

# 테스트
if __name__ == "__main__":
    board = ATAXXBoard()
    # 초기 보드 설정
    move = iterative_deepening(board, 1000)  # 1초 제한
    print(f"Best move: {move}")
```

"모두 따라 작성해보세요. 5분 드립니다."

*학생들 작성 시간*

### [00:20-00:25] 실행 및 검증

"완성하신 분들은 실행해보세요."

```bash
python id_agent.py
```

*예상 출력:*
```
Depth 1: score=2, move=(1,1,1,2)
Depth 2: score=3, move=(1,1,1,2)
Depth 3: score=5, move=(1,1,2,2)
Depth 4: score=5, move=(1,1,2,2)
Depth 5: score=7, move=(1,1,2,2)
Time limit reached at depth 5
Best move: (1,1,2,2)
```

"출력을 보면 깊이가 증가할수록 점수가 더 정확해집니다. 그리고 시간 제한에 도달하면 자동으로 중단되죠."

🎯 **학생 질문:** "왜 depth 1, 2에서는 같은 수인데 점수가 다른가요?"

"좋은 질문입니다! 얕은 깊이에서는 평가가 부정확하기 때문입니다. 깊이 1은 1수만 보지만, 깊이 2는 2수를 보니까 더 정확하죠."

---

## 4. 이론 2: Zobrist Hashing & Transposition Table (15분)

### [00:25-00:28] 보드 상태 식별 문제

"이제 두 번째 주제입니다. 여러분, 다음 상황을 봅시다."

*화면에 그림 표시:*

```
초기 보드
   ↙        ↘
수 A: (1,1)→(1,2)    수 B: (2,1)→(3,1)
   ↘        ↙
     같은 보드!
```

"두 가지 다른 경로로 같은 보드에 도달했습니다. Alpha-Beta는 어떻게 동작할까요?"

🎯 **학생 질문:** "둘 다 탐색하니까... 중복 계산이네요!"

"맞습니다! 이것을 **Transposition(치환)**이라고 합니다. 체스에서는 이런 경우가 매우 빈번합니다."

### [00:28-00:32] Zobrist Hashing 원리

"같은 보드를 다시 탐색하지 않으려면, 이전 결과를 저장해야 합니다. 어떻게 저장할까요?"

🎯 **학생 질문:** "딕셔너리요! 보드를 키로 사용하면..."

"맞는데, 문제가 있습니다. ATAXX 7×7 보드를 문자열로 만들면?"

```python
board_str = "1120000201100002..."  # 49글자
```

"메모리 낭비가 심하고, 비교가 느립니다 (O(49)). 더 나은 방법은 **해싱**입니다!"

*화면에 Zobrist Hashing 설명:*

```python
# 1. 초기화: 각 (위치, 돌) 조합에 랜덤 64비트 정수 할당
zobrist = {}
for x in range(7):
    for y in range(7):
        for piece in [EMPTY, BLACK, WHITE, WALL]:
            zobrist[(x, y, piece)] = random.getrandbits(64)

# 2. 보드의 해시값 계산
def compute_hash(board):
    h = 0
    for x in range(7):
        for y in range(7):
            piece = board[x][y]
            h ^= zobrist[(x, y, piece)]  # XOR 연산
    return h
```

"XOR 연산의 특징은 무엇일까요?"

🎯 **학생 질문:** "A XOR A = 0이요."

"맞습니다! 그래서 증분 업데이트가 가능합니다."

### [00:32-00:35] 증분 업데이트

```python
# 돌을 (x, y)에서 이동
def update_hash(hash_value, x, y, old_piece, new_piece):
    hash_value ^= zobrist[(x, y, old_piece)]  # 이전 상태 제거
    hash_value ^= zobrist[(x, y, new_piece)]  # 새 상태 추가
    return hash_value
```

"시간 복잡도는?"

🎯 **학생 질문:** "O(1)이요!"

"정확합니다! 전체 보드 해싱은 O(49)지만, 업데이트는 O(1)입니다."

### [00:35-00:40] Transposition Table 구조

"이제 해시값을 키로 사용하여 결과를 저장합니다. 이것을 **Transposition Table(TT)**이라고 합니다."

*화면에 TT 구조 표시:*

```python
class TTEntry:
    def __init__(self, best_move, flag, depth, value):
        self.best_move = best_move  # 최선의 수
        self.flag = flag            # PV_NODE, CUT_NODE, ALL_NODE
        self.depth = depth          # 탐색 깊이
        self.value = value          # 평가값

# 전역 TT
tt = {}  # {hash: TTEntry}
```

"Flag는 무엇일까요?"

*화면에 Flag 설명:*

```
PV_NODE (Exact Value):
- alpha < value < beta
- 정확한 값
- 가장 신뢰도 높음

CUT_NODE (Lower Bound):
- value >= beta
- Beta cutoff 발생
- 실제 값은 이 값 이상

ALL_NODE (Upper Bound):
- value <= alpha
- Alpha cutoff 발생
- 실제 값은 이 값 이하
```

"이 정보를 어떻게 사용할까요? 두 가지 방법이 있습니다."

1. **Move Ordering**: TT의 best_move를 먼저 탐색
2. **TT Cutoff**: 저장된 값을 재탐색 없이 사용

"먼저 Move Ordering부터 봅시다."

---

## 5. 실습 2: TT + Move Ordering 구현 (15분)

### [00:40-00:48] Move Ordering의 중요성

"Alpha-Beta의 성능은 Move Ordering에 달려있습니다."

*화면에 표 표시:*

```
순서 품질        탐색 노드 수
────────────────────────────
완벽한 순서      b^(d/2)      (최선)
무작위 순서      b^(3d/4)     (평균)
최악의 순서      b^d          (최악)

예시 (b=35, d=6):
완벽: 42,875 노드
무작위: 253,000 노드 (6배 차이!)
```

"Iterative Deepening + TT를 결합하면 최고의 Move Ordering을 얻을 수 있습니다!"

**메커니즘:**
1. Depth=5 탐색: 최선의 수는 e2e3
2. TT에 저장: `tt[hash] = TTEntry(best_move="e2e3", ...)`
3. Depth=6 탐색: TT에서 "e2e3를 먼저 탐색하라" 힌트 획득
4. e2e3를 첫 번째로 탐색 → 높은 확률로 여전히 최선 → Alpha 상승 → 가지치기 증가!

### [00:48-00:55] 코드 구현

"자, 이제 구현합니다."

*실습 파일: `tt_agent.py`*

```python
import time

# Zobrist 초기화
def xorshift64(x):
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7)
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x

def init_zobrist():
    zobrist = {}
    seed = 123456789
    for x in range(7):
        for y in range(7):
            for piece in range(4):  # EMPTY, BLACK, WHITE, WALL
                seed = xorshift64(seed)
                zobrist[(x, y, piece)] = seed
    return zobrist

ZOBRIST = init_zobrist()

# TT Entry
class TTEntry:
    def __init__(self, best_move, flag, depth, value):
        self.best_move = best_move
        self.flag = flag
        self.depth = depth
        self.value = value

PV_NODE, CUT_NODE, ALL_NODE = 0, 1, 2

# 전역 TT
tt = {}

class ATAXXBoard:
    def __init__(self):
        # ... 기존 코드 ...
        self.hash_value = self.compute_hash()

    def compute_hash(self):
        h = 0
        for x in range(7):
            for y in range(7):
                piece = self.board[x][y]
                h ^= ZOBRIST[(x, y, piece)]
        return h

    def hash(self):
        return self.hash_value

    def make_move(self, move):
        # ... 이동 처리 ...
        # 해시 증분 업데이트
        self.hash_value ^= ZOBRIST[(x, y, old_piece)]
        self.hash_value ^= ZOBRIST[(x, y, new_piece)]

    def undo_move(self):
        # ... Undo 처리 ...
        # 해시 복원 (XOR로 되돌리기)
        self.hash_value ^= ZOBRIST[(x, y, new_piece)]
        self.hash_value ^= ZOBRIST[(x, y, old_piece)]

def alpha_beta_tt(board, depth, alpha, beta):
    alpha_original = alpha

    # TT 조회
    board_hash = board.hash()
    tt_move = None
    if board_hash in tt:
        entry = tt[board_hash]
        tt_move = entry.best_move

    # 종료 조건
    if depth == 0 or board.is_terminal():
        return board.evaluate(), None

    # 수 생성
    moves = board.legal_moves()

    # Move Ordering: TT move를 맨 앞으로
    if tt_move and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)

    # 탐색
    best_value = -float('inf')
    best_move = None

    for move in moves:
        board.make_move(move)
        value, _ = alpha_beta_tt(board, depth-1, -beta, -alpha)
        value = -value
        board.undo_move()

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    # TT 저장
    if best_value <= alpha_original:
        flag = ALL_NODE
    elif best_value >= beta:
        flag = CUT_NODE
    else:
        flag = PV_NODE

    tt[board_hash] = TTEntry(best_move, flag, depth, best_value)

    return best_value, best_move

def iterative_deepening_tt(board, time_limit_ms):
    start_time = time.time()
    best_move = None

    for depth in range(1, 50):
        elapsed = (time.time() - start_time) * 1000
        if elapsed > time_limit_ms * 0.9:
            break

        score, move = alpha_beta_tt(board, depth, -float('inf'), float('inf'))
        if move:
            best_move = move

    return best_move
```

"작성해보세요. 8분 드립니다."

*학생들 작성 시간*

### [00:55] 실행 및 성능 비교

"실행해봅시다."

```bash
python tt_agent.py
```

"TT가 없는 버전과 비교해보세요. 같은 시간에 얼마나 깊이 탐색하나요?"

🎯 **학생 답변:** "1~2 depth 더 깊이 탐색해요!"

"정확합니다! TT Move Ordering만으로 **+50 Elo** 향상입니다!"

---

## 6. 이론 3: TT Cutoff (10분)

### [00:55-00:58] TT Cutoff의 아이디어

"TT를 더 적극적으로 사용할 수 있습니다. 저장된 값을 탐색 없이 바로 사용하는 거죠."

*화면에 조건 표시:*

```python
if board_hash in tt:
    entry = tt[board_hash]

    # 조건: 저장된 깊이 >= 현재 필요한 깊이
    if entry.depth >= depth:

        # Flag에 따라 cutoff
        if entry.flag == PV_NODE:
            # 정확한 값 → 바로 반환
            return entry.value, entry.best_move

        elif entry.flag == CUT_NODE:
            # Lower bound: value >= entry.value
            if entry.value >= beta:
                return entry.value, entry.best_move

        elif entry.flag == ALL_NODE:
            # Upper bound: value <= entry.value
            if entry.value <= alpha:
                return entry.value, entry.best_move
```

### [00:58-01:02] 깊이 조건의 중요성

"왜 저장된 깊이가 더 깊어야 할까요?"

*화면에 예시:*

```
이전 탐색: depth=3, value=+10
현재 탐색: depth=5

문제:
- depth=3는 3수만 본 결과
- depth=5는 5수를 봐야 함
- 얕은 결과를 사용하면 전략적 실수!
```

🎯 **학생 질문:** "그럼 항상 깊게 저장해야 하네요?"

"맞습니다. Depth-preferred replace 정책을 사용합니다."

```python
if board_hash not in tt or tt[board_hash].depth <= depth:
    tt[board_hash] = TTEntry(...)
```

### [01:02-01:05] 실전 성능 - 의외의 결과

"TT Cutoff는 얼마나 빠를까요?"

*화면에 실험 결과 표시:*

```
베이스라인: ID + AB + TT Move Ordering
개선: 위에 + TT Cutoff

결과: ±0 Elo (변화 없음!)
```

"왜 그럴까요?"

🎯 **학생 질문:** "오히려 느려진 건가요?"

"정확합니다! ATAXX에서는 다음 이유로 효과가 없습니다:"

1. **얕은 탐색:** depth=4~6에서는 탐색 자체가 빠름
2. **TT 오버헤드:** 해시 계산, 딕셔너리 접근 비용
3. **Iterative Deepening:** 매번 depth 증가 → cutoff 기회 적음

"하지만 깊은 탐색(depth>10)이나 체스 같은 게임에서는 효과적입니다!"

---

## 7. 이론 4: Principal Variation Search (10분)

### [01:05-01:08] PVS의 핵심 통찰

"마지막 주제입니다. 세 번째 최적화 기법인 **Principal Variation Search (PVS)**입니다."

"관찰: Move Ordering이 좋으면, 첫 번째 자식이 최선일 확률이 높습니다."

*화면에 표시:*

```
보드
 ↓ 수들 (Move Ordering 후)
[A, B, C, D, E]
 ↑
TT의 best move

확률:
- A가 최선: 80%
- B가 최선: 15%
- C가 최선: 4%
- D가 최선: 1%
- E가 최선: 0%
```

"그렇다면 나머지 수들(B, C, D, E)을 빠르게 확인할 방법은 없을까요?"

### [01:08-01:12] Null Window의 마법

"**Null Window** 개념을 소개합니다."

*화면에 비교:*

```python
# Full window: 정밀 측정 (느림)
value = alpha_beta(node, depth, alpha, beta)
# "이 수의 정확한 값은 얼마인가?"

# Null window: Yes/No 확인 (빠름)
value = alpha_beta(node, depth, alpha, alpha+1)
# "이 수가 alpha보다 나은가?" (Yes or No)
```

"Null window는 왜 빠를까요?"

🎯 **학생 질문:** "Window가 좁으니까 가지치기가 자주 일어나요!"

"정확합니다! alpha=10, beta=11이면, 11 이상만 나와도 즉시 cutoff입니다."

### [01:12-01:15] PVS 알고리즘

*화면에 의사코드:*

```python
def pvs(board, depth, alpha, beta):
    if depth == 0 or board.is_terminal():
        return evaluate(board), None

    moves = order_moves(board)  # TT 활용
    best_value = -INF
    best_move = None

    for i, move in enumerate(moves):
        board.make_move(move)

        if i == 0:
            # 첫 번째 자식: Full window
            value, _ = pvs(board, depth-1, -beta, -alpha)
            value = -value
        else:
            # 나머지: Null window
            value, _ = pvs(board, depth-1, -alpha-1, -alpha)
            value = -value

            # Null window 실패 → 재탐색
            if alpha < value < beta:
                value, _ = pvs(board, depth-1, -beta, -value)
                value = -value

        board.undo_move()

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    return best_value, best_move
```

"핵심은 세 가지입니다:"

1. **첫 수:** Full window (정확한 값 필요)
2. **나머지:** Null window (빠른 확인)
3. **재탐색:** Null window 실패 시 Full window로 재탐색

---

## 8. 실습 3: PVS 구현 + 누적 성능 비교 (10분)

### [01:15-01:22] PVS 구현

"마지막 실습입니다. PVS를 추가합니다."

*실습 파일: `pvs_agent.py`*

```python
def pvs(board, depth, alpha, beta, use_null_window):
    alpha_original = alpha

    # TT 조회
    board_hash = board.hash()
    tt_move = None
    if board_hash in tt:
        entry = tt[board_hash]
        tt_move = entry.best_move

    # 종료 조건
    if depth == 0 or board.is_terminal():
        return board.evaluate(), None

    # Move Ordering
    moves = board.legal_moves()
    if tt_move and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)

    # 탐색
    best_value = -float('inf')
    best_move = None

    for i, move in enumerate(moves):
        board.make_move(move)

        if i == 0:
            # 첫 번째: Full window
            value, _ = pvs(board, depth-1, -beta, -alpha, False)
            value = -value
        else:
            # Null window
            value, _ = pvs(board, depth-1, -alpha-1, -alpha, True)
            value = -value

            # 재탐색
            if alpha < value < beta:
                value, _ = pvs(board, depth-1, -beta, -value, False)
                value = -value

        board.undo_move()

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break

    # TT 저장
    if best_value <= alpha_original:
        flag = ALL_NODE
    elif best_value >= beta:
        flag = CUT_NODE
    else:
        flag = PV_NODE

    tt[board_hash] = TTEntry(best_move, flag, depth, best_value)

    return best_value, best_move
```

"작성해보세요. 7분 드립니다."

*학생들 작성 시간*

### [01:22-01:25] 성능 측정 및 비교

"완성된 분들은 성능을 측정해봅시다."

```python
# 성능 측정 스크립트
def benchmark():
    board = ATAXXBoard()
    versions = [
        ("Alpha-Beta", alpha_beta),
        ("+ ID + TT MO", alpha_beta_tt),
        ("+ PVS", pvs),
    ]

    time_limit = 1000  # 1초

    for name, func in versions:
        start = time.time()
        for depth in range(1, 20):
            score, move = func(board, depth, -INF, INF)
            if (time.time() - start) * 1000 > time_limit:
                print(f"{name}: max depth = {depth-1}")
                break
```

*화면에 결과 표시:*

```
Alpha-Beta: max depth = 6
+ ID + TT MO: max depth = 7
+ PVS: max depth = 8
```

"같은 시간에 2 depth 더 깊이 탐색합니다! 이것이 **+100 Elo** 향상입니다!"

---

## 9. 정리 (5분)

### [01:25-01:28] 누적 성능 비교표

*화면에 최종 표 표시:*

```
기법                    누적 Elo    단계 변화    효과
─────────────────────────────────────────────────────
Minimax (Week 1)        0           -            기준
Alpha-Beta (Week 2)     +200        +200         핵심 가지치기
+ Iterative Deepening   +250        +50          시간 관리
+ TT Move Ordering      +300        +50          가지치기 극대화
+ TT Cutoff             +300        ±0           ATAXX에서는 효과 미미
+ PVS                   +350        +50          Null window 최적화
─────────────────────────────────────────────────────
총 향상: +350 Elo (약 62% 승률)
```

"Week 3만으로 +150 Elo 향상했습니다!"

### [01:28-01:30] 다음 주 예고

"다음 주는 완전히 새로운 알고리즘을 배웁니다: **Monte Carlo Tree Search (MCTS)**"

*화면에 표시:*

```
Alpha-Beta의 한계:
- 분기 인수가 높은 게임 (바둑 b~250)
- 평가 함수가 부정확한 게임
- 깊이 제한으로 인한 손실

MCTS의 등장:
- 2006년, 바둑 AI의 혁명
- AlphaGo의 핵심
- 평가 함수 없이도 강력!

Week 4 내용:
1. Selection, Expansion, Simulation, Backpropagation
2. UCB1 알고리즘
3. ATAXX에 MCTS 적용
4. Alpha-Beta vs MCTS 비교
```

"준비물: 오늘 배운 TT는 MCTS에서도 사용합니다. 복습해오세요!"

### [01:30] 과제 안내

"과제는 두 가지입니다:"

**1. Baekjoon 문제 8개**
- Week 3 폴더의 baekjoon/ 참고
- 게임 이론 + 탐색 최적화 문제

**2. ALPHANO 제출**
- `alphano/id_tt_pvs_agent.py` 완성
- 오늘 배운 모든 기법 통합
- 리더보드에 제출

"질문 있나요?"

🎯 **학생 질문들:**

Q: "TT 크기는 얼마로 설정하나요?"
A: "ALPHANO는 메모리 제한이 크지 않으니 100만~1000만 엔트리면 충분합니다."

Q: "PVS 재탐색이 너무 많이 발생하면요?"
A: "Move Ordering을 개선하세요. TT를 제대로 사용하면 재탐색은 5% 미만입니다."

Q: "Zobrist 해시 충돌은 어떻게 처리하나요?"
A: "수백만 상태 정도에서는 충돌 확률이 극히 낮아서 무시해도 됩니다. 완벽을 원하면 실제 보드도 비교하세요."

"수고하셨습니다! 다음 주에 봅시다!"

---

## 수업 후 체크리스트 (강사용)

- [ ] 학생들 과제 제출 여부 확인
- [ ] ALPHANO 리더보드 모니터링
- [ ] 다음 주 MCTS 자료 준비
- [ ] 질문이 많았던 부분 정리 (다음 수업 때 재설명)

---

## 부록: 자주 묻는 질문 (FAQ)

**Q1: Iterative Deepening에서 이전 깊이의 TT를 다음 깊이에서 사용할 수 있나요?**
A: 네! 그것이 핵심입니다. TT는 계속 누적되며, 이전 깊이의 best_move가 다음 깊이에서 Move Ordering에 사용됩니다.

**Q2: TT 크기 제한은 어떻게 하나요?**
A: `len(tt) > MAX_SIZE`일 때 `tt.clear()` 또는 LRU 정책으로 오래된 엔트리를 삭제합니다.

**Q3: PVS에서 재탐색이 계속 실패하면?**
A: Move Ordering이 나쁜 것입니다. TT가 제대로 작동하는지 확인하세요.

**Q4: Zobrist 해시가 음수가 나올 수 있나요?**
A: Python의 정수는 임의 크기이므로, `& 0xFFFFFFFFFFFFFFFF`로 64비트로 제한하세요.

**Q5: ATAXX 말고 다른 게임에서도 같은 기법을 사용할 수 있나요?**
A: 네! Iterative Deepening, TT, PVS는 체스, 오델로, 장기 등 모든 2인 완전정보 게임에 적용 가능합니다.

---

**수업 대본 끝**
