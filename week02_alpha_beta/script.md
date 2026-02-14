# Week 2 수업 대본: Alpha-Beta Pruning & 에이전트 성능 검증

**수업 시간**: 90분
**대상**: 고등학생
**준비물**: 노트북, Python 환경, Week 1 코드

---

## 도입 (5분)

안녕하세요, 여러분! 오늘은 Week 2 수업입니다. 지난 주에 Minimax와 Negamax 알고리즘을 배웠는데, 기억나시나요?

🎯 **질문**: "지난주에 만든 Negamax 에이전트를 실행해봤는데, depth를 5로 설정하면 어떻게 되던가요?"

→ 학생 답변 유도: "너무 느려요!", "시간이 오래 걸려요!"

맞습니다. Minimax/Negamax의 가장 큰 문제는 **속도**입니다. 모든 가능한 수를 다 탐색하다 보니, 깊이가 조금만 깊어져도 시간이 기하급수적으로 늘어나죠.

**오늘의 목표**:
1. **Alpha-Beta Pruning**: 불필요한 탐색을 줄여서 5~10배 빠르게!
2. **성능 검증**: 내 AI가 정말 강해졌는지 통계적으로 증명하기
3. **평가 함수 개선**: 더 똑똑한 AI 만들기

자, 시작해볼까요?

---

## 이론 1: Alpha-Beta Pruning 원리 (20분)

### 1.1 Minimax의 문제점 (3분)

지난주에 배운 Minimax/Negamax는 **완전 탐색 (Complete Search)** 알고리즘이었습니다.

```
시간 복잡도: O(b^d)
- b = branching factor (각 노드에서 가능한 수의 개수)
- d = depth (탐색 깊이)
```

ATAXX 게임에서 실제로 계산해보면:
- 초반 평균 분기 계수: b ≈ 50
- depth=3: 50^3 = 125,000개 노드
- depth=4: 50^4 = 6,250,000개 노드
- depth=5: 50^5 = 312,500,000개 노드

🎯 **질문**: "depth가 1 증가하면 탐색 노드가 몇 배 늘어날까요?"

→ 답: 약 50배!

이게 문제입니다. depth=4까지는 괜찮은데, depth=5부터는 너무 느려서 실시간 게임에 쓸 수가 없어요.

**그런데 잠깐!** 정말 모든 노드를 다 봐야 할까요? 🤔

### 1.2 불필요한 탐색 찾기 (5분)

간단한 예시를 볼게요. 여러분이 MAX 플레이어라고 생각해보세요.

```
        MAX (당신)
       /    \
      A      B
     /      /
   MIN    MIN
   / \    / \
  3   5  1   ?
```

**상황 설명**:

1. 왼쪽 가지 A를 먼저 탐색했습니다.
   - MIN이 선택: min(3, 5) = 3
   - MAX는 A를 선택하면 3점을 보장받음

2. 오른쪽 가지 B를 탐색합니다.
   - MIN의 첫 번째 자식: 1
   - MIN은 작은 값을 선택하므로, 1 또는 그보다 작은 값 선택

🎯 **질문**: "B의 두 번째 자식(?)을 봐야 할까요? 설령 ?가 100이라도 MIN은 뭘 선택할까요?"

→ 답: MIN은 1을 선택할 것! (작은 값 선택이 목표)

그렇다면 MAX 입장에서:
- A를 선택하면 3점
- B를 선택하면 최대 1점

→ **B는 절대 선택하지 않을 것!**
→ **B의 나머지 자식들을 볼 필요 없음!** ✂️

이게 바로 **가지치기 (Pruning)** 입니다!

### 1.3 Alpha와 Beta (7분)

가지치기를 체계적으로 하려면 두 개의 값을 추적해야 합니다:

**Alpha (α)**:
- **MAX 플레이어가 보장받은 최소 점수**
- "지금까지 찾은 최선의 수"
- MAX는 적어도 α 이상의 점수를 얻을 수 있음

**Beta (β)**:
- **MIN 플레이어가 허용할 최대 점수**
- "상대방이 나에게 줄 최대 점수"
- MIN은 β 이하로 점수를 제한함

**Cutoff 조건**:
```
α ≥ β  →  가지치기!
```

왜냐하면:
- MAX가 보장받은 점수(α) ≥ MIN이 허용할 점수(β)
- 이 가지는 절대 선택되지 않을 것
- 더 탐색할 필요 없음!

**구체적인 예시**:

```
                MAX (α=-∞, β=+∞)
              /                  \
           MIN                   MIN
       (α=-∞, β=+∞)          (α=3, β=+∞)
        /    |    \              / | \
      3     14    5            2  ?  ?
```

**단계별 설명**:

**Step 1**: 왼쪽 MIN 탐색
- 자식들: 3, 14, 5
- MIN 선택: 3
- **ROOT의 α = 3으로 업데이트**
- "나는 최소한 3점은 보장받았어!"

**Step 2**: 오른쪽 MIN 탐색 시작
- ROOT에서 전달받음: α=3, β=+∞
- 첫 번째 자식: 2
- MIN은 2를 선택할 가능성이 높음
- **이 MIN의 β = 2로 업데이트**

**Step 3**: Cutoff 판단
- α = 3, β = 2
- **α(3) ≥ β(2)** → **Cutoff!** ✂️
- 나머지 ?, ?를 탐색하지 않음

**왜 Cutoff인가?**:
- ROOT는 이미 왼쪽에서 3점을 보장받음
- 오른쪽 MIN은 최대 2점을 줄 것
- 2 < 3이므로 ROOT는 오른쪽을 선택하지 않음
- → 오른쪽의 나머지는 의미 없음!

### 1.4 더 복잡한 예시 (5분)

좀 더 현실적인 게임 트리를 봅시다.

화이트보드에 그리며 설명:

```
                    MAX (나)
                  /  |  \  \
                 /   |   \  \
               MIN  MIN MIN MIN
               /|\  /|\  |   |
              3 5 2 1 8 7 6  9
```

**탐색 과정** (왼쪽부터):

1. **첫 번째 MIN**: 3, 5, 2 → min = 2
   - α = 2

2. **두 번째 MIN**:
   - 첫 자식: 1
   - β = 1 (이 MIN은 최대 1을 반환)
   - α(2) ≥ β(1) → **Cutoff!**
   - 8, 7 탐색 안 함 ✂️

3. **세 번째 MIN**:
   - 첫 자식: 6
   - β = 6
   - α(2) < β(6) → 계속 탐색 필요
   - 하지만 다음 자식이 없으면 6 반환
   - α = max(2, 6) = 6으로 업데이트

4. **네 번째 MIN**:
   - 첫 자식: 9
   - β = 9
   - α(6) < β(9) → 계속 (혹은 자식이 없으면 9 반환)

**탐색 노드 수**:
- Minimax: 9개 (모든 리프 노드)
- Alpha-Beta: 6개 (3+1+1+1, 3개 절약!)

실제 게임에서는 훨씬 더 많이 절약됩니다!

---

## 실습 1: Alpha-Beta 에이전트 구현 (15분)

자, 이제 직접 구현해봅시다!

### 1.1 코드 구조 (3분)

지난주 Negamax 코드를 기억하시나요?

```python
def negamax(board, depth, player):
    if depth == 0 or game_over(board):
        return evaluate(board, player)

    max_score = -INF
    for move in get_possible_moves(board, player):
        new_board = make_move(board, move, player)
        score = -negamax(new_board, depth - 1, opponent(player))
        max_score = max(max_score, score)

    return max_score
```

Alpha-Beta를 추가하면:

```python
def negamax_alpha_beta(board, depth, alpha, beta, player):
    if depth == 0 or game_over(board):
        return evaluate(board, player), None

    moves = get_possible_moves(board, player)
    if not moves:
        # 게임 종료 처리
        return evaluate_terminal(board, player), None

    best_move = None

    for move in moves:
        new_board = make_move(board, move, player)
        score = -negamax_alpha_beta(new_board, depth - 1, -beta, -alpha, opponent(player))[0]

        if score > alpha:
            alpha = score
            best_move = move

        if alpha >= beta:
            break  # Cutoff!

    return alpha, best_move
```

### 1.2 핵심 변경 사항 (5분)

**1. 파라미터 추가**:
```python
alpha, beta  # 추가된 파라미터
```

**2. 재귀 호출 시 부호 반전**:
```python
score = -negamax_alpha_beta(new_board, depth - 1, -beta, -alpha, opponent(player))[0]
```

🎯 **질문**: "왜 -beta, -alpha 순서로 전달할까요?"

→ 답: Negamax는 항상 현재 플레이어 입장에서 최대화합니다. 상대방 입장에서는 우리의 alpha가 그들의 -beta가 되고, 우리의 beta가 그들의 -alpha가 됩니다!

**3. Alpha 업데이트**:
```python
if score > alpha:
    alpha = score
    best_move = move
```

**4. Beta Cutoff**:
```python
if alpha >= beta:
    break  # 나머지 move 탐색 안 함!
```

### 1.3 실제 실행 (7분)

자, 이제 실행해봅시다!

```bash
# Week 1 Minimax vs Week 2 Alpha-Beta (같은 depth=3)
python game_arena.py --agent1 minimax --agent2 alphabeta --depth 3 --games 10
```

**예상 결과**:
```
=== Game Arena ===
Agent 1: Minimax (depth=3)
Agent 2: Alpha-Beta (depth=3)

Game 1: Alpha-Beta wins
Game 2: Minimax wins
Game 3: Alpha-Beta wins
...

Results:
Minimax: 5 wins
Alpha-Beta: 5 wins
```

🎯 **관찰**: "승률이 비슷하죠? 왜 그럴까요?"

→ 답: Alpha-Beta는 Minimax와 **정확히 같은 결과**를 보장합니다! 단지 더 빠를 뿐!

**노드 수 비교**:
```python
# 노드 카운터 추가
nodes_minimax = 125834
nodes_alphabeta = 24167
reduction = (1 - 24167/125834) * 100  # 80.8% 감소!

print(f"감소율: {reduction:.1f}%")
```

**와! 80% 이상 줄었네요!** 🎉

---

## 이론 2: SPRT & Elo Rating (15분)

### 2.1 성능 비교의 문제 (3분)

자, 이제 질문을 하나 해볼게요.

🎯 **질문**: "우리가 Alpha-Beta로 depth=4까지 탐색할 수 있게 되었어요. 이게 depth=3인 Minimax보다 정말 강할까요?"

→ 학생 답변 유도: "당연히 강하죠!", "더 깊게 보니까요!"

그렇죠. 직관적으로는 그렇습니다. **하지만 증명할 수 있나요?**

**단순 비교**:
- 10판 대전 → 6승 4패 → 강한가? (운일 수도...)
- 100판 대전 → 55승 45패 → 강한가?

**문제**:
1. 몇 판을 해야 확신할 수 있는가?
2. 차이가 통계적으로 유의미한가?
3. 시간이 너무 오래 걸리지 않는가?

→ **SPRT (Sequential Probability Ratio Test)** 가 답입니다!

### 2.2 SPRT 개념 (5분)

SPRT는 1940년대에 개발된 통계 기법으로, **최소한의 샘플**로 결론을 내릴 수 있습니다.

**핵심 아이디어**:
- 고정된 게임 수가 아님
- 차이가 명확하면 빨리 종료
- 차이가 없으면 빨리 종료
- **필요한 만큼만** 게임 진행

**가설 설정**:

**H0 (귀무 가설)**: 두 에이전트의 실력이 같다
- Elo 차이 = 0
- 승률 = 50%

**H1 (대립 가설)**: Agent A가 더 강하다
- Elo 차이 ≥ 50 (예시)
- 승률 ≥ 57.15%

**LLR (Log-Likelihood Ratio)**:
```
LLR = Σ log(P(결과|H1) / P(결과|H0))

승리 시: LLR += log(p1 / p0)
패배 시: LLR += log((1-p1) / (1-p0))
```

여기서:
- p0 = 0.5 (H0 하에서 승률)
- p1 = 0.5715 (H1 하에서 승률, Elo +50)

**결정 경계**:
```
A = log(19) ≈ 2.944
B = log(1/19) ≈ -2.944
```

**의사결정**:
- LLR ≥ A → **H1 채택** (A가 강함!)
- LLR ≤ B → **H0 채택** (차이 없음)
- B < LLR < A → **계속 게임**

**시각화** (화이트보드):
```
LLR
 3 +........................ H1 (A가 강함!)
 2 +           /
 1 +      /   /
 0 +-----/---/--------
-1 +    /
-2 +   /
-3 +........................ H0 (차이 없음)
   0  10  20  30  40  게임 수
```

### 2.3 Elo Rating (7분)

**Elo Rating**은 체스에서 시작된 실력 평가 시스템입니다.

**기대 승률 공식**:
```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

**예시**:

| R_A | R_B | Elo 차이 | 승률 |
|-----|-----|---------|------|
| 1500 | 1500 | 0 | 50.0% |
| 1550 | 1500 | +50 | 57.2% |
| 1600 | 1500 | +100 | 64.0% |
| 1700 | 1500 | +200 | 76.0% |

🎯 **퀴즈**: "Elo가 200 높으면 대략 몇 % 승률일까요?"

→ 답: 약 75~76%

**Rating 업데이트**:
```
R'_A = R_A + K * (S_A - E_A)

K = 32 (변동 폭)
S_A = 실제 결과 (승=1, 무=0.5, 패=0)
E_A = 기대 승률
```

**예시**:

**약자가 강자를 이김**:
```
R_A = 1500, R_B = 1700 (200점 차이)
E_A = 0.24 (24% 승률 예상)
A 승리! S_A = 1

R'_A = 1500 + 32 * (1 - 0.24) = 1524.3 (+24.3!)
R'_B = 1700 + 32 * (0 - 0.76) = 1675.7 (-24.3)
```

**강자가 약자를 이김**:
```
R_A = 1700, R_B = 1500
E_A = 0.76 (76% 승률 예상)
A 승리! S_A = 1

R'_A = 1700 + 32 * (1 - 0.76) = 1707.7 (+7.7)
R'_B = 1500 + 32 * (0 - 0.24) = 1492.3 (-7.7)
```

→ **예상대로 이기면 조금만 오름, 업셋 승리하면 많이 오름!**

**AI 에이전트 Elo 예시**:
- Random: 1000
- Greedy: 1200
- Minimax (d=2): 1400
- Minimax (d=3): 1600
- Alpha-Beta (d=4): 1850
- Alpha-Beta (d=5): 2100

---

## 실습 2: SPRT 실험 (10분)

### 2.1 SPRT 코드 (5분)

간단한 SPRT 시뮬레이터를 만들어봅시다.

```python
import math

def sprt_test(agent_a, agent_b, elo0=50, alpha=0.05, beta=0.05, max_games=500):
    """SPRT를 이용한 에이전트 비교"""

    # 승률 계산
    p0 = 0.5
    p1 = 1 / (1 + 10**(-elo0 / 400))

    # 결정 경계
    A = math.log((1 - beta) / alpha)
    B = math.log(beta / (1 - alpha))

    print(f"p0={p0:.4f}, p1={p1:.4f}")
    print(f"Upper bound (A) = {A:.3f}")
    print(f"Lower bound (B) = {B:.3f}")
    print()

    llr = 0
    wins = 0
    losses = 0
    draws = 0

    for game_num in range(1, max_games + 1):
        # 게임 진행
        result = play_game(agent_a, agent_b)

        if result == 'A':
            wins += 1
            llr += math.log(p1 / p0)
        elif result == 'B':
            losses += 1
            llr += math.log((1 - p1) / (1 - p0))
        else:  # Draw
            draws += 1

        # 10판마다 출력
        if game_num % 10 == 0:
            print(f"Game {game_num}: W={wins} L={losses} D={draws}, LLR={llr:.3f}")

        # 의사결정
        if llr >= A:
            print(f"\n결론: H1 채택! Agent A가 통계적으로 유의미하게 강합니다.")
            print(f"총 {game_num}판 소요")
            return 'H1', game_num

        if llr <= B:
            print(f"\n결론: H0 채택! 두 에이전트의 실력 차이가 없습니다.")
            print(f"총 {game_num}판 소요")
            return 'H0', game_num

    print(f"\n{max_games}판 동안 결론 못 냄. 차이가 애매합니다.")
    return 'Undecided', max_games
```

### 2.2 실행 및 관찰 (5분)

```bash
# Alpha-Beta (depth=4) vs Minimax (depth=3) SPRT 테스트
python sprt_test.py --agent1 alphabeta --depth1 4 --agent2 minimax --depth2 3
```

**예상 출력**:
```
p0=0.5000, p1=0.5715
Upper bound (A) = 2.944
Lower bound (B) = -2.944

Game 10: W=6 L=4 D=0, LLR=0.563
Game 20: W=13 L=7 D=0, LLR=1.689
Game 30: W=20 L=10 D=0, LLR=2.815
Game 38: W=25 L=13 D=0, LLR=3.128

결론: H1 채택! Agent A가 통계적으로 유의미하게 강합니다.
총 38판 소요
```

🎯 **관찰**: "38판만에 결론이 났네요! 100판, 200판 할 필요 없이!"

**LLR 그래프 그리기**:
```python
# 실시간으로 LLR 변화 시각화
import matplotlib.pyplot as plt

# ... (게임 진행하며 LLR 기록)
plt.plot(llr_history)
plt.axhline(y=A, color='g', linestyle='--', label='H1 boundary')
plt.axhline(y=B, color='r', linestyle='--', label='H0 boundary')
plt.xlabel('Games')
plt.ylabel('LLR')
plt.legend()
plt.show()
```

---

## 이론 3: 평가 함수 개선 (10분)

### 3.1 기본 평가 함수의 한계 (3분)

지난 주 우리가 사용한 평가 함수를 기억하시나요?

```python
def evaluate(board, player):
    return count_pieces(board, player) - count_pieces(board, opponent)
```

**문제점**:
1. 단순히 돌 개수만 세음
2. **위치의 가치**를 무시
3. **이동 가능성 (Mobility)**를 무시
4. 게임 단계를 구분하지 않음

**예시 상황**:
```
보드 A: 내 돌 10개 (모두 코너)
보드 B: 내 돌 10개 (모두 가장자리)
```

🎯 **질문**: "기본 평가 함수는 둘을 어떻게 평가할까요?"

→ 답: 똑같이 0점! (10 - 0 = 0)

**하지만 실제로는?**
- 보드 A가 훨씬 유리! (코너는 안전, 감염 안 됨)
- 보드 B는 위험 (감염되기 쉬움)

### 3.2 개선 방향 (4분)

**1. 위치 가중치 (Positional Weights)**:

```python
POSITION_WEIGHTS = [
    [100,  -20,  10,   5,  10,  -20, 100],  # 코너 매우 중요
    [-20,  -40,  -5,  -5,  -5,  -40, -20],  # 코너 인접 위험
    [ 10,   -5,  10,   5,  10,   -5,  10],
    [  5,   -5,   5,   0,   5,   -5,   5],  # 중앙 중립
    [ 10,   -5,  10,   5,  10,   -5,  10],
    [-20,  -40,  -5,  -5,  -5,  -40, -20],
    [100,  -20,  10,   5,  10,  -20, 100],
]
```

**왜 이렇게?**
- **코너 (100)**: 절대 감염 안 됨, 매우 안전
- **코너 인접 (-40)**: 상대에게 코너를 내줄 위험
- **중앙 (0~10)**: 이동성은 좋지만 감염 위험
- **가장자리 (10~-20)**: 이동성 제한

**2. 이동성 (Mobility)**:

```python
def count_mobility(board, player):
    return len(get_possible_moves(board, player))
```

- 수가 많을수록 유리
- 상대의 수를 제한하는 것도 전략

**3. 게임 단계별 가중치**:

```python
total_pieces = my_pieces + opp_pieces

if total_pieces < 20:  # 초반
    weight_piece = 1
    weight_position = 3
    weight_mobility = 5  # 이동성이 가장 중요!
elif total_pieces < 35:  # 중반
    weight_piece = 2
    weight_position = 2
    weight_mobility = 3
else:  # 후반
    weight_piece = 5  # 돌 개수가 가장 중요!
    weight_position = 1
    weight_mobility = 1
```

### 3.3 종합 평가 함수 (3분)

```python
def evaluate_advanced(board, player):
    # 1. 게임 종료 체크
    if game_over(board):
        winner = get_winner(board)
        if winner == player:
            return 10000
        elif winner == opponent(player):
            return -10000
        else:
            return 0

    # 2. 돌 개수
    my_pieces = count_pieces(board, player)
    opp_pieces = count_pieces(board, opponent(player))
    piece_score = my_pieces - opp_pieces

    # 3. 위치 가치
    my_position = sum(POSITION_WEIGHTS[i][j]
                      for i in range(7) for j in range(7)
                      if board[i][j] == player)
    opp_position = sum(POSITION_WEIGHTS[i][j]
                       for i in range(7) for j in range(7)
                       if board[i][j] == opponent(player))
    position_score = my_position - opp_position

    # 4. 이동성
    mobility_score = count_mobility(board, player) - count_mobility(board, opponent(player))

    # 5. 가중치 적용
    total = my_pieces + opp_pieces
    if total < 20:
        w_p, w_pos, w_m = 1, 3, 5
    elif total < 35:
        w_p, w_pos, w_m = 2, 2, 3
    else:
        w_p, w_pos, w_m = 5, 1, 1

    return w_p * piece_score + w_pos * position_score + w_m * mobility_score
```

---

## 실습 3: 평가 함수 비교 (10분)

### 3.1 코드 수정 (3분)

`alpha_beta_agent.py`에서 평가 함수를 교체해봅시다.

```python
# 기존
def evaluate(board, player):
    return count_pieces(board, player) - count_pieces(board, opponent(player))

# 개선
def evaluate(board, player):
    return evaluate_advanced(board, player)
```

### 3.2 대전 실행 (7분)

```bash
# 기본 평가 함수 vs 개선 평가 함수 (같은 depth=4)
python game_arena.py --agent1 basic_eval --agent2 advanced_eval --depth 4 --games 50
```

**예상 결과**:
```
Game 1: advanced_eval wins
Game 2: advanced_eval wins
Game 3: basic_eval wins
...

Results after 50 games:
basic_eval: 14 wins (28.0%)
advanced_eval: 36 wins (72.0%)
```

🎉 **와! 승률이 28% → 72%로 올랐네요!**

**SPRT로 검증**:
```bash
python sprt_test.py --agent1 advanced_eval --agent2 basic_eval --depth 4
```

```
Game 10: W=7 L=3 D=0, LLR=1.126
Game 20: W=15 L=5 D=0, LLR=2.815
Game 26: W=19 L=7 D=0, LLR=3.128

결론: H1 채택! advanced_eval이 통계적으로 유의미하게 강합니다.
총 26판 소요
```

**Elo 추정**:
```python
# 승률 72% → Elo 차이 계산
# 0.72 = 1 / (1 + 10^(-elo_diff / 400))
# 풀면: elo_diff ≈ 152

print(f"추정 Elo 차이: +152")
```

→ **평가 함수 개선만으로 Elo +150!** 🚀

**왜 이렇게 효과적인가?**
- Alpha-Beta는 탐색 효율을 올림 (더 깊이)
- 평가 함수는 판단 품질을 올림 (더 똑똑하게)
- 둘의 조합 = 강력한 AI!

---

## 정리 및 다음 주 예고 (5분)

### 오늘 배운 내용

**1. Alpha-Beta Pruning**:
- α ≥ β일 때 가지치기
- 평균 60~80% 노드 감소
- **같은 결과, 5배 이상 빠름**
- depth를 더 깊게 탐색 가능

**2. SPRT (Sequential Probability Ratio Test)**:
- 최소한의 게임으로 통계적 검증
- LLR이 경계에 도달하면 종료
- 효율적인 성능 비교

**3. Elo Rating**:
- 상대 평가 시스템
- Elo 차이 200 ≈ 승률 75%
- AI 강도 측정에 유용

**4. 평가 함수 개선**:
- 위치 가중치 + 이동성
- 게임 단계별 전략
- 큰 성능 향상 (Elo +150)

🎯 **최종 질문**: "오늘 배운 것 중 가장 인상 깊었던 게 뭐였나요?"

→ 학생 답변 유도 및 정리

### 숙제

**1. ALPHANO 제출**:
- Alpha-Beta Pruning 구현
- depth=4 이상 권장
- 개선된 평가 함수 시도
- 순위 경쟁!

**2. Baekjoon 문제 9개**:
- 게임 이론 문제들
- 님 게임, 틱택토 등
- Alpha-Beta 활용

**3. 선택 과제**:
- SPRT 실험 보고서
- 평가 함수 개선 실험
- Elo Rating 분석

### 다음 주 예고: Monte Carlo Tree Search (MCTS)

**왜 MCTS인가?**
- Alpha-Beta는 평가 함수에 의존
- 복잡한 게임은 평가 함수 설계가 어려움
- MCTS는 **랜덤 시뮬레이션**으로 평가
- **AlphaGo의 핵심 알고리즘!**

**MCTS 4단계**:
1. Selection: 탐색할 노드 선택 (UCB1)
2. Expansion: 새 노드 추가
3. Simulation: 랜덤 플레이아웃
4. Backpropagation: 결과 업데이트

**준비물**:
- Alpha-Beta 코드 완성
- ALPHANO 제출
- 게임 이론 기초 복습

---

**다음 주에 만나요! 수고하셨습니다!** 🎉

---

## 보충 자료

### FAQ

**Q1**: "Alpha-Beta는 항상 Minimax보다 빠른가요?"
**A**: 대부분 그렇지만, 최악의 경우 (최악의 move ordering)에는 Minimax와 같을 수 있습니다. 하지만 실전에서는 거의 항상 훨씬 빠릅니다.

**Q2**: "SPRT는 항상 결론을 내나요?"
**A**: 아니요. 두 에이전트의 실력이 정확히 elo0/2 정도 차이나면 결론이 나지 않을 수 있습니다. 이 경우 max_games에 도달하면 종료합니다.

**Q3**: "평가 함수를 너무 복잡하게 만들면 느려지지 않나요?"
**A**: 맞습니다! 평가 함수는 매우 자주 호출되므로 효율성이 중요합니다. 복잡도와 정확도 사이의 균형이 필요합니다.

**Q4**: "Elo Rating은 어떻게 초기값을 정하나요?"
**A**: 보통 임의로 정합니다 (예: 1500). 충분히 많은 게임을 하면 실제 실력을 반영하는 값으로 수렴합니다.

### 디버깅 팁

**Alpha-Beta가 이상한 수를 둘 때**:
1. 평가 함수 확인 (부호가 맞는지)
2. Alpha-Beta 전달 순서 확인 (-beta, -alpha)
3. Cutoff 조건 확인 (alpha >= beta)
4. 로그 출력으로 탐색 과정 추적

**SPRT가 너무 오래 걸릴 때**:
1. elo0을 크게 설정 (50 → 100)
2. alpha, beta를 크게 설정 (0.05 → 0.1)
3. 실력 차이가 정말 애매할 수 있음

**평가 함수 튜닝**:
1. 가중치를 점진적으로 조정
2. SPRT로 각 변경 효과 측정
3. 너무 많은 요소를 한 번에 추가하지 말 것

### 참고 링크

- Chess Programming Wiki: https://www.chessprogramming.org/Alpha-Beta
- SPRT Calculator: https://www.stat.auckland.ac.nz/~wild/SPRT/
- Elo Rating System: https://en.wikipedia.org/wiki/Elo_rating_system

---

**수업 대본 끝**
