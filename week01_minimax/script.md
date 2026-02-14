# Week 1 강의 스크립트: 게임 에이전트 기초 (Random → Greedy → Minimax)

**강의 시간**: 90분
**대상**: 고등학생
**목표**: 게임 AI의 기본 개념을 이해하고 Random, Greedy, Minimax 에이전트를 직접 구현하여 대전시켜보기

---

## 도입 (10분)

안녕하세요! 오늘부터 10주 동안 함께 게임 AI를 공부할 건데요, 정말 재미있을 거예요.

🎯 **"혹시 게임 AI가 뭔지 알아요? 게임에서 컴퓨터가 어떻게 움직이는지 생각해본 적 있나요?"**

*(학생 답변 듣기)*

맞아요. 롤에서 봇이랑 싸우거나, 스타크래프트에서 컴퓨터랑 대전할 때 그게 다 AI예요. 그런데 이런 AI들이 어떻게 판단하는지 궁금하지 않았나요?

### AlphaGo 이야기

2016년에 엄청난 일이 있었어요. 구글 딥마인드가 만든 **알파고**라는 AI가 이세돌 9단을 이겼거든요. 바둑은 우주의 원자 수보다 경우의 수가 많은 게임인데, AI가 인간 최고수를 이긴 거죠.

근데 알파고가 처음부터 딥러닝으로만 만들어진 건 아니에요. 그 바탕에는 오늘 우리가 배울 **게임 트리 탐색**과 **평가 함수** 개념이 들어가 있어요.

### 오늘의 목표

오늘은 **세균 전쟁**이라는 간단한 게임에서 AI를 직접 만들어볼 거예요.

1. 먼저 **아무거나 두는 AI** (Random)
2. 그 다음 **1수만 생각하는 AI** (Greedy)
3. 마지막으로 **상대의 수까지 예측하는 AI** (Minimax)

이렇게 세 가지를 만들어서 서로 대전시켜 볼 건데요, 어떤 AI가 이길지 한번 예상해보세요.

🎯 **"어떤 AI가 제일 강할 것 같아요?"**

*(학생 답변 듣기)*

네, 맞아요. 더 많이 생각하는 AI가 이길 확률이 높겠죠? 그럼 지금부터 그걸 직접 확인해봅시다!

---

## 이론 1: 게임 트리 (15분)

AI가 게임을 어떻게 생각하는지 알려면, 먼저 **게임 트리**라는 개념을 이해해야 해요.

### 틱택토로 이해하기

*(칠판이나 화면에 틱택토 판 그리기)*

여러분 틱택토 알죠? 3×3 판에 O, X를 번갈아 놓는 게임이요.

```
. . .
. . .
. . .
```

첫 수에서 X가 둘 수 있는 곳은 몇 군데일까요?

🎯 **"대칭 고려하면 실질적으로 3가지죠. 가운데, 모서리, 변."**

자, X가 가운데를 두면:

```
. . .
. X .
. . .
```

이제 O 차례예요. O가 둘 수 있는 곳은? 8군데죠.

이렇게 **각 상태에서 다음 상태로 가는 모든 경우의 수**를 트리 구조로 그린 게 바로 **게임 트리**예요.

```
        [시작 상태]
       /    |    \
     [X중앙] [X모서리] [X변]
     / | | \
   [O의 응수들...]
```

### 게임 트리의 구성 요소

1. **상태 (State)**: 현재 판의 모습
2. **행동 (Action)**: 돌을 놓는 위치
3. **전이 (Transition)**: 행동으로 인해 상태가 바뀌는 것

🎯 **"틱택토에서 가능한 게임 상태가 몇 개나 될까요? 대충 짐작해보세요."**

*(학생 답변 듣기)*

정답은... 약 **255,168개**예요. 생각보다 많죠? 하지만 이건 그래도 작은 편이에요.

### 세균 전쟁(ATAXX) 게임 규칙

이제 우리가 오늘 다룰 **세균 전쟁** 게임을 소개할게요. 영어로는 **ATAXX**라고 불러요.

*(ALPHANO 사이트 https://alphano.co.kr/problem/1 열어서 보여주기)*

**게임 규칙:**
- **7×7 보드**
- 두 플레이어 (O, X)가 번갈아 이동
- 시작 위치: O는 (1,1)과 (7,7), X는 (1,7)과 (7,1) — 대각선 배치

```
  1 2 3 4 5 6 7
1 O . . . . . X
2 . . . . . . .
3 . . . . . . .
4 . . . . . . .
5 . . . . . . .
6 . . . . . . .
7 X . . . . . O
```

**이동 방식 두 가지:**

1. **분열 (거리 1)**: 인접 8방향에 복제. 원본이 그대로 남고 새 칸에도 생겨요!
```
. . .      . O .
. O .  →   O O .   ← 1개가 2개로!
. . .      . . .
```

2. **도약 (거리 2)**: 2칸 떨어진 곳으로 점프. 원본은 사라져요.
```
. . . . .      . . O . .
. . . . .      . . . . .
. . O . .  →   . . . . .   ← 이동
. . . . .      . . . . .
```

**감염**: 이동/복제 후 도착 칸 인접 8방향의 **적 말이 내 색으로 변환**!
```
이동 전:        이동 후 (O가 가운데로):
X X .          O O .
X . .    →     O O .   ← X 3개가 모두 O로!
. . .          . . .
```

이게 세균이 퍼지는 것처럼 보인다고 해서 "세균 전쟁"이라고 불러요. 분열하면 자기가 늘어나고, 주변 적까지 감염시키니까요!

**게임 종료 조건:**
- 한쪽 세균이 모두 소멸
- 빈 칸이 없어짐
- **400턴** 초과 (무승부)
- 돌이 많은 쪽이 승리!

🎯 **"이 게임의 상태 공간은 얼마나 클까요? 틱택토보다 클까요, 작을까요?"**

*(학생 생각할 시간 주기)*

7×7 = 49칸이고, 각 칸은 빈칸/O/X 3가지 상태가 가능해요. 그러니까 이론적으로는 **3^49 ≈ 약 2경** 정도의 상태가 있어요. 물론 실제로는 규칙 때문에 불가능한 상태도 많지만, 어쨌든 엄청 크죠.

그래서 **모든 경우의 수를 다 볼 수는 없어요.** 이게 바로 AI가 필요한 이유예요.

---

## 이론 2: 평가 함수 (5분)

게임 트리가 너무 크면 끝까지 볼 수 없잖아요? 그럼 AI는 어떻게 판단할까요?

### 평가 함수란?

**평가 함수 (Evaluation Function)** = 현재 상태가 나한테 얼마나 유리한지 숫자로 나타낸 것

예를 들어:
- 내가 이기는 상태: +1000
- 상대가 이기는 상태: -1000
- 비기는 상태: 0
- 게임 중간 상태: ???

게임 중간 상태를 어떻게 평가할까요?

### 가장 간단한 평가 함수

세균 전쟁에서 가장 단순한 평가 함수는:

```
평가값 = 내 돌의 개수 - 상대 돌의 개수
```

예를 들어:
- 내가 20개, 상대가 15개 → 평가값 = +5
- 내가 10개, 상대가 18개 → 평가값 = -8

🎯 **"이 평가 함수가 완벽할까요? 뭔가 놓치는 게 있을까요?"**

맞아요. 위치도 중요하고, 어떻게 배치되어 있는지도 중요하죠. 하지만 일단 이걸로 시작할 거예요. 나중에 더 좋은 평가 함수를 만들어볼 수도 있어요.

---

## 실습 1: Random Agent (10분)

자, 이제 첫 번째 AI를 만들어봅시다!

### 코드 설명

*(agents/random_agent.py 파일 열기)*

```python
import random

def random_agent(state, player_number):
    valid_moves = state.get_valid_moves()
    if valid_moves:
        return random.choice(valid_moves)
    return None
```

이게 전부예요. 정말 간단하죠?

**동작 원리:**
1. `get_valid_moves()`: 현재 놓을 수 있는 모든 위치를 가져옴
2. `random.choice()`: 그중에서 무작위로 하나 선택

🎯 **"이 에이전트가 왜 약할까요?"**

*(학생 답변 듣기)*

네, 맞아요. **아무 생각이 없어요.** 좋은 수인지, 나쁜 수인지 전혀 고려하지 않고 그냥 랜덤으로 두는 거죠.

### 실행해보기

한번 랜덤 에이전트끼리 대전시켜 볼까요?

```bash
cd /Users/simjoon/megastudy/RL_GAME_ALGORITHM/week01_minimax
python game.py random random --visualize
```

*(실행 결과 보기)*

보세요, 정말 아무렇게나 두죠? 가끔은 좋은 수를 두기도 하지만, 완전히 운에 맡기는 거예요.

### ALPHANO 제출

나중에 시간 있으면 이걸 ALPHANO 플랫폼에 제출해서 다른 사람들 AI랑 대전시켜볼 수도 있어요. 아마 꼴찌할 거예요. (웃음)

---

## 실습 2: Greedy Agent (10분)

이제 조금 더 똑똑한 AI를 만들어봅시다.

### Greedy 전략

**Greedy = 욕심쟁이 = 당장 눈앞의 이득만 챙기는 전략**

"1수만 보고 판단하기":
1. 모든 가능한 수를 시도해봄
2. 각 수를 뒀을 때 평가 함수 계산
3. 가장 점수가 높은 수를 선택

### 코드 작성

*(agents/greedy_agent.py 파일 열기)*

```python
def greedy_agent(state, player_number):
    valid_moves = state.get_valid_moves()
    if not valid_moves:
        return None

    best_move = None
    best_score = float('-inf')

    for move in valid_moves:
        # 이 수를 뒀을 때의 상태를 시뮬레이션
        next_state = state.clone()
        next_state.play_move(move, player_number)

        # 평가 함수 계산
        score = evaluate(next_state, player_number)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def evaluate(state, player_number):
    # 간단한 평가: 내 돌 수 - 상대 돌 수
    my_count = state.count_stones(player_number)
    opp_count = state.count_stones(3 - player_number)
    return my_count - opp_count
```

**핵심 아이디어:**
- `clone()`: 현재 상태를 복사 (실제로 두지 않고 시뮬레이션)
- `play_move()`: 복사본에서 수를 둬봄
- `evaluate()`: 그 결과가 얼마나 좋은지 평가
- 가장 좋은 수를 선택!

🎯 **"이 코드에서 for 루프가 몇 번 돌까요?"**

정답: 가능한 수의 개수만큼! 처음에는 36번, 게임이 진행되면 점점 줄어들겠죠.

### Random vs Greedy 대전

자, 이제 재미있는 실험을 해볼까요?

```bash
python game.py random greedy --visualize
```

*(실행하면서 설명)*

어때요? Greedy가 압도적으로 이기죠?

100번 대전시켜 볼게요:

```bash
python benchmark.py random greedy --games 100
```

결과가 어떻게 나왔나요? 아마 Greedy가 90% 이상 이길 거예요.

🎯 **"왜 이렇게 차이가 날까요?"**

Random은 아무 생각 없이 두는데, Greedy는 적어도 "이 수를 두면 내 돌이 몇 개 늘어난다"를 계산하거든요. 엄청난 차이죠.

---

## 이론 3: Minimax 알고리즘 (20분)

자, 이제 본격적인 게임 AI의 핵심으로 들어갑니다.

### Greedy의 문제점

🎯 **"Greedy 에이전트가 완벽할까요? 어떤 문제가 있을까요?"**

*(학생 생각할 시간 주기)*

맞아요. **상대가 어떻게 대응할지 생각하지 않아요.**

예를 들어 보죠:

```
현재 상태에서 A라는 수를 두면:
→ 당장 +3 이득

B라는 수를 두면:
→ 당장 +2 이득
→ 하지만 상대가 막지 못하면 다음 턴에 +10 이득!
```

Greedy는 A를 선택해요. 하지만 현명한 선택은 B일 수도 있죠.

### "상대도 최선을 다한다"는 가정

게임 AI의 핵심 가정:
> **상대도 나만큼 똑똑하고, 자기한테 최선인 수를 둔다.**

이걸 반영하려면 어떻게 해야 할까요?

**답: 상대의 수까지 내다봐야 한다!**

### Minimax의 핵심 아이디어

틱택토 예시로 설명할게요.

*(칠판에 그리면서)*

```
        [현재 상태]
       나의 차례 (MAX)
       /    |    \
    [A]   [B]   [C]  ← 내가 선택할 수 있는 수들
    상대 차례 (MIN)
   / | \  / | \  / | \
  [...]  [...]  [...] ← 상대가 응수할 수 있는 수들
```

- **MAX 노드 (내 차례)**: 최대값을 원함 → 가장 좋은 수를 선택
- **MIN 노드 (상대 차례)**: 최소값을 강요함 → 나한테 가장 나쁜 수를 선택

### Minimax 동작 원리

1. 게임 트리를 일정 깊이까지 펼침 (예: 4수 앞까지)
2. 리프 노드에서 평가 함수 계산
3. MIN 노드: 자식들 중 최소값 선택
4. MAX 노드: 자식들 중 최대값 선택
5. 루트까지 역전파

예시:

```
                [MAX: ?]
               /         \
          [MIN: ?]      [MIN: ?]
          /    \        /    \
        +5    +3      +7    +2

MIN 노드들 평가:
- 왼쪽 MIN: min(5, 3) = 3
- 오른쪽 MIN: min(7, 2) = 2

MAX 노드 평가:
- max(3, 2) = 3

따라서 왼쪽 수를 선택!
```

🎯 **"왜 상대 차례에서 최소값을 선택한다고 가정할까요?"**

*(학생 답변 듣기)*

네, 상대는 자기한테 좋은 수를 두는데, 그게 나한테는 나쁜 수니까 내 입장에서는 최소값이 되는 거죠!

### 깊이 제한의 필요성

이론적으로는 게임 끝까지 봐야 하지만, 현실적으로 불가능해요.

틱택토 예시:
- 깊이 1: 9가지
- 깊이 2: 9 × 8 = 72가지
- 깊이 3: 9 × 8 × 7 = 504가지
- ...
- 끝까지: 255,168가지

세균 전쟁은 훨씬 더 복잡하죠.

그래서 **깊이 제한**을 둬요. 보통 2~6 정도.

🎯 **"깊이를 늘리면 항상 더 강해질까요?"**

*(학생 생각할 시간)*

대부분은 그래요. 하지만 두 가지 문제가 있어요:
1. **시간**: 깊이가 1 늘 때마다 계산량이 기하급수적으로 증가
2. **평가 함수의 정확도**: 평가 함수가 부정확하면 깊이를 늘려도 오히려 잘못된 결론을 내릴 수 있음

### Negamax: Minimax의 간단한 변형

MIN과 MAX를 번갈아 계산하는 게 복잡해 보이죠?

수학적 트릭이 있어요:

```
min(a, b) = -max(-a, -b)
```

이걸 이용하면 항상 MAX만 계산하면 돼요!

**Negamax 의사코드:**

```python
def negamax(state, depth, player):
    # 종료 조건
    if depth == 0 or game_over(state):
        return evaluate(state, player)

    max_score = -무한대

    for move in valid_moves:
        next_state = state.after_move(move)
        # 상대 입장에서 계산한 값의 음수 = 내 입장의 값
        score = -negamax(next_state, depth-1, opponent(player))
        max_score = max(max_score, score)

    return max_score
```

**핵심**:
- 상대 입장에서 계산한 값에 `-`를 붙이면 내 입장의 값!
- 항상 최대값만 찾으면 됨
- 코드가 훨씬 간단해짐

---

## 실습 3: Minimax Agent (15분)

드디어 가장 똑똑한 AI를 만들 시간입니다!

### 코드 구현

*(agents/minimax_agent.py 파일 열기)*

```python
def minimax_agent(state, player_number, depth=3):
    valid_moves = state.get_valid_moves()
    if not valid_moves:
        return None

    best_move = None
    best_score = float('-inf')

    for move in valid_moves:
        next_state = state.clone()
        next_state.play_move(move, player_number)

        # Negamax로 평가
        score = -negamax(next_state, depth - 1, 3 - player_number)

        if score > best_score:
            best_score = score
            best_move = move

    return best_move

def negamax(state, depth, player_number):
    # 종료 조건
    if depth == 0 or state.is_game_over():
        return evaluate(state, player_number)

    valid_moves = state.get_valid_moves()
    if not valid_moves:
        return evaluate(state, player_number)

    max_score = float('-inf')

    for move in valid_moves:
        next_state = state.clone()
        next_state.play_move(move, player_number)

        # 재귀: 상대 입장에서 계산
        score = -negamax(next_state, depth - 1, 3 - player_number)
        max_score = max(max_score, score)

    return max_score

def evaluate(state, player_number):
    if state.is_game_over():
        winner = state.get_winner()
        if winner == player_number:
            return 1000
        elif winner == 3 - player_number:
            return -1000
        else:
            return 0

    my_count = state.count_stones(player_number)
    opp_count = state.count_stones(3 - player_number)
    return my_count - opp_count
```

### 코드 설명

**핵심 부분 설명:**

1. **`minimax_agent`**: 실제로 수를 선택하는 함수
   - 모든 가능한 수를 시도
   - 각 수에 대해 negamax로 평가
   - 가장 좋은 수 반환

2. **`negamax`**: 재귀적으로 게임 트리 탐색
   - `depth == 0`: 더 이상 깊이 들어가지 않음 → 평가
   - 각 수를 둬보고, 상대 입장에서 재귀 호출
   - `-`를 붙여서 내 입장의 점수로 변환

3. **`evaluate`**: 상태 평가
   - 게임 끝났으면 승/패/무 점수
   - 아니면 돌 개수 차이

🎯 **"왜 재귀 호출에서 depth - 1을 하나요?"**

한 수 더 깊이 들어갔으니까 남은 깊이를 1 줄이는 거죠!

### 실험: 깊이별 성능

깊이를 바꿔가며 실험해봅시다.

**깊이 2:**
```bash
python game.py greedy minimax2 --visualize
```

**깊이 3:**
```bash
python game.py greedy minimax3 --visualize
```

**깊이 4:**
```bash
python game.py greedy minimax4 --visualize
```

🎯 **"뭘 관찰할 수 있나요?"**

*(학생과 함께 관찰)*

- 깊이가 깊을수록 생각하는 시간이 길어짐
- 하지만 더 좋은 수를 선택함
- Greedy보다 훨씬 강함!

### Greedy vs Minimax 대전

승률을 측정해봅시다:

```bash
python benchmark.py greedy minimax3 --games 50
```

결과가 어떻게 나왔나요? 아마 Minimax가 70-80% 이상 이길 거예요.

🎯 **"왜 100% 이기지 않을까요?"**

좋은 질문이에요!

1. **깊이 제한**: 끝까지 보지 못하므로 완벽하지 않음
2. **평가 함수**: 우리의 평가 함수가 단순해서 가끔 실수할 수 있음
3. **게임의 운**: 초반 배치 등에 따라 유불리가 생길 수 있음

하지만 확실히 더 강하죠!

### 시간 복잡도 분석

🎯 **"Minimax가 Greedy보다 몇 배나 느릴까요?"**

*(칠판에 계산)*

- Greedy: 가능한 수 b개를 1번씩 확인 → O(b)
- Minimax (깊이 d): 매 단계마다 b개 분기 → O(b^d)

예를 들어:
- b = 20, d = 3이면: 20^3 = 8,000배 느림!
- b = 20, d = 4이면: 20^4 = 160,000배 느림!

그래서 다음 주에 배울 **Alpha-Beta Pruning**이 중요해요. 쓸데없는 계산을 건너뛰는 기법이에요.

---

## 정리 (5분)

오늘 정말 많은 걸 배웠어요. 정리해봅시다.

### 오늘 배운 것

1. **게임 트리**
   - 상태, 행동, 전이
   - 게임을 트리 구조로 생각하기

2. **평가 함수**
   - 현재 상태의 좋고 나쁨을 숫자로 표현
   - 간단한 예: 내 돌 수 - 상대 돌 수

3. **세 가지 에이전트**
   - **Random**: 아무거나 둠 (생각 안 함)
   - **Greedy**: 1수만 봄 (근시안적)
   - **Minimax**: 여러 수를 내다봄 (상대도 고려)

4. **Negamax 트릭**
   - MIN과 MAX를 하나로 통합
   - 코드가 간결해짐

### 핵심 교훈

> **"더 많이 생각하는 AI가 더 강하다. 하지만 시간이 많이 걸린다."**

이게 AI의 영원한 딜레마예요. 다음 주에는 이 딜레마를 조금이라도 해결하는 방법을 배울 거예요.

### 다음 주 예고: Alpha-Beta Pruning

다음 주에는 **Alpha-Beta Pruning**을 배웁니다.

간단히 말하면:
- Minimax와 결과는 똑같음
- 하지만 쓸데없는 계산을 건너뜀
- 속도가 수십 배~수백 배 빠름!

예를 들어:

```
      [MAX]
      /    \
   [MIN]  [MIN]
   / \     ...
  10  5
```

왼쪽 MIN에서 10을 발견했으면, 그 다음 5는 볼 필요 없어요. 어차피 MIN은 더 작은 값을 선택하니까요.

이런 식으로 가지치기를 하면 엄청나게 빨라집니다!

### 과제

1. **코드 완성하기**
   - 오늘 만든 세 에이전트를 모두 완성
   - 자신만의 평가 함수 추가해보기 (선택)

2. **백준 문제 풀기**
   - 백준 1327번 "소트 게임" (BFS로 게임 트리 탐색)
   - 백준 9095번 "1, 2, 3 더하기" (DP 연습)

3. **생각해오기**
   - "틱택토 AI를 만든다면 어떻게 만들 것인가?"
   - "평가 함수를 어떻게 개선할 수 있을까?"

### 질문 있나요?

🎯 **"오늘 배운 내용 중에 궁금한 거 있어요?"**

*(학생 질문 받기)*

---

### 마무리 인사

오늘 정말 수고했어요! 처음 듣는 개념들이 많았을 텐데 잘 따라와줬어요.

게임 AI의 기초를 이해했으니, 앞으로는 점점 더 재미있어질 거예요. 다음 주에는 훨씬 빠른 AI를 만들어볼 거니까 기대하세요!

과제 풀다가 막히면 언제든지 질문하고요. 다음 주에 만나요!

---

## 참고 자료 (학생에게 전달)

- `week01_minimax/README.md`: 오늘 내용 요약
- `agents/` 폴더: 모든 에이전트 코드
- `game.py`: 세균 전쟁 게임 엔진
- `benchmark.py`: 에이전트 대전 도구

**추천 읽을거리:**
- 위키백과: Minimax Algorithm
- 유튜브: "How AlphaGo Works" (영어지만 자막 있음)
- 책: "게임 인공지능 프로그래밍" (추천만, 안 읽어도 됨)

---

**강사 노트:**
- 학생 반응 보면서 속도 조절
- 코드는 너무 깊이 들어가지 말고 큰 그림 위주로
- 시각화 자료 적극 활용 (칠판 그림, 게임 실행 화면)
- 질문을 많이 던져서 학생이 스스로 생각하게 유도
- 다음 주 Alpha-Beta로 자연스럽게 연결
