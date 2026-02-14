# Week 3: 탐색 최적화 - Iterative Deepening, Transposition Table, PVS

## 주제 개요

Alpha-Beta Pruning의 성능을 극대화하는 세 가지 핵심 최적화 기법을 학습합니다.

### 학습 목표

1. **Iterative Deepening (반복 심화)**: 시간 제한 내에서 최적의 깊이까지 자동으로 탐색
2. **Transposition Table (치환표)**: 중복 계산 제거 + Move Ordering으로 가지치기 효율 극대화
3. **Principal Variation Search (PVS)**: Null window를 활용한 초고속 탐색

### 성능 향상 요약

| 기법 | Elo 변화 | 주요 효과 |
|------|----------|-----------|
| Iterative Deepening | +50 | 시간 관리 자동화 |
| TT Move Ordering | +50 | 가지치기 효율 증가 |
| TT Cutoff | ±0 | 얕은 탐색에서는 오버헤드 |
| PVS | +50 | Null window 최적화 |
| **총 향상** | **+150 Elo** | Week 2 대비 |

---

## 디렉토리 구조

```
week03_search_opt/
├── README.md                        # 이 파일
├── lecture.md                       # 수업 자료 (500+ lines)
├── script.md                        # 수업 대본 (600+ lines, 90분)
├── alphano/
│   └── id_tt_pvs_agent.py          # ALPHANO 제출 코드 (완전 최적화)
└── baekjoon/
    ├── boj_16877.py                # 핌버 (Gold II)
    ├── boj_16895.py                # 님 게임 3 (Gold III)
    ├── boj_9661.py                 # 돌 게임 7 (Gold II)
    ├── boj_2600.py                 # 구슬게임 (Gold IV)
    ├── boj_4370.py                 # 곱셈 게임 (Gold IV)
    ├── boj_16882.py                # 카드 게임 (Gold III)
    ├── boj_16890.py                # 창업 (Gold I)
    └── boj_5232.py                 # Grid Nim (Gold I)
```

---

## 파일 설명

### 1. lecture.md (수업 자료)

**분량**: 500+ lines

**구성**:
1. 복습: Alpha-Beta Pruning
2. Iterative Deepening
   - 문제 상황과 해결책
   - 시간 복잡도 분석 (20% 오버헤드)
   - 시간 관리 전략
3. Zobrist Hashing
   - XOR 기반 O(1) 증분 업데이트
   - XORShift 해시 함수
4. Transposition Table
   - 엔트리 구조 (best_move, flag, depth, value)
   - PV_NODE, CUT_NODE, ALL_NODE
   - Move Ordering (+50 Elo)
5. TT Cutoff
   - 깊이 조건의 중요성
   - 실전 성능 (±0 Elo in ATAXX)
6. Principal Variation Search
   - Null window 개념
   - 재탐색 메커니즘
   - +50 Elo 향상
7. 누적 성능 비교표
8. 다음 주 예고 (MCTS)

**특징**:
- 한국어로 작성
- 수학적 증명 포함
- 실험 결과 및 성능 분석
- 코드 예시 다수

---

### 2. script.md (수업 대본)

**분량**: 600+ lines
**수업 시간**: 90분

**타임라인**:
- [00:00-00:05] 도입 및 복습
- [00:05-00:15] 이론 1: Iterative Deepening
- [00:15-00:25] 실습 1: ID 구현
- [00:25-00:40] 이론 2: Zobrist + TT
- [00:40-00:55] 실습 2: TT + Move Ordering
- [00:55-01:05] 이론 3: TT Cutoff
- [01:05-01:15] 이론 4: PVS
- [01:15-01:25] 실습 3: PVS 구현 + 성능 비교
- [01:25-01:30] 정리 및 과제 안내

**특징**:
- 강사가 읽으며 진행하는 형식
- 🎯 표시로 학생 질문 포함
- 실습 코드 및 실행 명령어
- FAQ 섹션 포함

---

### 3. alphano/id_tt_pvs_agent.py (ALPHANO 제출 코드)

**구현 기법**:
- ✅ Negamax + Alpha-Beta
- ✅ Iterative Deepening (시간 기반 종료)
- ✅ Zobrist Hashing (XORShift)
- ✅ Transposition Table (dict 기반)
- ✅ Move Ordering (TT best move 우선)
- ✅ Principal Variation Search
- ✅ 시간 관리 (my_time 기반 적응형)

**ALPHANO 프로토콜**:
```
READY FIRST/SECOND → OK
TURN my_time opp_time → MOVE x1 y1 x2 y2 (1-indexed)
OPP x1 y1 x2 y2 → update board
FINISH → exit
```

**ATAXX 규칙**:
- 7x7 보드, 1-indexed 좌표
- Split (거리 1): 복사하여 배치
- Jump (거리 2): 이동
- 인접 8방향 감염

**시간 관리**:
```python
my_time > 60000ms → 50ms/턴
my_time > 20000ms → 150ms/턴
my_time ≤ 20000ms → 10ms/턴
```

**평가 함수**:
```python
score = 돌 수 차이 + 이동성 * 0.1
```

**실행 방법**:
```bash
python3 id_tt_pvs_agent.py
```

---

### 4. Baekjoon 문제 풀이 (8개)

모든 문제는 게임 이론 + 탐색 최적화를 다룹니다.

#### boj_16877.py - 핌버 (Gold II)
- **주제**: Fibonacci Nim + Sprague-Grundy
- **핵심**: 피보나치 수만큼 제거, 그런디 수 XOR
- **난이도**: ★★★☆☆

#### boj_16895.py - 님 게임 3 (Gold III)
- **주제**: Standard Nim + 경우의 수
- **핵심**: 선공이 이기는 첫 수의 개수
- **난이도**: ★★★☆☆

#### boj_9661.py - 돌 게임 7 (Gold II)
- **주제**: 4^k Nim + 주기성
- **핵심**: N % 5 패턴 발견
- **난이도**: ★★☆☆☆

#### boj_2600.py - 구슬게임 (Gold IV)
- **주제**: 2D 게임 DP
- **핵심**: dp[i][j] = (i개, j개 남았을 때 승패)
- **난이도**: ★★★☆☆

#### boj_4370.py - 곱셈 게임 (Gold IV)
- **주제**: 게임 DP + 실수 처리
- **핵심**: 2~9 곱하여 n 이상 만들기
- **난이도**: ★★★☆☆

#### boj_16882.py - 카드 게임 (Gold III)
- **주제**: Nim 변형 + 최적 전략
- **핵심**: 1장 또는 2장(같은 숫자) 가져가기
- **난이도**: ★★★★☆

#### boj_16890.py - 창업 (Gold I)
- **주제**: 그리디 + 게임 이론
- **핵심**: 양쪽 최선, 덱으로 앞/뒤 선택
- **난이도**: ★★★★☆

#### boj_5232.py - Grid Nim (Gold I)
- **주제**: 2D Nim
- **핵심**: 모든 칸의 돌 개수 XOR
- **난이도**: ★★★☆☆

**실행 방법**:
```bash
python3 baekjoon/boj_16877.py < input.txt
```

---

## 핵심 개념 정리

### 1. Iterative Deepening

**장점**:
- 시간 관리 자동화
- 항상 유효한 수 반환
- Move Ordering 힌트 제공

**오버헤드**:
- 이론적 20%
- 실전에서는 장점으로 상쇄

**구현**:
```python
for depth in range(1, MAX_DEPTH):
    if time_remaining < threshold:
        break
    score, move = alpha_beta(board, depth, -INF, INF)
    best_move = move
```

### 2. Zobrist Hashing

**핵심**:
- XOR 기반 O(1) 증분 업데이트
- 64비트 정수로 보드 표현

**XORShift**:
```python
def xorshift64(x):
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7)
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x
```

### 3. Transposition Table

**엔트리 구조**:
```python
class TTEntry:
    best_move: Move
    flag: PV_NODE | CUT_NODE | ALL_NODE
    depth: int
    value: float
```

**Move Ordering**:
- TT의 best_move를 첫 번째로 탐색
- +50 Elo 향상

**Replace 정책**:
- Depth-preferred: 더 깊은 탐색 결과 우선

### 4. Principal Variation Search

**핵심 아이디어**:
- 첫 수: Full window (-beta, -alpha)
- 나머지: Null window (-alpha-1, -alpha)
- 실패 시: Full window로 재탐색

**성능**:
- +50 Elo
- 탐색 노드 30~40% 감소

---

## 학습 로드맵

### 선수 지식
- Week 1: Minimax
- Week 2: Alpha-Beta Pruning

### Week 3 학습 순서
1. lecture.md 읽기 (1시간)
2. script.md로 복습 (30분)
3. id_tt_pvs_agent.py 분석 (1시간)
4. Baekjoon 문제 풀이 (3시간)
5. ALPHANO 제출 및 테스트 (1시간)

### 다음 단계
- Week 4: Monte Carlo Tree Search (MCTS)

---

## 실습 가이드

### ALPHANO 제출

1. **테스트 실행**:
```bash
cd alphano
python3 id_tt_pvs_agent.py
```

2. **입력 예시**:
```
READY FIRST
TURN 120000 120000
OPP 1 1 1 2
TURN 119950 119900
FINISH
```

3. **출력 예시**:
```
OK
MOVE 7 7 6 6
MOVE 6 6 5 5
```

### Baekjoon 문제 풀이

1. **문제 선택**: 난이도 순으로 시작
   - boj_9661.py (가장 쉬움)
   - boj_16895.py
   - boj_5232.py
   - boj_16877.py
   - boj_2600.py
   - boj_4370.py
   - boj_16882.py
   - boj_16890.py (가장 어려움)

2. **실행**:
```bash
python3 baekjoon/boj_16877.py
```

3. **입력 예시** (boj_16877):
```
3
1 2 3
```

4. **출력**:
```
koosaga
```

---

## 디버깅 팁

### TT 검증
```python
if tt_move in legal_moves:
    ...
else:
    print(f"WARNING: TT move {tt_move} is illegal!")
```

### 시간 초과 방지
```python
if elapsed_time > time_limit * 0.9:
    break  # 90%에서 중단
```

### PVS 재탐색 통계
```python
re_searches / total_searches
# 이상적: < 10%
```

---

## 참고 자료

### 논문
1. Zobrist (1970): "A New Hashing Method with Application for Game Playing"
2. Schaeffer (1989): "The History Heuristic and Alpha-Beta Search Enhancements"
3. Marsland & Campbell (1982): "Parallel Search of Strongly Ordered Game Trees"

### 온라인 자료
- [Chess Programming Wiki](https://www.chessprogramming.org/)
- [Infossm Blog](https://infossm.github.io/) (Week 3 참고 자료)
- [Alpha-Beta 시뮬레이터](https://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab_tree_practice/)

### 추가 학습
- Killer Move Heuristic
- History Heuristic
- Aspiration Windows
- MTD(f) 알고리즘

---

## 과제

### 1. ALPHANO 제출
- `alphano/id_tt_pvs_agent.py` 완성
- 리더보드에 제출
- 성능 측정 및 분석

### 2. Baekjoon 문제 풀이
- 8개 문제 모두 해결
- 각 문제의 핵심 개념 정리

### 3. 성능 비교 실험
- Alpha-Beta vs ID vs ID+TT vs ID+TT+PVS
- 각 버전의 탐색 깊이 측정
- Elo 변화 추정

---

## FAQ

**Q1: Iterative Deepening에서 이전 깊이의 TT를 사용할 수 있나요?**
A: 네! TT는 계속 누적되며, 이전 깊이의 best_move가 다음 깊이의 Move Ordering에 사용됩니다.

**Q2: TT 크기 제한은 어떻게 하나요?**
A: `len(tt) > MAX_SIZE`일 때 `tt.clear()` 또는 LRU 정책으로 오래된 엔트리를 삭제합니다.

**Q3: PVS에서 재탐색이 자주 발생하면?**
A: Move Ordering이 나쁜 것입니다. TT가 제대로 작동하는지 확인하세요.

**Q4: ATAXX 말고 다른 게임에서도 사용 가능한가요?**
A: 네! 모든 2인 완전정보 게임 (체스, 오델로, 장기 등)에 적용 가능합니다.

**Q5: TT Cutoff가 효과가 없는 이유는?**
A: ATAXX에서는 얕은 탐색이 빠르고, Iterative Deepening으로 인해 cutoff 기회가 적습니다. 깊은 탐색(depth>10)에서는 효과적입니다.

---

## 라이센스

이 자료는 교육 목적으로 제작되었습니다.

---

**Week 3 완료 체크리스트**:
- [ ] lecture.md 학습
- [ ] script.md 복습
- [ ] id_tt_pvs_agent.py 이해 및 실행
- [ ] Baekjoon 8문제 해결
- [ ] ALPHANO 제출
- [ ] 성능 비교 실험
- [ ] Week 4 준비 (MCTS)
