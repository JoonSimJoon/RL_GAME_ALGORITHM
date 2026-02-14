# Week 2: Alpha-Beta Pruning & 에이전트 성능 검증

## 개요

Week 2에서는 Minimax 알고리즘의 효율성을 크게 향상시키는 **Alpha-Beta Pruning**을 학습하고, AI 에이전트의 성능을 통계적으로 검증하는 방법을 배웁니다.

## 주요 학습 내용

1. **Alpha-Beta Pruning**: 불필요한 탐색을 제거하여 5~10배 속도 향상
2. **SPRT (Sequential Probability Ratio Test)**: 통계적 성능 검증
3. **Elo Rating System**: 상대 평가 시스템
4. **평가 함수 개선**: 위치 가중치, 이동성, 게임 단계별 전략

## 디렉토리 구조

```
week02_alpha_beta/
├── README.md                          # 이 파일
├── lecture.md                         # 수업 자료 (1,120줄)
├── script.md                          # 수업 대본 (876줄)
├── alphano/
│   └── alpha_beta_agent.py           # ALPHANO 제출용 Alpha-Beta 에이전트
└── baekjoon/
    ├── boj_9659.py                   # 돌 게임 5 (Silver III)
    ├── boj_9660.py                   # 돌 게임 6 (Silver V)
    ├── boj_11062.py                  # 카드 게임 (Gold III)
    ├── boj_2040.py                   # 수 게임 (Gold IV)
    ├── boj_11868.py                  # 님 게임 2 (Silver II)
    ├── boj_11694.py                  # 님 게임 (Silver I)
    ├── boj_16894.py                  # 약수 게임 (Gold IV)
    ├── boj_16571.py                  # 알파 틱택토 (Gold III)
    └── boj_4664.py                   # Find the Winning Move (Gold I)
```

## 파일 설명

### 1. lecture.md (1,120줄)

고등학생을 대상으로 한 상세한 수업 자료:

- **복습**: Minimax/Negamax의 문제점
- **Alpha-Beta Pruning 원리**: α, β 개념, Cutoff 조건
- **상세 예시**: ASCII 트리로 가지치기 과정 시각화
- **Negamax + Alpha-Beta 구현**: 의사코드 및 Python 구현
- **성능 분석**: O(b^(d/2)) vs O(b^d)
- **SPRT**: 통계적 검증 방법, LLR 계산
- **Elo Rating**: 기대 승률, Rating 업데이트
- **평가 함수 개선**: 위치 가중치, 이동성, 게임 단계별 전략
- **실험 결과**: Minimax vs Alpha-Beta 비교

### 2. script.md (876줄)

90분 수업을 위한 강사용 대본:

**구성**:
- 도입 (5분): 복습 및 목표 소개
- 이론 1 (20분): Alpha-Beta Pruning 원리 + 예시
- 실습 1 (15분): Alpha-Beta 구현 및 실행
- 이론 2 (15분): SPRT, Elo Rating
- 실습 2 (10분): SPRT 실험
- 이론 3 (10분): 평가 함수 개선
- 실습 3 (10분): 평가 함수 실험
- 정리 (5분): 핵심 요약, 다음 주 예고

**특징**:
- 친근한 말투, 고등학생 눈높이
- 🎯 표시로 학생 참여 질문
- 코드 블록 및 실행 예시
- 화이트보드 그림 설명

### 3. alphano/alpha_beta_agent.py

ALPHANO 프로토콜을 따르는 Alpha-Beta Pruning 에이전트:

**핵심 기능**:
- Negamax + Alpha-Beta Pruning (depth=4)
- 개선된 평가 함수:
  - 위치 가중치 (코너 중시)
  - 이동성 (가능한 수의 개수)
  - 게임 단계별 가중치 조정
- Move Ordering (간단한 휴리스틱)
- ATAXX 7×7 게임 규칙 완벽 구현

**실행 방법**:
```bash
python alpha_beta_agent.py
```

**ALPHANO 제출**:
1. 코드를 복사
2. ALPHANO 플랫폼에 제출
3. 다른 에이전트와 대전

### 4. Baekjoon 문제 풀이 (9개)

#### 게임 이론 기초

**boj_9659.py - 돌 게임 5 (Silver III)**
- 1개 또는 3개 가져가기
- 패턴: 홀수면 SK, 짝수면 CY
- N ≤ 10^18 처리

**boj_9660.py - 돌 게임 6 (Silver V)**
- 1, 3, 4개 가져가기
- 주기 7 패턴 발견
- N % 7로 O(1) 해결

**boj_11868.py - 님 게임 2 (Silver II)**
- 전형적인 님 게임
- XOR 연산으로 승패 판별
- Sprague-Grundy 정리

**boj_11694.py - 님 게임 (Silver I)**
- Misère Nim (마지막 돌 가져가면 패배)
- 모든 더미 ≤ 1일 때 특수 처리
- XOR 연산

**boj_16894.py - 약수 게임 (Gold IV)**
- 약수를 이용한 게임
- 그런디 수 계산
- 홀짝성 패턴

#### 구간 DP + Minimax

**boj_11062.py - 카드 게임 (Gold III)**
- 양쪽 끝 카드 선택 게임
- 구간 DP: dp[i][j] = 선공의 최대 점수
- 두 플레이어 모두 최적 플레이

**boj_2040.py - 수 게임 (Gold IV)**
- boj_11062와 유사
- 점수 차이 출력

#### Minimax/Alpha-Beta 응용

**boj_16571.py - 알파 틱택토 (Gold III)**
- 3×3 틱택토
- Minimax with memoization
- 3진수 상태 인코딩

**boj_4664.py - Find the Winning Move (Gold I)**
- 4×4 틱택토 강제 승리 찾기
- **Alpha-Beta Pruning 필수**
- 완전 탐색은 TLE

## 실습 예제

### Alpha-Beta vs Minimax 비교

```python
# Minimax (depth=3)
def minimax(board, depth, player):
    # 모든 노드 탐색
    # 평균 125,834개 노드

# Alpha-Beta (depth=3)
def negamax_alpha_beta(board, depth, alpha, beta, player):
    # 가지치기 적용
    # 평균 24,167개 노드 (80.8% 감소!)
```

### SPRT 실험

```bash
# Alpha-Beta (depth=4) vs Minimax (depth=3)
python sprt_test.py --agent1 alphabeta --depth1 4 --agent2 minimax --depth2 3

# 예상 결과:
# Game 38: LLR=3.128 → H1 채택!
# Alpha-Beta(d=4)가 통계적으로 유의미하게 강함
```

### Elo Rating 계산

```python
# 승률 → Elo 차이
# 승률 57.2% → Elo +50
# 승률 64.0% → Elo +100
# 승률 76.0% → Elo +200

E_A = 1 / (1 + 10**((R_B - R_A) / 400))
```

## 성능 비교 실험 결과

### 실험 1: 정확성 검증

| 에이전트 | 승 | 패 | 승률 |
|---------|----|----|------|
| Minimax (d=3) | 143 | 143 | 50.0% |
| Alpha-Beta (d=3) | 143 | 143 | 50.0% |

**결론**: 동일한 결과 보장!

### 실험 2: 효율성 검증

| 에이전트 | 평균 노드 수 | 평균 시간 (초) |
|---------|------------|--------------|
| Minimax (d=3) | 125,834 | 0.523 |
| Alpha-Beta (d=3) | 24,167 | 0.098 |

**결론**: 약 5.3배 빠름!

### 실험 3: 성능 향상

| 에이전트 | 승 | 패 | 승률 | Elo |
|---------|----|----|------|-----|
| Alpha-Beta (d=3) | 97 | 189 | 33.9% | 1591 |
| Alpha-Beta (d=4) | 189 | 97 | 66.1% | 1823 |

**결론**: 깊이 +1 → 승률 +32%, Elo +232!

### 실험 4: 평가 함수 개선

| 평가 함수 | 승 | 패 | 승률 | Elo |
|----------|----|----|------|-----|
| 기본 (돌 개수만) | 78 | 208 | 27.3% | 1650 |
| 개선 (위치+이동성) | 208 | 78 | 72.7% | 1950 |

**결론**: 평가 함수 개선 → Elo +300!

## 학습 목표

이번 주차를 마치면:

- [ ] Alpha-Beta Pruning 원리를 이해하고 구현할 수 있음
- [ ] α, β의 의미와 Cutoff 조건을 설명할 수 있음
- [ ] SPRT로 통계적 검증을 수행할 수 있음
- [ ] Elo Rating을 계산하고 해석할 수 있음
- [ ] 평가 함수를 게임에 맞게 개선할 수 있음
- [ ] ALPHANO에 Alpha-Beta 에이전트를 제출할 수 있음
- [ ] 게임 이론 문제 9개를 해결할 수 있음

## 다음 주 예고

**Week 3: Monte Carlo Tree Search (MCTS)**

- Minimax/Alpha-Beta의 한계
- Monte Carlo 방법
- UCB1 알고리즘
- MCTS 4단계: Selection, Expansion, Simulation, Backpropagation
- AlphaGo의 핵심 알고리즘!

## 참고 자료

**논문**:
- Shannon, C. (1950). "Programming a Computer for Playing Chess"
- Knuth, D. & Moore, R. (1975). "An Analysis of Alpha-Beta Pruning"

**온라인**:
- Chess Programming Wiki: https://www.chessprogramming.org/Alpha-Beta
- AlphaGo 논문: https://www.nature.com/articles/nature24270

**책**:
- "Artificial Intelligence: A Modern Approach" - Russell & Norvig
- "Algorithms" - Robert Sedgewick

## 문의

강의 내용이나 코드에 대한 질문은 메가스터디 플랫폼을 통해 문의해주세요.

---

**Week 2 자료 끝**

모든 파일이 한국어로 작성되었으며, 고등학생 수준에 맞춰 친근하고 이해하기 쉽게 설명되어 있습니다.
