# 게임 알고리즘 & 강화학습 스터디

> 게임 AI 에이전트를 직접 구현하며 알고리즘과 강화학습을 배우는 10주 커리큘럼
>
> 실습 플랫폼: [ALPHANO](https://alphano.co.kr) - AI 에이전트 대전 온라인 저지

---

## 목표

1. 게임 알고리즘의 핵심 개념(탐색, 평가, 최적화)을 이해한다
2. 강화학습 기초부터 Deep RL까지 단계적으로 학습한다
3. ALPHANO 플랫폼에서 실제 AI 에이전트를 구현하고 대전한다
4. 학습 결과를 소논문으로 작성하여 학술대회에 투고한다

---

## ALPHANO 문제 맵

| # | 문제 | 유형 | 보드 | 핵심 알고리즘 |
|---|------|------|------|---------------|
| 1 | **세균 전쟁** | 영토 확장 (1v1) | 7×7 | Minimax, Alpha-Beta, MCTS |
| 2 | **쥐를 잡자** | 비대칭 대전 (공격/수비) | 7×11 | 게임 트리 탐색, 휴리스틱 평가 |
| 3 | **Betris** | 베팅 + 블록 배치 | 5×5 | 확률적 의사결정, 게임 이론 |
| 4 | **Betris (Small)** | 베팅 + 블록 배치 | 3×3 | 완전 탐색, 기대값 계산 |
| 5 | **Betris (Large)** | 베팅 + 블록 배치 | 7×7 | MCTS, 강화학습 |

---

## 10주 커리큘럼

### Phase 1: 게임 알고리즘 기초 (1~3주)

> **참고 자료**
> - [Advanced Game Search Algorithms (1)](https://infossm.github.io/blog/2025/10/25/adv-game-search/) - Random → Greedy → Minimax → Alpha-Beta
> - [Advanced Game Search Algorithms (2)](https://infossm.github.io/blog/2025/11/24/adv-game-search-2/) - Iterative Deepening → Transposition Table → PVS

#### 1주차 - 게임 에이전트 기초: Random → Greedy → Minimax

| 항목 | 내용 |
|------|------|
| **읽기** | [Adv. Game Search (1)](https://infossm.github.io/blog/2025/10/25/adv-game-search/) - Introduction ~ Minimax Algorithm |
| **이론** | 게임 트리, 상태 공간, 평가 함수(`내 돌 수 - 상대 돌 수`), Minimax, Negamax |
| **실습 1** | 세균 전쟁(ATAXX) **Random Agent** 구현 → ALPHANO 제출 (baseline) |
| **실습 2** | **Greedy Agent** 구현 (1수 앞만 평가) → Random 대비 승률 확인 |
| **실습 3** | **Minimax Agent** 구현 (깊이 제한 탐색) → Greedy 대비 승률 확인 |
| **ALPHANO** | 세균 전쟁 - 3가지 에이전트 모두 제출, 리더보드 비교 |
| **핵심 개념** | Negamax 변환: `min(a,b) = -max(-b,-a)` → 코드 단순화 |
| **과제** | SPRT(순차확률비검정)로 에이전트 간 실력 차이 통계적 검증 |

<details>
<summary><b>백준 문제 (1주차)</b></summary>

**Step 1: 게임 승패 판별 (구현 워밍업)**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [7682](https://www.acmicpc.net/problem/7682) | 틱택토 | Silver II | 틱택토 게임 상태가 유효한지 판별. 게임 규칙을 코드로 옮기는 연습 |
| [2615](https://www.acmicpc.net/problem/2615) | 오목 | Silver I | 19×19 보드 위 오목 승리 판별. 보드 게임 상태 탐색 기초 |

**Step 2: 필승 전략 게임 입문 (게임 이론 첫걸음)**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [9655](https://www.acmicpc.net/problem/9655) | 돌 게임 | Silver V | 돌 1·3개 가져가기, 마지막 돌 = 승리. **홀짝 패턴** 발견 |
| [9656](https://www.acmicpc.net/problem/9656) | 돌 게임 2 | Silver V | 돌 1·3개, 마지막 돌 = 패배. 조건 반전 시 전략 변화 관찰 |
| [9657](https://www.acmicpc.net/problem/9657) | 돌 게임 3 | Silver III | 돌 1·3·4개 선택. **DP로 필승/필패 상태** 구하기 |
| [9658](https://www.acmicpc.net/problem/9658) | 돌 게임 4 | Silver II | 돌 1·3·4개, 마지막 돌 = 패배. DP 테이블 확장 |

**Step 3: Minimax 직접 구현**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [14429](https://www.acmicpc.net/problem/14429) | 배스킨라빈스 31 | Bronze I | 31 게임의 필승 전략 찾기. 간단한 게임에서 역방향 분석 |
| [28472](https://www.acmicpc.net/problem/28472) | Minimax Tree | Gold V | **Minimax 트리 값 계산** 직접 구현. MAX/MIN 노드 번갈아 선택 |

</details>

#### 2주차 - Alpha-Beta Pruning & 에이전트 성능 검증

| 항목 | 내용 |
|------|------|
| **읽기** | [Adv. Game Search (1)](https://infossm.github.io/blog/2025/10/25/adv-game-search/) - Alpha-Beta Pruning ~ Summary |
| **이론** | Alpha-Beta Pruning (α ≥ β 가지치기), SPRT, Elo Rating |
| **실습 1** | Minimax에 **Alpha-Beta Pruning** 추가 → 동일 결과, 5~10배 속도 향상 확인 |
| **실습 2** | 속도 향상분으로 **탐색 깊이 증가** → 승률 변화 측정 |
| **ALPHANO** | 세균 전쟁 - Alpha-Beta 에이전트 제출, Minimax 대비 Elo 비교 |
| **핵심 개념** | Elo Rating: `E_A = 1/(1+10^((R_B-R_A)/400))`, SPRT로 유의미한 차이 판별 |
| **과제** | 평가 함수 개선 실험 (돌 수 차이 외: 이동성, 코너 점유, 감염 가능성 가중치) |

<details>
<summary><b>백준 문제 (2주차)</b></summary>

**Step 1: 게임 DP 심화 (평가 함수의 원리)**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [9659](https://www.acmicpc.net/problem/9659) | 돌 게임 5 | Silver III | 돌 1·3개, N이 매우 큼. **수학적 패턴 일반화** |
| [9660](https://www.acmicpc.net/problem/9660) | 돌 게임 6 | Silver V | 돌 1·3·4개, N이 매우 큼. 주기성 발견 |
| [11062](https://www.acmicpc.net/problem/11062) | 카드 게임 | Gold III | 양쪽 끝 카드 선택. **구간 DP + Minimax** 핵심 문제 |
| [2040](https://www.acmicpc.net/problem/2040) | 수 게임 | Gold IV | 수열에서 번갈아 선택. 최적 전략 DP |

**Step 2: Nim 게임과 XOR (스프라그-그런디 입문)**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [11868](https://www.acmicpc.net/problem/11868) | 님 게임 2 | Silver II | 돌 더미 k개, **XOR 연산**으로 승패 판별 |
| [11694](https://www.acmicpc.net/problem/11694) | 님 게임 | Silver I | Misère Nim (마지막 돌 = 패배). XOR 변형 |
| [16894](https://www.acmicpc.net/problem/16894) | 약수 게임 | Gold IV | 약수를 이용한 Nim 변형. 그런디 수 계산 연습 |

**Step 3: Alpha-Beta 적용 가능 게임**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [16571](https://www.acmicpc.net/problem/16571) | 알파 틱택토 | Gold III | 3×3 틱택토 **Minimax로 최적해** 구하기. Alpha-Beta 적용 실습 |
| [4664](https://www.acmicpc.net/problem/4664) | Find the Winning Move | Gold I | 4×4 틱택토에서 첫 수 강제승 찾기. **Alpha-Beta Pruning 필수** |

</details>

#### 3주차 - 탐색 최적화: Iterative Deepening, Transposition Table, PVS

| 항목 | 내용 |
|------|------|
| **읽기** | [Adv. Game Search (2)](https://infossm.github.io/blog/2025/11/24/adv-game-search-2/) 전체 |
| **이론** | Iterative Deepening, Zobrist Hashing, Transposition Table (Move Ordering + TT Cutoff), Principal Variation Search (PVS) |
| **실습 1** | **Iterative Deepening** 구현 - 시간 제한 내 최대 깊이 탐색 (+50 Elo) |
| **실습 2** | **Transposition Table** 추가 - Zobrist Hash로 보드 상태 캐싱, Move Ordering으로 최적 수 우선 탐색 (+50 Elo) |
| **실습 3** | **PVS** 구현 - 첫 수는 full window, 나머지는 null window(α, α+1)로 빠르게 확인 (+50 Elo) |
| **ALPHANO** | 세균 전쟁 - 각 기법 추가할 때마다 제출, **누적 Elo 향상** 기록 |
| **핵심 개념** | 시간 관리 전략: 남은 시간 > 1000ms → 150ms/턴, 아니면 10ms/턴 |
| **과제** | 기법별 Elo 향상 비교표 작성 + 탐색 깊이/노드 수 통계 시각화 |

> **3주차 누적 성능 향상 (블로그 실험 결과 기준)**
>
> | 기법 | Elo 변화 | 효과 |
> |------|----------|------|
> | Alpha-Beta (2주차) | baseline | 5~10배 속도 향상 |
> | + Iterative Deepening | +50 Elo | 시간 제한 내 최적 깊이 자동 탐색 |
> | + Move Ordering (TT) | +50 Elo | 가지치기 효율 증가 → 더 깊은 탐색 |
> | + TT Cutoff | ±0 Elo | 얕은 탐색에서는 오버헤드 |
> | + PVS | +50 Elo | Move Ordering과 결합 시 최고 효율 |

<details>
<summary><b>백준 문제 (3주차)</b></summary>

**Step 1: 스프라그-그런디 정리 (게임 상태의 수학적 분석)**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [16877](https://www.acmicpc.net/problem/16877) | 핌버 | Gold II | 피보나치 수만큼 제거. **그런디 수 + mex 연산** |
| [16895](https://www.acmicpc.net/problem/16895) | 님 게임 3 | Gold III | 선공이 이기는 첫 수의 개수 세기. 그런디 정리 심화 |
| [9661](https://www.acmicpc.net/problem/9661) | 돌 게임 7 | Gold II | 돌 4^k개만 가능. **주기성 + 게임 이론** 종합 |

**Step 2: 복합 게임 전략 (탐색 최적화 사고방식 훈련)**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [2600](https://www.acmicpc.net/problem/2600) | 구슬게임 | Gold IV | 두 종류 구슬 동시 진행. **복합 게임 분석** |
| [4370](https://www.acmicpc.net/problem/4370) | 곱셈 게임 | Gold IV | 수를 곱해서 목표 도달. 탐색 공간이 커서 효율적 탐색 필요 |
| [16882](https://www.acmicpc.net/problem/16882) | 카드 게임 | Gold III | Nim 변형 + 전략 최적화. 여러 게임 합산 |
| [16890](https://www.acmicpc.net/problem/16890) | 창업 | Gold I | 배스킨라빈스 + 최적 전략. **탐색 공간 큰 게임의 최적해** |

**Step 3: 종합 도전**

| # | 문제 | 난이도 | 핵심 |
|---|------|--------|------|
| [5232](https://www.acmicpc.net/problem/5232) | Grid Nim | Gold I | 2D 격자 위 Nim. **보드 게임 + 게임 이론** 종합 |

</details>

### Phase 2: MCTS & 강화학습 입문 (4~6주)

#### 4주차 - Monte Carlo Tree Search (MCTS)

| 항목 | 내용 |
|------|------|
| **이론** | MCTS 4단계 (Selection → Expansion → Simulation → Backpropagation), UCB1 공식 |
| **실습 1** | 세균 전쟁에 **MCTS 에이전트** 구현 |
| **실습 2** | Simulation 횟수별 성능 변화 실험 |
| **ALPHANO** | 세균 전쟁 - MCTS 에이전트 제출, 3주차 PVS 에이전트와 대전 비교 |
| **과제** | MCTS vs Alpha-Beta+PVS 승률 비교 그래프 + Elo 측정 |

#### 5주차 - 강화학습 기초 개념

| 항목 | 내용 |
|------|------|
| **이론** | MDP (상태, 행동, 보상, 전이확률), 벨만 방정식, 감가율(γ) |
| **실습** | GridWorld 환경 구현, Value Iteration / Policy Iteration |
| **ALPHANO** | 쥐를 잡자 규칙 분석 + 상태 공간 설계 |
| **과제** | 쥐를 잡자의 상태를 MDP로 모델링하기 |

#### 6주차 - Q-Learning

| 항목 | 내용 |
|------|------|
| **이론** | Q-Learning, SARSA, ε-greedy 탐험 전략, Q-table |
| **실습** | OpenAI Gymnasium FrozenLake Q-Learning 구현 |
| **ALPHANO** | 쥐를 잡자 - 간단한 휴리스틱 에이전트 구현 |
| **과제** | 학습률(α), 감가율(γ), 탐험률(ε) 변화에 따른 학습 곡선 비교 |

### Phase 3: 심화 & 적용 (7~9주)

#### 7주차 - Deep Q-Network (DQN)

| 항목 | 내용 |
|------|------|
| **이론** | 함수 근사, 신경망 기초, Experience Replay, Target Network |
| **실습** | CartPole DQN 구현 (PyTorch) |
| **ALPHANO** | Betris - 베팅 전략에 DQN 적용 실험 |
| **과제** | Replay Buffer 크기, Target Network 갱신 주기 실험 |

#### 8주차 - Policy Gradient & Actor-Critic

| 항목 | 내용 |
|------|------|
| **이론** | REINFORCE, Baseline, Actor-Critic, A2C |
| **실습** | 간단한 환경에서 REINFORCE vs DQN 비교 |
| **ALPHANO** | 세균 전쟁 또는 Betris (Large)에 학습 기반 에이전트 설계 |
| **과제** | 알고리즘별 (Alpha-Beta+PVS vs MCTS vs DQN vs Policy Gradient) 성능 비교표 작성 |

#### 9주차 - 종합 에이전트 개발

| 항목 | 내용 |
|------|------|
| **이론** | Self-play, 앙상블 전략, 하이브리드 접근 (MCTS + 신경망) |
| **실습** | 소논문 주제에 맞는 최종 에이전트 개발 |
| **ALPHANO** | 선택한 문제에 최종 에이전트 제출, 리더보드 순위 기록 |
| **과제** | 실험 데이터 정리 (학습 곡선, 승률, 알고리즘 비교) |

### Phase 4: 소논문 작성 (10주)

#### 10주차 - 논문 작성 & 발표 준비

| 항목 | 내용 |
|------|------|
| **활동** | 실험 결과 시각화 (matplotlib), 논문 초안 작성, 피어리뷰, 수정, 발표 자료 제작 |
| **논문 구조** | 서론 → 관련 연구 → 방법론 → 실험 및 결과 → 결론 |
| **핵심** | 연구 질문 명확화, 비교 실험 결과 해석, 그래프/표 작성 |
| **마감** | 최종 논문 제출 + 학술대회 투고 |

---

## 소논문 주제 예시

| # | 주제 | 비교 실험 |
|---|------|-----------|
| 1 | 세균 전쟁에서 Minimax vs MCTS 전략 비교 연구 | Alpha-Beta 깊이별 vs MCTS 시뮬레이션 횟수별 승률 |
| 2 | 보드 게임 AI에서 탐색 알고리즘과 강화학습의 성능 비교 | MCTS vs DQN vs 하이브리드 접근법 |
| 3 | 비대칭 게임에서의 강화학습 에이전트 설계 | 쥐를 잡자: 공격/수비 역할별 최적 전략 학습 |
| 4 | 불완전 정보 게임에서의 의사결정 전략 연구 | Betris: 베팅 전략의 게임 이론적 분석 + RL 적용 |

---

## 투고 대상 학술대회

실제 고등학생 참가/수상이 확인된 대회만 수록했습니다.

### 고등학생 참가 실적 확인됨

| 대회 | 시기 | 실적 | 비고 |
|------|------|------|------|
| [한국정보과학회 KCC 주니어논문경진대회](https://www.kiise.or.kr/) | 매년 7월 | 충북과학고·청원고 학생이 딥러닝 논문으로 **최우수상** 수상 | AI 논문 직접 부합 |
| [청소년 IT학술대회](https://kitpa.org/) | 연 2회 (하계/동계) | 포항제철고, 세종과학고, 부산SW마이스터고 등 다수 고교생 AI 연구 발표 | 초3~고3 참가 |
| [KSCY 한국청소년학술대회](https://www.kscy.kr/) | 매년 | 1,000명+ 중고등학생 참가, 자연과학/공학 세션 | 아시아 최대 청소년 학술대회 |
| [한화 사이언스 챌린지](https://www.sciencechallenge.or.kr/) | 매년 | 758팀(1,516명) 참가, AI 논문 수상 사례 있음 | 총상금 2억원 |
| [HOBY 국제 청소년 소논문 대회](https://www.hobykorea.com/page/iysc) | 매년 9월 | 중1~고3, 주제 무제한 | 영어 논문 |

### 투고 일정 역산 (KCC 7월 목표 기준)

```
3월 초    커리큘럼 시작
5월 중순  실험 완료 + 논문 초안
6월 초    논문 투고 마감 (KCC 기준)
7월       KCC 발표
```

---

## 기술 스택

| 도구 | 용도 |
|------|------|
| **Python 3.10+** | 메인 언어 (ALPHANO Python3 지원) |
| **PyTorch** | DQN, Policy Gradient 구현 (ALPHANO에서 torch 2.9.0 지원) |
| **NumPy** | 보드 상태 표현, 수치 연산 (ALPHANO에서 numpy 지원) |
| **OpenAI Gymnasium** | 강화학습 기초 실습 환경 |
| **Matplotlib** | 실험 결과 시각화 |
| **ALPHANO** | AI 에이전트 대전 및 평가 플랫폼 |

---

## 프로젝트 구조

```
RL_GAME_ALGORITHM/
├── README.md
├── week01_minimax/          # 1주차: Random → Greedy → Minimax
├── week02_alpha_beta/       # 2주차: Alpha-Beta Pruning & SPRT
├── week03_search_opt/       # 3주차: Iterative Deepening, TT, PVS
├── week04_mcts/             # 4주차: Monte Carlo Tree Search
├── week05_rl_basics/        # 5주차: MDP & 강화학습 기초
├── week06_q_learning/       # 6주차: Q-Learning
├── week07_dqn/              # 7주차: Deep Q-Network
├── week08_policy_gradient/  # 8주차: Policy Gradient & Actor-Critic
├── week09_final_agent/      # 9주차: 종합 에이전트
├── week10_paper/            # 10주차: 논문 작성 & 발표
└── alphano_agents/          # ALPHANO 제출용 에이전트 코드
    ├── bacteria_war/        # 세균 전쟁
    ├── catch_mouse/         # 쥐를 잡자
    └── betris/              # Betris 시리즈
```

---

## 참고 자료

### 게임 탐색 알고리즘 (필독)
- [Advanced Game Search Algorithms (1)](https://infossm.github.io/blog/2025/10/25/adv-game-search/) - Random → Greedy → Minimax → Alpha-Beta, SPRT 검증
- [Advanced Game Search Algorithms (2)](https://infossm.github.io/blog/2025/11/24/adv-game-search-2/) - Iterative Deepening → Transposition Table → PVS

### ALPHANO
- [ALPHANO 블로그](https://alphano.co.kr/blog) - 문제 해설 및 전략 가이드
- [ALPHANO 도움말](https://alphano.co.kr/help) - 채점 환경, 입출력 프로토콜

### 강화학습 & 게임 AI
- Sutton & Barto, *Reinforcement Learning: An Introduction* (2nd ed.)
- Silver et al., *Mastering the Game of Go with Deep Neural Networks and Tree Search* (2016)
- Silver et al., *A General Reinforcement Learning Algorithm that Masters Chess, Shogi, and Go* (2018)
