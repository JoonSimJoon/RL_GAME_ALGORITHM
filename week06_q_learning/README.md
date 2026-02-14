# Week 6: Q-Learning과 SARSA

## 개요

이번 주차에서는 **Model-free 강화학습**의 기초인 Q-Learning과 SARSA 알고리즘을 배웁니다.

### 학습 목표
- 환경 모델 없이 경험으로 학습하는 방법 이해
- Q-Learning과 SARSA 알고리즘 구현
- ε-greedy 탐험 전략 이해
- 하이퍼파라미터의 영향 분석
- FrozenLake 환경에서 실습

---

## 파일 구조

```
week06_q_learning/
├── README.md                          # 이 파일
├── lecture.md                         # 수업 자료 (500+ lines)
├── script.md                          # 수업 대본 (90분)
├── practice/                          # 실습 코드
│   ├── q_learning_frozenlake.py      # Q-Learning 기본 구현
│   ├── sarsa_frozenlake.py           # SARSA 구현 및 비교
│   └── hyperparameter_experiment.py  # 하이퍼파라미터 실험
└── alphano/                           # ALPHANO 대회 에이전트
    └── catch_mouse_heuristic.py      # 쥐를 잡자 휴리스틱 에이전트
```

---

## 준비 사항

### 필수 라이브러리 설치

```bash
pip install gymnasium numpy matplotlib
```

### 환경 확인

```python
import gymnasium as gym
env = gym.make('FrozenLake-v1')
print("설치 완료!")
env.close()
```

---

## 실습 순서

### 1단계: Q-Learning 기초 (30분)

**파일**: `practice/q_learning_frozenlake.py`

```bash
python practice/q_learning_frozenlake.py
```

**학습 내용**:
- Q-table 기반 Q-Learning 구현
- ε-greedy 탐험 전략
- FrozenLake 환경 학습
- 학습 곡선 관찰
- 학습된 정책 시각화

**실습 과제**:
1. 기본 코드 실행 및 결과 확인
2. `is_slippery=True`로 변경하여 실험
3. 성공률 90% 이상 달성하기

### 2단계: SARSA 비교 (20분)

**파일**: `practice/sarsa_frozenlake.py`

```bash
python practice/sarsa_frozenlake.py
```

**학습 내용**:
- SARSA 알고리즘 구현
- Q-Learning과의 차이점 이해
- On-policy vs Off-policy
- 두 알고리즘의 학습 곡선 비교

**실습 과제**:
1. Q-Learning과 SARSA 성능 비교
2. 학습된 정책 비교
3. 어떤 상황에서 어떤 알고리즘이 유리한지 분석

### 3단계: 하이퍼파라미터 실험 (30분)

**파일**: `practice/hyperparameter_experiment.py`

```bash
python practice/hyperparameter_experiment.py
```

**학습 내용**:
- α (학습률)의 영향
- γ (할인율)의 영향
- ε_decay (탐험률 감소)의 영향
- 그리드 서치로 최적 조합 찾기

**실습 과제**:
1. 각 파라미터 변화에 따른 학습 곡선 분석
2. 최적 파라미터 조합 찾기
3. 확률적 환경(`is_slippery=True`)에서 실험

### 4단계: ALPHANO 도전 (선택, 30분)

**파일**: `alphano/catch_mouse_heuristic.py`

```bash
python alphano/catch_mouse_heuristic.py
```

**학습 내용**:
- 쥐를 잡자 게임 규칙 이해
- 간단한 휴리스틱 전략 구현
- ALPHANO 프로토콜 이해

**실습 과제**:
1. 휴리스틱 에이전트 실행 및 분석
2. 더 나은 전략 설계
3. (도전) Q-Learning 적용하기

---

## 핵심 개념 요약

### Q-Learning

**업데이트 수식**:
```
Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
```

**특징**:
- Off-policy: 행동 정책과 학습 정책이 다름
- max Q 사용: 최적 정책 학습
- 공격적, 위험 감수

### SARSA

**업데이트 수식**:
```
Q(s,a) ← Q(s,a) + α[R + γ·Q(s',a') - Q(s,a)]
```

**특징**:
- On-policy: 행동 정책과 학습 정책이 같음
- 실제 선택한 행동의 Q값 사용
- 보수적, 안전

### ε-Greedy

```python
if random() < ε:
    action = random_action()  # 탐험
else:
    action = best_action()     # 활용
```

**ε-decay**:
```python
ε = max(ε_min, ε * decay_rate)
```

### 하이퍼파라미터

| 파라미터 | 의미 | 권장값 |
|---------|------|--------|
| α | 학습률 | 0.1 |
| γ | 할인율 | 0.99 |
| ε_start | 초기 탐험률 | 1.0 |
| ε_min | 최소 탐험률 | 0.01 |
| ε_decay | 탐험률 감소 | 0.995 |

---

## Q&A

### Q1. Q-Learning이 항상 최적 정책을 찾나요?

**A**: 이론적으로는 무한히 학습하면 최적 정책에 수렴합니다. 하지만 실전에서는:
- 충분한 탐험 필요
- 적절한 하이퍼파라미터 선택
- 충분한 에피소드 수

조건이 맞으면 수렴합니다!

### Q2. Q-Learning과 SARSA 중 뭘 선택해야 하나요?

**A**: 상황에 따라 다릅니다:

**Q-Learning 선택**:
- 시뮬레이션 환경
- 실패해도 괜찮음
- 최적 성능 추구

**SARSA 선택**:
- 실제 환경 (로봇 등)
- 안전이 중요
- 학습 중 사고 방지

### Q3. Q-table의 한계는?

**A**:
- 큰 상태공간 처리 불가 (바둑: 10^170 상태)
- 연속 상태 처리 불가
- 일반화 능력 부족

→ 다음 주에 배울 **DQN**이 해결책!

### Q4. 하이퍼파라미터는 어떻게 선택하나요?

**A**:
1. 기본값으로 시작 (α=0.1, γ=0.99, decay=0.995)
2. 환경에서 테스트
3. 성능이 안 좋으면:
   - 학습이 느리면 α 증가
   - 장기 계획 필요하면 γ 증가
   - 탐험 부족하면 decay 감소
4. 그리드 서치로 최적값 탐색

---

## 추가 자료

### 참고 문헌

- **Sutton & Barto**, "Reinforcement Learning: An Introduction" (2nd ed.)
  - Chapter 6: Temporal-Difference Learning

### 온라인 자료

- [OpenAI Spinning Up](https://spinningup.openai.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [ALPHANO Platform](https://alphano.kr/)

### 다음 주 예고

**Week 7: DQN (Deep Q-Network)**
- Q-table → Q-network (신경망)
- Experience Replay
- Target Network
- Atari 게임 학습

**준비물**:
```bash
pip install torch gymnasium[atari] gymnasium[accept-rom-license]
```

---

## 과제

### 필수 과제

1. **FrozenLake Q-Learning**:
   - `is_slippery=False`에서 성공률 95% 이상
   - `is_slippery=True`에서 성공률 80% 이상
   - 학습 곡선 그래프 제출

2. **SARSA vs Q-Learning 비교**:
   - 두 알고리즘의 학습 곡선 비교
   - 학습된 정책 시각화 비교
   - 차이점 분석 (200자 이상)

3. **하이퍼파라미터 실험**:
   - α, γ, ε_decay 각각 3개 이상 값으로 실험
   - 결과 그래프 및 분석
   - 최적 조합 제시 및 이유 설명

### 선택 과제

1. **쥐를 잡자 개선**:
   - 휴리스틱 전략 개선
   - (도전) Q-Learning 적용

2. **다른 환경 실습**:
   - CliffWalking-v0 환경 실험
   - Taxi-v3 환경 실험

3. **연구 과제**:
   - Double Q-Learning 조사 및 구현
   - Expected SARSA 조사 및 구현

---

## 문제 해결

### 학습이 너무 느려요

1. α 증가 (0.1 → 0.3)
2. 에피소드 수 증가
3. ε_decay 증가 (0.995 → 0.99)

### 성공률이 낮아요

1. γ 증가 (0.9 → 0.99)
2. 에피소드 수 증가
3. ε_decay 감소 (더 많이 탐험)

### 학습이 불안정해요

1. α 감소 (0.3 → 0.1)
2. 여러 번 실행하여 평균 확인
3. ε_decay 감소 (천천히 수렴)

---

## 연락처

질문이나 피드백은 아래로 연락주세요:
- 강의 게시판
- 이메일

---

**화이팅! 다음 주에 만나요!**
