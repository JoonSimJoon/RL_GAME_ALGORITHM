# Week 5 빠른 시작 가이드

## 설치

### 1. 필요한 패키지 설치

```bash
cd /Users/simjoon/megastudy/RL_GAME_ALGORITHM/week05_rl_basics
pip install -r requirements.txt
```

필요한 패키지:
- numpy (수치 계산)
- matplotlib (시각화)

### 2. 파일 구조 확인

```
week05_rl_basics/
├── README.md              # 전체 개요
├── QUICKSTART.md          # 이 파일
├── lecture.md             # 강의 자료 (500+ lines)
├── script.md              # 수업 대본 (600+ lines)
├── requirements.txt       # 필요 패키지
└── practice/
    ├── gridworld.py       # GridWorld 환경
    ├── value_iteration.py # Value Iteration
    ├── policy_iteration.py # Policy Iteration
    └── test_all.py        # 전체 테스트
```

## 실행 방법

### Option 1: 전체 테스트 실행

모든 구현이 정상 작동하는지 확인:

```bash
cd practice
python test_all.py
```

예상 출력:
```
************************************************************
*                                                          *
*  Week 5 강화학습 기초 - 전체 구현 테스트                *
*                                                          *
************************************************************

============================================================
테스트 1: GridWorld 환경
============================================================
✓ GridWorld 생성 성공
✓ 환경 초기화 성공
...
전체: 6/6 테스트 통과

🎉 모든 테스트 통과! Week 5 구현이 완벽합니다!
```

### Option 2: GridWorld 환경 테스트

```bash
cd practice
python gridworld.py
```

GridWorld 환경의 기본 동작을 확인할 수 있습니다.

### Option 3: Value Iteration 실행

```bash
cd practice
python value_iteration.py
```

출력:
- 반복 과정 (delta 변화)
- 최적 가치 함수
- 최적 정책
- 가치 함수 히트맵 (PNG 이미지)
- 수렴 그래프
- Gamma 값 비교

생성 파일:
- `value_iteration_heatmap.png`
- `value_iteration_convergence.png`

### Option 4: Policy Iteration 실행

```bash
cd practice
python policy_iteration.py
```

출력:
- 각 반복의 정책 평가/개선
- 최적 가치 함수
- 최적 정책
- Value Iteration과 비교
- Gamma 값 비교

## 단계별 학습 가이드

### 1일차: 이론 학습 (1-2시간)

**목표**: MDP와 벨만 방정식 이해

1. `lecture.md` 읽기
   - 섹션 1-5: 강화학습 기초, MDP, 가치 함수
   - 예제 직접 계산해보기

2. 핵심 개념 정리
   - MDP 5-tuple 암기
   - 벨만 방정식 유도 연습

### 2일차: GridWorld 실습 (2-3시간)

**목표**: 환경 이해 및 수동 탐색

1. GridWorld 코드 읽기
   ```bash
   cd practice
   python
   >>> from gridworld import GridWorld
   >>> env = GridWorld()
   >>> env.render()
   ```

2. 수동으로 이동해보기
   ```python
   state = env.reset()
   next_state, reward, done = env.step(None, GridWorld.ACTION_RIGHT)
   env.render()
   ```

3. 다양한 경로 시도
   - 최단 경로 찾기
   - 장애물 피하기
   - 벽 충돌 테스트

### 3일차: Value Iteration (2-3시간)

**목표**: Value Iteration 완전 이해

1. 이론 복습
   - `lecture.md` 섹션 7 읽기
   - 벨만 최적 방정식 이해

2. 코드 분석
   - `value_iteration.py` 읽기
   - 핵심 루프 이해

3. 실행 및 분석
   ```bash
   python value_iteration.py
   ```
   - 수렴 과정 관찰
   - 히트맵 분석
   - 정책 검증

4. 실험
   - gamma 값 변경 (0.5, 0.7, 0.9, 0.99)
   - theta 값 변경
   - 장애물 위치 변경

### 4일차: Policy Iteration (2-3시간)

**목표**: Policy Iteration 이해 및 비교

1. 이론 복습
   - `lecture.md` 섹션 6 읽기
   - 정책 평가와 개선 이해

2. 코드 분석
   - `policy_iteration.py` 읽기
   - 두 단계 구조 이해

3. 실행 및 비교
   ```bash
   python policy_iteration.py
   ```
   - Value Iteration과 결과 비교
   - 수렴 속도 비교
   - 정책 변화 과정 관찰

### 5일차: 과제 및 실험 (3-4시간)

**목표**: 자유 실험 및 확장

1. 기초 과제
   - Gamma 값 실험
   - 환경 수정 (장애물, 보상)
   - 결과 분석 및 보고서

2. 도전 과제
   - 큰 격자 (6×6, 8×8)
   - 확률적 환경
   - 쥐를 잡자 간소화 버전

## 일반적인 문제 해결

### 문제 1: 수렴하지 않음

**증상**: Value Iteration이 계속 반복됨

**해결책**:
```python
# theta 값을 키우기
V, policy = value_iteration(env, gamma=0.9, theta=0.01)  # 0.001 → 0.01

# 또는 max_iterations 확인
V, policy = value_iteration(env, gamma=0.9, max_iterations=200)
```

### 문제 2: 정책이 이상함

**증상**: 정책이 목표를 향하지 않음

**가능한 원인**:
1. Gamma가 너무 작음 (근시안적)
   ```python
   V, policy = value_iteration(env, gamma=0.9)  # gamma를 높이기
   ```

2. 보상 설계 문제
   ```python
   # gridworld.py의 _get_reward() 확인
   # 이동 패널티가 너무 크지 않은지 확인
   ```

3. 장애물 배치 문제
   ```python
   env = GridWorld()
   env.render()  # 장애물 위치 확인
   ```

### 문제 3: 시각화 오류

**증상**: matplotlib 그래프가 안 뜸

**해결책**:
```bash
# matplotlib 백엔드 확인
pip install matplotlib --upgrade

# 또는 이미지만 저장
# value_iteration.py는 자동으로 PNG 파일 저장
```

### 문제 4: 너무 느림

**증상**: Policy Iteration이 매우 오래 걸림

**해결책**:
```python
# 정책 평가의 theta 값 키우기
def policy_evaluation(env, policy, gamma=0.9, theta=0.01):  # 0.001 → 0.01
    ...

# 작은 격자에서 먼저 테스트
env = GridWorld(grid_size=3)  # 4 → 3
```

## 코드 수정 가이드

### GridWorld 수정

**장애물 추가**:
```python
# gridworld.py
class GridWorld:
    def __init__(self, grid_size=4):
        ...
        self.obstacles = [(1, 1), (2, 2), (1, 3)]  # 장애물 추가
```

**보상 변경**:
```python
# gridworld.py의 _get_reward() 메서드
def _get_reward(self, state):
    if state == self.goal:
        return 10.0  # 1.0 → 10.0
    elif state in self.obstacles:
        return -10.0  # -1.0 → -10.0
    else:
        return -0.1  # -0.04 → -0.1
```

**격자 크기 변경**:
```python
env = GridWorld(grid_size=6)  # 4 → 6
```

### Value Iteration 수정

**Gamma 변경**:
```python
V, policy = value_iteration(env, gamma=0.95)  # 0.9 → 0.95
```

**수렴 조건 변경**:
```python
V, policy = value_iteration(env, gamma=0.9, theta=0.0001)  # 더 정밀
```

## 추가 실험 아이디어

### 실험 1: 보상 민감도 분석

```python
rewards = [-0.01, -0.04, -0.1, -0.5]
for r in rewards:
    # gridworld.py 수정하여 이동 보상 변경
    # Value Iteration 실행
    # 정책이 어떻게 달라지는지 관찰
```

### 실험 2: 다양한 초기 가치

```python
# value_iteration.py에서 초기 가치 변경
V = {state: 1.0 for state in env.states}  # 0.0 → 1.0
# 수렴 속도 비교
```

### 실험 3: 확률적 정책

```python
# policy_iteration.py에서 ε-greedy 정책 시도
# 최선의 행동: 90%
# 나머지 행동: 10% / 3
```

## 다음 단계

Week 5를 완료했다면:

1. **복습**: 핵심 개념 정리
   - MDP란?
   - 벨만 방정식의 의미
   - Value Iteration vs Policy Iteration

2. **심화**: 추가 학습 자료
   - Sutton & Barto 책 Chapter 4
   - David Silver Lecture 3

3. **준비**: Week 6 Q-Learning
   - Model-Free RL 개념 미리보기
   - Temporal Difference 학습
   - 쥐를 잡자 게임 준비

## 유용한 명령어 모음

```bash
# 환경 테스트
cd practice
python gridworld.py

# Value Iteration 실행
python value_iteration.py

# Policy Iteration 실행
python policy_iteration.py

# 전체 테스트
python test_all.py

# 간단한 예시만 실행 (Python 인터프리터에서)
python
>>> from value_iteration import simple_example
>>> simple_example()
```

## 도움말

질문이나 문제가 있으면:
1. README.md의 "문제 해결" 섹션 참조
2. 코드 주석 읽기
3. 수업 시간에 질문

---

**화이팅! 강화학습의 기초를 마스터하세요!** 🚀
