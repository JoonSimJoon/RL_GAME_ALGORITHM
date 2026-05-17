# CartPole + DQN 수업 자료

**대상**: 강화학습 입문자  
**예상 시간**: 45~60분  
**목표**:

1. `CartPole-v1` 환경의 상태, 행동, 보상을 이해한다.
2. `Q-Learning`만으로는 어려운 이유를 설명하고 `DQN`이 왜 필요한지 연결한다.
3. `Replay Buffer`, `Target Network`, `epsilon-greedy`의 역할을 이해한다.
4. `gymnasium_lab` 프로젝트에서 실제 학습, 평가, 렌더/녹화를 실행한다.

---

## 수업 전 준비

프로젝트 폴더로 이동합니다.

```bash
cd /Users/simjoon/megastudy/RL_GAME_ALGORITHM/gymnasium_lab
```

패키지를 설치합니다.

```bash
python -m pip install -r requirements.txt
```

짧은 스모크 테스트를 먼저 해봅니다.

```bash
python -m rl_lab.train --env cartpole --algo dqn --train-episodes 40 --eval-episodes 5
```

수업 중 바로 보여줄 기본 명령:

```bash
python -m rl_lab.train --env cartpole --algo dqn
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 5
```

GUI 렌더가 안 되면 녹화 파일로 대체합니다.

```bash
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 1 --record-path runs/cartpole_dqn_preview.gif
```

---

## 1. 도입

**교사**: 지난 시간에는 `FrozenLake + Q-Learning`으로 강화학습의 기본 구조를 봤습니다. 그때는 상태가 적어서 `Q-table`을 직접 만들 수 있었어요.

**교사**: 그런데 오늘은 `CartPole`입니다. 이 환경은 강화학습에서 거의 "Hello World"처럼 쓰이지만, 상태를 표로 다 적어놓는 방식으로는 바로 어려워집니다.

**교사**: 오늘 핵심 질문은 이겁니다.

```text
상태가 많거나 연속적이면,
Q-table 대신 무엇을 써야 할까?
```

**교사**: 답이 바로 `DQN`, `Deep Q-Network`입니다.

---

## 2. CartPole 환경 소개

**교사**: `CartPole`은 카트 위에 막대가 하나 서 있고, 이 막대가 넘어지지 않도록 카트를 좌우로 움직이는 문제입니다.

행동은 매우 단순합니다.

- `0`: 왼쪽으로 힘 주기
- `1`: 오른쪽으로 힘 주기

하지만 상태는 `FrozenLake`처럼 "현재 몇 번째 칸" 한 줄로 끝나지 않습니다.

관측값은 보통 네 개입니다.

1. 카트 위치
2. 카트 속도
3. 막대 각도
4. 막대 각속도

**교사**: 여기서 중요한 포인트는, 이 값들이 대부분 **연속값**이라는 점입니다.

예를 들어 상태가 이런 식입니다.

```text
[0.02, -0.15, 0.04, 0.31]
```

**교사**: 이런 값을 상태 하나하나 표로 저장하려고 하면 어떨까요?

**학생**: 너무 많아요.

**교사**: 맞아요. 그래서 `CartPole`은 "`Q-table`에서 신경망으로 넘어가야 하는 이유"를 보여주기에 아주 좋습니다.

---

## 3. 왜 Q-Learning만으로는 부족한가

**교사**: `FrozenLake`에서는 상태 수가 작아서 이렇게 쓸 수 있었습니다.

```text
Q[state][action]
```

**교사**: 하지만 `CartPole`은 상태가 네 개의 연속값 조합입니다. 실제 가능한 상태를 다 따지면 거의 무한대에 가깝습니다.

**교사**: 그래서 오늘부터는 생각을 바꿔야 합니다.

```text
표에 값을 저장하지 말고,
상태를 입력하면 Q값을 계산해주는 함수를 학습하자.
```

**교사**: 그 함수를 신경망으로 만들면 `Deep Q-Network`가 됩니다.

---

## 4. DQN의 핵심 아이디어

**교사**: DQN도 여전히 `Q(s, a)`를 배우는 알고리즘입니다. 달라지는 건 저장 방식입니다.

- `Q-Learning`: 표에 직접 저장
- `DQN`: 신경망이 `Q(s, a)`를 근사

`CartPole`에서는 한 상태를 넣으면 행동 2개에 대한 Q값 2개가 나옵니다.

```text
입력: state = [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
출력: [Q(state, left), Q(state, right)]
```

예를 들어 출력이 이렇게 나왔다고 해봅시다.

```text
[1.8, 2.4]
```

그러면 오른쪽 행동을 고릅니다.

```python
action = argmax(Q(state))
```

---

## 5. DQN이 바로 잘 안 되는 이유

**교사**: 그런데 신경망만 붙인다고 끝나지 않습니다. 그냥 `Q-Learning`을 신경망으로 바꾸면 학습이 매우 불안정해질 수 있습니다.

**교사**: DQN이 유명한 이유는, 두 가지 장치를 넣어서 그 불안정을 완화했기 때문입니다.

1. `Experience Replay`
2. `Target Network`

---

## 6. Experience Replay 설명

**교사**: 에이전트는 매 스텝마다 경험을 하나 얻습니다.

```text
(state, action, reward, next_state, done)
```

이걸 바로바로 학습에 쓰면 어떤 문제가 생길까요?

- 방금 본 경험들끼리 너무 비슷함
- 데이터 순서에 크게 흔들림
- 한 번의 이상한 경험에 과하게 끌려감

**교사**: 그래서 경험을 `Replay Buffer`에 저장해두고, 거기서 **랜덤 샘플**을 뽑아 학습합니다.

**교사**: 이렇게 하면 장점이 있습니다.

- 데이터가 섞여서 더 안정적임
- 같은 경험을 여러 번 재사용할 수 있음
- 현재 시점에 덜 끌려감

---

## 7. Target Network 설명

**교사**: DQN은 목표값을 만들 때도 자기 자신의 예측을 사용합니다.

핵심 식은 이렇습니다.

```text
target = reward + gamma * max_a' Q_target(next_state, a')
```

**교사**: 만약 목표를 만드는 네트워크와 지금 학습 중인 네트워크가 완전히 같은 녀석이면, 기준이 계속 흔들립니다.

**교사**: 그래서 DQN은 네트워크를 둘로 둡니다.

- `q_network`: 실제로 계속 업데이트되는 현재 네트워크
- `target_network`: 일정 간격으로만 복사되는 느린 기준 네트워크

**교사**: 즉, 공부하는 학생과 채점 기준을 완전히 같은 속도로 바꾸지 않는 셈입니다.

---

## 8. epsilon-greedy 설명

**교사**: DQN도 처음부터 똑똑하지 않습니다. 그래서 탐험이 필요합니다.

행동 선택은 보통 이렇게 합니다.

- 확률 `epsilon`으로 랜덤 행동
- 확률 `1 - epsilon`으로 현재 Q값이 가장 큰 행동

**교사**: 초반에는 `epsilon`이 크고, 나중에는 작아지게 둡니다.

**교사**: 우리 프로젝트 기본 설정도 그렇게 되어 있습니다.

`configs/cartpole/dqn.yaml`

```yaml
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_decay: 0.985
```

**교사**: 즉, 처음에는 거의 아무거나 해보면서 배우고, 뒤로 갈수록 점점 배운 걸 더 믿는 구조입니다.

---

## 9. 오늘 코드에서 꼭 볼 부분

오늘 실습 코드는 이 파일을 중심으로 봅니다.

`rl_lab/algorithms/value_based/dqn.py`

### 9-1. `ReplayBuffer`

**교사**: 먼저 경험을 저장하는 통입니다.

- `append(...)`: 경험 추가
- `sample(batch_size)`: 랜덤으로 배치 뽑기

**교사**: 여기서 학생들이 알아야 할 핵심은 "`학습 데이터가 한 줄씩 들어오더라도, 신경망은 미니배치로 학습한다`"는 점입니다.

### 9-2. `_select_action`

**교사**: 행동 선택 함수입니다.

- `epsilon`보다 작은 랜덤 값이면 무작위 행동
- 아니면 네트워크가 예측한 Q값 중 최대값 행동

**교사**: 이 부분이 `exploration vs exploitation`을 직접 보여줍니다.

### 9-3. `_optimize`

**교사**: DQN의 핵심 업데이트가 들어 있습니다.

여기서 하는 일:

1. Replay Buffer에서 배치를 샘플링
2. 현재 `q_network`의 예측 Q값 계산
3. `target_network`로 목표값 계산
4. 손실 계산
5. 역전파로 신경망 업데이트

**교사**: 손실 함수는 `smooth_l1_loss`를 사용하고 있습니다. `MSE`보다 이상치에 조금 더 안정적인 편이라 DQN 계열에서 자주 씁니다.

### 9-4. `train`

**교사**: 실제 학습 루프입니다.

- 매 에피소드마다 환경 초기화
- 매 스텝마다 행동 선택
- 경험 저장
- `_optimize()` 호출
- 일정 스텝마다 `target_network` 동기화
- 일정 에피소드마다 평가

**교사**: 이 흐름을 보면서 학생들에게 "강화학습도 결국 데이터 수집 + 모델 업데이트의 반복"이라는 감각을 잡아주면 좋습니다.

---

## 10. 실습 진행 순서

### 10-1. 기본 학습 실행

```bash
python -m rl_lab.train --env cartpole --algo dqn
```

**교사**: 이 명령이 끝나면 `runs/cartpole/dqn/...` 폴더 아래에 결과가 저장됩니다.

생성되는 대표 파일:

- `config.yaml`
- `checkpoint.pt`
- `train_metrics.csv`
- `summary.json`
- `learning_curve.png`

### 10-2. 학습 로그 해석

**교사**: 터미널에 보이는 숫자 중 중요한 건 두 가지입니다.

- `Final eval return`
- `Elapsed seconds`

**교사**: `CartPole-v1`은 오래 버틸수록 점수가 커집니다. 즉, return이 커지는 방향이면 학습이 잘 되고 있는 겁니다.

대략적인 감각:

- 랜덤 정책: 보통 낮은 점수
- 초반 학습: 점수가 들쭉날쭉함
- 학습 진행 후: 평균 return이 점점 올라감

### 10-3. 최신 런 평가

```bash
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 5
```

### 10-4. GUI 렌더 또는 녹화

GUI가 가능한 환경:

```bash
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 1 --render
```

GUI가 안 되면:

```bash
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 1 --record-path runs/cartpole_dqn_preview.gif
```

---

## 11. 수업 중 강조할 비교 포인트

### 11-1. FrozenLake vs CartPole

- `FrozenLake`: 이산 상태, 표 기반 학습이 가능
- `CartPole`: 연속 상태, 표 기반 접근이 비현실적

### 11-2. Q-Learning vs DQN

- `Q-Learning`: `Q[state][action]`
- `DQN`: `Q(state, action)`를 신경망으로 근사

### 11-3. 왜 DQN이 더 복잡한가

- 신경망이 필요함
- Replay Buffer가 필요함
- Target Network가 필요함
- 하이퍼파라미터 영향을 더 크게 받음

---

## 12. 학생 질문 유도용 멘트

아래 질문을 던지면 토론이 잘 됩니다.

1. `CartPole`에서 상태를 표로 저장하면 왜 어려울까요?
2. 랜덤 행동만 하면 평균적으로 왜 오래 못 버틸까요?
3. 경험을 순서대로 바로 학습하는 것보다 섞어서 학습하는 게 왜 좋을까요?
4. 목표값을 만드는 네트워크까지 계속 같이 바꾸면 왜 불안정할까요?
5. `epsilon`이 너무 빨리 줄어들면 어떤 문제가 생길까요?

---

## 13. 자주 나오는 질문과 답변

### Q1. DQN은 결국 무엇을 배우는 건가요?

**답변**: 상태를 보면 각 행동이 얼마나 좋은지 점수처럼 예측하는 함수를 배웁니다. `CartPole`에서는 "지금 이 상태에서 왼쪽이 좋은지, 오른쪽이 좋은지"를 숫자로 예측합니다.

### Q2. DQN은 왜 `CartPole`에 잘 맞나요?

**답변**: 행동은 2개뿐이라 이산 행동 문제이고, 상태는 연속값이라 표 기반 방법은 어렵습니다. 즉, DQN이 딱 필요한 조합입니다.

### Q3. DQN이 모든 강화학습 문제에 만능인가요?

**답변**: 아닙니다. DQN은 기본적으로 이산 행동 문제에 적합합니다. 연속 행동 문제는 `Actor-Critic`, `DDPG`, `SAC`, `PPO` 같은 다른 계열이 더 자연스럽습니다.

### Q4. 점수가 오르다가 다시 내려갈 수도 있나요?

**답변**: 가능합니다. 강화학습은 지도학습보다 변동성이 큰 편입니다. 그래서 한 번의 평가보다 여러 평가 지점과 평균 추세를 같이 보는 게 좋습니다.

---

## 14. 확장 활동

오늘 수업이 끝난 뒤 이어서 해볼 만한 활동:

1. `configs/cartpole/dqn.yaml`에서 `epsilon_decay`를 바꾸고 학습 속도를 비교해보기
2. `target_update_interval`을 크게 또는 작게 바꿔보기
3. `batch_size`를 바꿔서 손실과 점수 변화를 비교해보기
4. `python -m rl_lab.compare --env cartpole --algos dqn reinforce a2c`로 다른 알고리즘과 비교해보기
5. `CartPole`에서 잘 되던 방법이 `Pendulum`에서는 왜 바로 안 통하는지 토론해보기

---

## 15. 마무리 정리

**교사**: 오늘 기억해야 할 한 줄은 이겁니다.

```text
DQN은 Q-Learning의 아이디어를 유지하면서,
Q-table 대신 신경망으로 큰 상태 공간을 다루는 방법이다.
```

**교사**: 그리고 DQN을 실전에서 가능하게 만드는 핵심 장치는 세 가지입니다.

1. 신경망으로 Q값 근사
2. Replay Buffer로 경험 재사용
3. Target Network로 학습 안정화

**교사**: 오늘 `CartPole`에서 이 구조를 이해하면, 다음에는 더 복잡한 환경과 더 발전된 알고리즘으로 자연스럽게 넘어갈 수 있습니다.

---

## 부록: 진행자 치트시트

빠르게 실행할 핵심 명령만 다시 모아두면:

```bash
cd /Users/simjoon/megastudy/RL_GAME_ALGORITHM/gymnasium_lab
python -m pip install -r requirements.txt
python -m rl_lab.train --env cartpole --algo dqn
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 5
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 1 --render
python -m rl_lab.evaluate --env cartpole --algo dqn --num-episodes 1 --record-path runs/cartpole_dqn_preview.gif
python -m rl_lab.compare --env cartpole --algos dqn reinforce a2c
```

수업 중 꼭 짚을 파일:

- `configs/cartpole/dqn.yaml`
- `rl_lab/algorithms/value_based/dqn.py`
- `runs/cartpole/dqn/.../train_metrics.csv`
- `runs/cartpole/dqn/.../learning_curve.png`
