# Gymnasium Lab Architecture

## 1. 목표

이 프로젝트는 "주차 자료"가 아니라 "반복 가능한 실험실"을 만드는 것이 목적입니다.

필요한 기능은 다음과 같습니다.

1. 환경별로 여러 RL 알고리즘을 붙여서 비교할 수 있어야 한다.
2. 알고리즘별 공통 학습 루프와 결과 저장 포맷이 있어야 한다.
3. 새 환경과 새 알고리즘을 적은 수정으로 추가할 수 있어야 한다.
4. 강의용 실습과 연구용 작은 벤치마크를 동시에 지원할 수 있어야 한다.

## 2. 핵심 설계 결정

### 2.1 사용자 경험은 Environment-First

사용자는 보통 이렇게 생각합니다.

- "FrozenLake에서 Q-Learning과 SARSA를 비교하고 싶다."
- "CartPole에서 DQN과 REINFORCE를 비교하고 싶다."
- "Pendulum에서 연속 행동 정책을 학습시키고 싶다."

따라서 실험 진입점은 `env + algo` 조합이 되어야 합니다.

### 2.2 내부 구현은 Algorithm-Reusable

코드는 반대로 구성하는 편이 좋습니다.

- `Q-Learning`은 여러 작은 이산 환경에 재사용 가능
- `DQN`은 여러 이산 관측/행동 환경에 재사용 가능
- `REINFORCE`와 `Actor-Critic`은 정책 네트워크만 바꾸면 확장 가능

즉, 바깥은 환경 중심, 안쪽은 알고리즘 재사용 중심으로 나누는 것이 가장 안정적입니다.

## 3. 권장 디렉토리 구조

```text
gymnasium_lab/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── frozenlake/
│   │   ├── q_learning.yaml
│   │   └── sarsa.yaml
│   ├── cartpole/
│   │   ├── dqn.yaml
│   │   ├── reinforce.yaml
│   │   └── a2c.yaml
│   └── pendulum/
│       ├── reinforce.yaml
│       └── actor_critic.yaml
├── rl_lab/
│   ├── __init__.py
│   ├── core/
│   │   ├── registry.py
│   │   ├── config.py
│   │   ├── experiment.py
│   │   └── interfaces.py
│   ├── algorithms/
│   │   ├── tabular/
│   │   │   ├── q_learning.py
│   │   │   └── sarsa.py
│   │   ├── value_based/
│   │   │   └── dqn.py
│   │   └── policy_based/
│   │       ├── reinforce_discrete.py
│   │       ├── reinforce_continuous.py
│   │       └── actor_critic.py
│   ├── envs/
│   │   ├── specs.py
│   │   ├── factory.py
│   │   └── wrappers.py
│   ├── runners/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── compare.py
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       ├── metrics.py
│       └── plotting.py
└── runs/
    └── <env>/<algo>/<timestamp>_seed<seed>/
```

## 4. 공통 인터페이스

알고리즘마다 학습 코드는 달라도, 외부에서 다루는 방식은 같아야 합니다.

현재 구현은 아래 인터페이스를 기준으로 정리되어 있습니다.

```python
class RLAlgorithm:
    def train(self, run_dir): ...
    def evaluate(self, checkpoint_path, num_episodes=10): ...
    def save(self, checkpoint_path): ...
    def load(self, checkpoint_path): ...
```

여기서 중요한 점:

- `tabular` 알고리즘도 같은 인터페이스를 따라갑니다.
- `deep` 알고리즘도 같은 방식으로 저장/평가합니다.
- CLI는 내부 구현 차이를 몰라도 됩니다.

## 5. 환경 메타데이터 설계

환경마다 가능한 알고리즘이 다르므로 메타정보를 분리하는 것이 좋습니다.

예시:

```python
EnvSpec(
    key="cartpole",
    gym_id="CartPole-v1",
    observation_type="continuous_vector",
    action_type="discrete",
    compatible_algorithms=["dqn", "reinforce", "a2c"],
    reward_metric="episode_return",
    solved_threshold=475.0,
)
```

이 메타데이터를 두면 다음이 쉬워집니다.

- 잘못된 조합 차단
- 기본 하이퍼파라미터 자동 선택
- 평가 기준 통일

예를 들어 `Pendulum-v1`에 순수 DQN을 바로 붙이려 하면 CLI 단계에서 막을 수 있습니다.

## 6. 설정 파일 전략

실험은 코드 수정 대신 설정 파일로 바꾸는 쪽이 좋습니다.

설정 파일에는 아래 정도가 들어가면 충분합니다.

- 환경 이름
- 알고리즘 이름
- seed
- 에피소드 수
- 학습률
- 감가율
- 탐험률 또는 entropy 계수
- 네트워크 크기
- 평가 주기
- 저장 주기

예시:

```yaml
env: cartpole
algo: dqn
seed: 42
train_episodes: 500
eval_interval: 25

algo_params:
  gamma: 0.99
  lr: 0.001
  batch_size: 64
  buffer_size: 10000
  target_update_interval: 200
  epsilon_start: 1.0
  epsilon_end: 0.05
  epsilon_decay: 0.995

network:
  hidden_sizes: [128, 128]
```

## 7. 결과 저장 규칙

결과가 뒤섞이면 실험 프로젝트는 금방 망가집니다. 저장 규칙을 초반에 고정하는 것이 중요합니다.

권장 구조:

```text
runs/
└── cartpole/
    └── dqn/
        └── 20260503_160000_seed42/
            ├── config.yaml
            ├── train_metrics.csv
            ├── eval_metrics.json
            ├── checkpoint.pt
            └── learning_curve.png
```

필수 산출물:

- 실행에 사용한 설정 파일 복사본
- episode 단위 학습 로그
- 최종 평가 결과
- 모델 체크포인트
- 기본 학습 곡선 이미지

## 8. 환경별 권장 알고리즘 매트릭스

### 8.1 Phase 1

| 환경 | 추천 알고리즘 | 이유 |
|---|---|---|
| `FrozenLake-v1` | Q-Learning, SARSA | 표 기반 RL 비교에 적합 |
| `CartPole-v1` | DQN, REINFORCE | 이산 행동에서 value/policy 기반 비교 가능 |
| `Pendulum-v1` | Gaussian REINFORCE, Actor-Critic | 연속 행동 입문용 |

### 8.2 Phase 2

| 환경 | 추가 알고리즘 | 비고 |
|---|---|---|
| `Acrobot-v1` | DQN, A2C | 더 어려운 이산 제어 |
| `MountainCar-v0` | DQN, REINFORCE | sparse reward 실험 |
| `LunarLander-v3` | DQN, A2C | Box2D 의존성 별도 관리 필요 |

처음에는 `classic-control` 중심으로 시작하고, 외부 의존성이 큰 환경은 나중에 넣는 편이 안전합니다.

## 9. 실행 흐름

### 9.1 학습

```bash
python -m rl_lab.train --config configs/cartpole/dqn.yaml
python -m rl_lab.train --env cartpole --algo dqn --seed 42
```

### 9.2 평가

```bash
python -m rl_lab.evaluate --run runs/cartpole/dqn/20260503_160000_seed42
```

### 9.3 비교

```bash
python -m rl_lab.compare --env cartpole --algos dqn reinforce a2c
```

비교 스크립트는 최소한 아래를 그려주면 충분합니다.

- 평균 return 곡선
- seed별 최종 성능 표
- 학습 속도 비교

## 10. 구현 상태

현재 완료된 항목:

- `FrozenLake-v1`용 `Q-Learning`, `SARSA`
- `CartPole-v1`용 `DQN`, `REINFORCE`, `A2C`
- `Pendulum-v1`용 Gaussian `REINFORCE`, `Actor-Critic`
- 공통 `train`, `evaluate`, `compare` CLI
- `runs/` 기반 결과 저장
- 학습 곡선과 비교 그래프 저장

## 11. 확장 우선순위

### 단계 1: 기반 공사

- `requirements.txt`
- 기본 폴더 구조
- `EnvSpec`와 registry
- 공통 seed/logging 유틸
- `train/evaluate` CLI 뼈대

### 단계 2: 다음 환경 추가

- `Acrobot-v1`
- `MountainCar-v0`
- `LunarLander-v3`
- 환경별 기본 config 추가

### 단계 3: 알고리즘 확장

- Double DQN
- PPO
- N-step Actor-Critic
- Vectorized environment 지원

## 12. 현재 레포와의 관계

이 프로젝트는 기존 `week05`, `week06`, `week07` 내용을 대체하기보다, 그 실습을 재조합하는 "통합 실험실" 역할을 합니다.

정리하면:

- `weekXX_*` 폴더는 강의 자료 보관
- `gymnasium_lab/`는 실제 실험과 비교 프로젝트

이렇게 분리하면 강의 흐름과 실험 코드가 서로 덜 엉킵니다.

## 13. 다음 작업 추천

설계 다음 순서로는 아래가 가장 효율적입니다.

1. `compare`를 여러 seed 평균 비교까지 확장
2. 환경별 기본 하이퍼파라미터 자동 탐색 스크립트 추가
3. 체크포인트 재개 학습 기능 추가
4. `LunarLander`와 `MountainCar` 벤치마크 추가
5. 결과 Markdown 리포트 자동 생성

지금 단계에서는 "확장 가능한 실험 구조"를 먼저 고정하는 것이 핵심입니다.
