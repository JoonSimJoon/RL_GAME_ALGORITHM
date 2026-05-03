# Gymnasium Lab

`Gymnasium Lab`은 주차별 실습 폴더 대신, 여러 환경에서 여러 강화학습 알고리즘을 비교 실험하기 위한 독립 프로젝트입니다.

핵심 목표는 세 가지입니다.

1. 같은 환경에서 알고리즘을 바꿔가며 성능을 비교한다.
2. 같은 알고리즘을 여러 환경에 적용하며 일반화한다.
3. 실행, 평가, 결과 저장 방식을 통일해서 반복 실험을 쉽게 만든다.

## 왜 별도 프로젝트로 분리하나?

기존 레포의 `weekXX_*` 폴더는 강의 흐름에는 좋지만, 실험을 반복하고 확장하기에는 불편합니다.

- 환경별 실험 이력이 흩어진다.
- 알고리즘 구현을 재사용하기 어렵다.
- 결과 비교 형식이 통일되지 않는다.

그래서 이 프로젝트는 `environment-first` 관점으로 정리합니다.

## 현재 구현 범위

현재 아래 조합이 기본 설정과 함께 구현되어 있습니다.

| 환경 | 행동 공간 | 1차 알고리즘 | 2차 알고리즘 |
|---|---|---|---|
| `FrozenLake-v1` | 이산 | Q-Learning | SARSA |
| `CartPole-v1` | 이산 | DQN | REINFORCE, A2C |
| `Pendulum-v1` | 연속 | Gaussian REINFORCE | Actor-Critic |

이 구성이 좋은 이유:

- 난이도가 단계적으로 올라갑니다.
- 표 기반 RL, value 기반 deep RL, policy 기반 RL을 모두 다룹니다.
- `Gymnasium`의 `classic-control` 범위 안에서 시작할 수 있어 설치가 비교적 쉽습니다.

## 실제 구조

현재 프로젝트는 아래 구조로 동작합니다.

```text
gymnasium_lab/
├── README.md
├── ARCHITECTURE.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── frozenlake/
│   ├── cartpole/
│   └── pendulum/
├── rl_lab/
│   ├── core/
│   ├── algorithms/
│   ├── envs/
│   ├── runners/
│   └── utils/
├── scripts/
└── runs/
```

## 설계 원칙

- 사용자 입장에서는 환경 중심으로 진입합니다.
- 내부 구현은 알고리즘 재사용이 쉽도록 공통 인터페이스를 둡니다.
- 모든 실험은 `config + seed + 결과 로그`를 남깁니다.
- 새 환경이나 새 알고리즘을 추가할 때 기존 코드를 거의 건드리지 않게 만듭니다.

## 빠른 시작

먼저 프로젝트 디렉토리로 들어갑니다.

```bash
cd gymnasium_lab
```

의존성을 설치합니다.

```bash
python -m pip install -r requirements.txt
```

학습을 실행합니다.

```bash
python -m rl_lab.train --env frozenlake --algo q_learning
python -m rl_lab.train --env cartpole --algo dqn --seed 42
python -m rl_lab.train --env cartpole --algo reinforce
python -m rl_lab.train --env cartpole --algo a2c
python -m rl_lab.train --env pendulum --algo actor_critic
```

평가를 실행합니다.

```bash
python -m rl_lab.evaluate --env cartpole --algo dqn
python -m rl_lab.evaluate --run runs/cartpole/dqn/<timestamp>_seed42
```

비교를 실행합니다.

```bash
python -m rl_lab.compare --env cartpole --algos dqn reinforce a2c
```

## 이번 설계에서 정한 중요한 방향

- `openai/gym` 대신 `gymnasium` 기준으로 맞춥니다.
- 주차별 폴더 아래에 코드를 흩뿌리지 않고 최상단 독립 프로젝트로 둡니다.
- 환경별 학습 자료는 `configs/`와 환경별 문서에서 관리합니다.
- 알고리즘 코드는 공통 인터페이스로 묶어서 재사용합니다.

## 생성되는 결과물

학습이 끝나면 결과는 `runs/<env>/<algo>/<timestamp>_seed<seed>/` 아래에 저장됩니다.

- `config.yaml`: 실행에 사용한 설정 복사본
- `checkpoint.pt` 또는 `checkpoint.npz`: 저장된 모델
- `train_metrics.csv`: episode 단위 학습 로그
- `eval_metrics.json`: 최종 평가 결과
- `summary.json`: 요약 정보
- `learning_curve.png`: 학습 곡선

`compare`를 실행하면 `runs/<env>/comparisons/<timestamp>/` 아래에 비교 그래프와 요약 JSON이 생성됩니다.

자세한 구조와 단계별 구현 계획은 [ARCHITECTURE.md](/Users/simjoon/megastudy/RL_GAME_ALGORITHM/gymnasium_lab/ARCHITECTURE.md)에서 정리합니다.
