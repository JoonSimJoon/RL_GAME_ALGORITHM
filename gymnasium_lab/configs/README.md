# Config Layout

환경별 설정 파일을 이 폴더 아래에 둡니다.

예상 구조:

```text
configs/
├── frozenlake/
│   ├── q_learning.yaml
│   ├── q_learning_4x4.yaml
│   ├── q_learning_8x8.yaml
│   ├── q_learning_custom.yaml
│   └── sarsa.yaml
├── cartpole/
│   ├── dqn.yaml
│   ├── reinforce.yaml
│   └── a2c.yaml
└── pendulum/
    ├── reinforce_gaussian.yaml
    └── actor_critic.yaml
```

원칙:

- 설정 파일은 `env + algo` 조합 단위로 나눕니다.
- 실행 중 바뀐 값은 결과 폴더에 복사 저장합니다.
- 코드 수정 없이 실험 반복이 가능해야 합니다.

FrozenLake 예시:

```bash
python -m rl_lab.train --config configs/frozenlake/q_learning_4x4.yaml
python -m rl_lab.train --config configs/frozenlake/q_learning_8x8.yaml
python -m rl_lab.train --config configs/frozenlake/q_learning_custom.yaml
```
