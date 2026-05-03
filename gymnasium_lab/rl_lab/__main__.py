from __future__ import annotations

import textwrap


def main() -> int:
    print(
        textwrap.dedent(
            """
            Gymnasium Lab

            Entry points:
              python -m rl_lab.train --env frozenlake --algo q_learning
              python -m rl_lab.evaluate --env frozenlake --algo q_learning
              python -m rl_lab.compare --env cartpole --algos dqn reinforce a2c
            """
        ).strip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
