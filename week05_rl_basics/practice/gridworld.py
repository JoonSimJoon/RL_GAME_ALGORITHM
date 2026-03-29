"""
GridWorld 환경 구현
4x4 격자 세계에서 에이전트가 목표를 찾아가는 간단한 MDP 환경
"""


class GridWorld:
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_NAMES = ["↑", "↓", "←", "→"]

    def __init__(self, grid_size=4):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.obstacles = [(1, 1), (2, 2)]
        self.current_state = self.start
        self.n_actions = 4
        self.states = [(r, c) for r in range(grid_size) for c in range(grid_size)]

    def reset(self):
        self.current_state = self.start
        return self.current_state

    def step(self, action, state=None):
        """상태 전이 수행. state가 주어지면 해당 상태에서 전이, 아니면 current_state 사용."""
        curr = state if state is not None else self.current_state
        next_state = self.get_next_state(curr, action)
        reward = self.get_reward(next_state)
        done = self.is_terminal(next_state)
        if state is None:
            self.current_state = next_state
        return next_state, reward, done

    def get_next_state(self, state, action):
        row, col = state
        if action == self.ACTION_UP:
            row -= 1
        elif action == self.ACTION_DOWN:
            row += 1
        elif action == self.ACTION_LEFT:
            col -= 1
        elif action == self.ACTION_RIGHT:
            col += 1
        else:
            raise ValueError(f"Invalid action: {action}")

        if not self.is_valid_position(row, col):
            return state
        return (row, col)

    def is_valid_position(self, row, col):
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def get_reward(self, state):
        if state == self.goal:
            return 1.0
        elif state in self.obstacles:
            return -1.0
        else:
            return -0.04

    def is_terminal(self, state):
        return state == self.goal or state in self.obstacles

    def get_possible_actions(self, state):
        if self.is_terminal(state):
            return []
        return [0, 1, 2, 3]

    def render(self, show_agent=True):
        print()
        print("=" * (self.grid_size * 4 + 1))
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                pos = (r, c)
                if show_agent and pos == self.current_state:
                    cell = " A "
                elif pos == self.goal:
                    cell = " G "
                elif pos in self.obstacles:
                    cell = " X "
                elif pos == self.start:
                    cell = " S "
                else:
                    cell = "   "
                row_str += cell + "|"
            print(row_str)
            print("=" * (self.grid_size * 4 + 1))
        print()

    def render_policy(self, policy):
        print("\n정책 시각화:")
        print("=" * (self.grid_size * 4 + 1))
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                pos = (r, c)
                if pos == self.goal:
                    cell = " G "
                elif pos in self.obstacles:
                    cell = " X "
                elif pos in policy:
                    cell = " " + self.ACTION_NAMES[policy[pos]] + " "
                else:
                    cell = "   "
                row_str += cell + "|"
            print(row_str)
            print("=" * (self.grid_size * 4 + 1))
        print()

    def render_values(self, values):
        print("\n가치 함수 시각화:")
        print("=" * (self.grid_size * 7 + 1))
        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                pos = (r, c)
                if pos in values:
                    row_str += f"{values[pos]:6.2f}|"
                else:
                    row_str += "  N/A |"
            print(row_str)
            print("=" * (self.grid_size * 7 + 1))
        print()


def test_gridworld():
    print("=== GridWorld 테스트 ===\n")
    env = GridWorld(4)

    print("초기 상태:")
    env.render()

    print("\n행동 시퀀스 테스트:")
    actions = [
        (GridWorld.ACTION_RIGHT, "오른쪽"),
        (GridWorld.ACTION_RIGHT, "오른쪽"),
        (GridWorld.ACTION_DOWN, "아래"),
        (GridWorld.ACTION_DOWN, "아래"),
    ]

    total_reward = 0
    for action, action_name in actions:
        next_state, reward, done = env.step(action)
        total_reward += reward

        print(f"\n행동: {action_name}")
        print(f"다음 상태: {next_state}")
        print(f"보상: {reward:.2f}")
        print(f"종료: {done}")
        print(f"누적 보상: {total_reward:.2f}")
        env.render()

        if done:
            print("에피소드 종료!")
            break

    # 장애물 테스트
    print("\n\n=== 장애물 테스트 ===")
    env.reset()
    print("\n초기 위치에서 아래로 이동 (장애물):")
    next_state, reward, done = env.step(GridWorld.ACTION_DOWN, state=(0, 0))
    print(f"다음 상태: {next_state}")
    print(f"보상: {reward:.2f}")
    print(f"종료: {done}")

    # 벽 테스트
    print("\n\n=== 벽 테스트 ===")
    print("(0, 0)에서 위로 이동 (벽):")
    next_state, reward, done = env.step(GridWorld.ACTION_UP, state=(0, 0))
    print(f"다음 상태: {next_state} (제자리)")
    print(f"보상: {reward:.2f}")

    print("\n(0, 0)에서 왼쪽으로 이동 (벽):")
    next_state, reward, done = env.step(GridWorld.ACTION_LEFT, state=(0, 0))
    print(f"다음 상태: {next_state} (제자리)")
    print(f"보상: {reward:.2f}")


def test_policy_visualization():
    print("\n\n=== 정책 시각화 테스트 ===")
    env = GridWorld(4)

    # 간단한 정책 (항상 오른쪽과 아래로)
    policy = {}
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            pos = (r, c)
            if not env.is_terminal(pos):
                if c < env.grid_size - 1:
                    policy[pos] = GridWorld.ACTION_RIGHT
                else:
                    policy[pos] = GridWorld.ACTION_DOWN

    env.render_policy(policy)

    # 가치 함수 시각화
    values = {}
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            pos = (r, c)
            if pos == env.goal:
                values[pos] = 1.0
            elif pos in env.obstacles:
                values[pos] = -1.0
            else:
                dist = abs(r - env.goal[0]) + abs(c - env.goal[1])
                values[pos] = 1.0 - dist * 0.1

    env.render_values(values)


if __name__ == "__main__":
    test_gridworld()
    test_policy_visualization()
    print("\n\n=== GridWorld 환경 준비 완료! ===")
    print("이제 value_iteration.py와 policy_iteration.py에서 사용할 수 있습니다.")
