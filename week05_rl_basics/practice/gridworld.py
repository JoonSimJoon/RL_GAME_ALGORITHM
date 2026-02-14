"""
GridWorld 환경 구현
4x4 격자 세계에서 에이전트가 목표를 찾아가는 간단한 MDP 환경
"""

import numpy as np
from typing import Tuple, List, Optional


class GridWorld:
    """
    4x4 GridWorld 환경

    상태: (row, col) 좌표
    행동: 0(상), 1(하), 2(좌), 3(우)
    보상: 목표(+1), 장애물(-1), 이동(-0.04)
    """

    # 행동 정의
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTION_NAMES = ['↑', '↓', '←', '→']

    def __init__(self, grid_size: int = 4):
        """
        GridWorld 초기화

        Args:
            grid_size: 격자 크기 (기본 4x4)
        """
        self.grid_size = grid_size
        self.start = (0, 0)  # 시작 위치
        self.goal = (grid_size - 1, grid_size - 1)  # 목표 위치
        self.obstacles = [(1, 1), (2, 2)]  # 장애물 위치

        # 현재 상태
        self.current_state = self.start

        # 모든 가능한 상태 리스트
        self.states = [
            (r, c)
            for r in range(self.grid_size)
            for c in range(self.grid_size)
        ]

        # 행동 개수
        self.n_actions = 4

    def reset(self) -> Tuple[int, int]:
        """
        환경을 초기 상태로 리셋

        Returns:
            초기 상태
        """
        self.current_state = self.start
        return self.current_state

    def step(self, state: Optional[Tuple[int, int]], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        상태 전이 수행

        Args:
            state: 현재 상태 (None이면 self.current_state 사용)
            action: 행동 (0: 상, 1: 하, 2: 좌, 3: 우)

        Returns:
            next_state: 다음 상태
            reward: 보상
            done: 에피소드 종료 여부
        """
        if state is None:
            state = self.current_state

        # 다음 상태 계산
        next_state = self._get_next_state(state, action)

        # 보상 계산
        reward = self._get_reward(next_state)

        # 종료 확인
        done = self.is_terminal(next_state)

        # 현재 상태 업데이트 (state가 None이었을 경우)
        if state == self.current_state:
            self.current_state = next_state

        return next_state, reward, done

    def _get_next_state(self, state: Tuple[int, int], action: int) -> Tuple[int, int]:
        """
        행동에 따른 다음 상태 계산 (벽 처리 포함)

        Args:
            state: 현재 상태
            action: 행동

        Returns:
            다음 상태
        """
        row, col = state

        # 행동에 따른 이동
        if action == self.ACTION_UP:
            next_row, next_col = row - 1, col
        elif action == self.ACTION_DOWN:
            next_row, next_col = row + 1, col
        elif action == self.ACTION_LEFT:
            next_row, next_col = row, col - 1
        elif action == self.ACTION_RIGHT:
            next_row, next_col = row, col + 1
        else:
            raise ValueError(f"Invalid action: {action}")

        # 벽 체크 (격자를 벗어나면 제자리)
        if not self._is_valid_position(next_row, next_col):
            return state

        return (next_row, next_col)

    def _is_valid_position(self, row: int, col: int) -> bool:
        """
        위치가 격자 내부인지 확인

        Args:
            row: 행
            col: 열

        Returns:
            유효하면 True
        """
        return 0 <= row < self.grid_size and 0 <= col < self.grid_size

    def _get_reward(self, state: Tuple[int, int]) -> float:
        """
        상태에 대한 보상 계산

        Args:
            state: 상태

        Returns:
            보상
        """
        if state == self.goal:
            return 1.0  # 목표 도달
        elif state in self.obstacles:
            return -1.0  # 장애물 충돌
        else:
            return -0.04  # 일반 이동 (에너지 소모)

    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """
        종료 상태인지 확인

        Args:
            state: 상태

        Returns:
            종료 상태면 True
        """
        return state == self.goal or state in self.obstacles

    def get_possible_actions(self, state: Tuple[int, int]) -> List[int]:
        """
        주어진 상태에서 가능한 행동 리스트

        Args:
            state: 상태

        Returns:
            가능한 행동 리스트
        """
        if self.is_terminal(state):
            return []
        return list(range(self.n_actions))

    def render(self, show_agent: bool = True) -> None:
        """
        현재 상태 시각화 (텍스트)

        Args:
            show_agent: 에이전트 위치 표시 여부
        """
        print("\n" + "=" * (self.grid_size * 4 + 1))

        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                pos = (r, c)

                if show_agent and pos == self.current_state:
                    cell = " A "  # Agent
                elif pos == self.goal:
                    cell = " G "  # Goal
                elif pos in self.obstacles:
                    cell = " X "  # Obstacle
                elif pos == self.start:
                    cell = " S "  # Start
                else:
                    cell = "   "

                row_str += cell + "|"

            print(row_str)
            print("=" * (self.grid_size * 4 + 1))

        print()

    def render_policy(self, policy: dict) -> None:
        """
        정책 시각화

        Args:
            policy: 정책 딕셔너리 {state: action}
        """
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
                    action = policy[pos]
                    cell = f" {self.ACTION_NAMES[action]} "
                else:
                    cell = "   "

                row_str += cell + "|"

            print(row_str)
            print("=" * (self.grid_size * 4 + 1))

        print()

    def render_values(self, values: dict) -> None:
        """
        가치 함수 시각화

        Args:
            values: 가치 딕셔너리 {state: value}
        """
        print("\n가치 함수 시각화:")
        print("=" * (self.grid_size * 7 + 1))

        for r in range(self.grid_size):
            row_str = "|"
            for c in range(self.grid_size):
                pos = (r, c)

                if pos in values:
                    value = values[pos]
                    cell = f"{value:6.2f}"
                else:
                    cell = "  N/A "

                row_str += cell + "|"

            print(row_str)
            print("=" * (self.grid_size * 7 + 1))

        print()


def test_gridworld():
    """GridWorld 환경 테스트"""
    print("=== GridWorld 테스트 ===\n")

    # 환경 생성
    env = GridWorld()

    # 초기 상태 표시
    print("초기 상태:")
    env.render()

    # 몇 번 이동 시도
    print("\n행동 시퀀스 테스트:")
    actions = [
        (GridWorld.ACTION_RIGHT, "오른쪽"),
        (GridWorld.ACTION_RIGHT, "오른쪽"),
        (GridWorld.ACTION_DOWN, "아래"),
        (GridWorld.ACTION_DOWN, "아래"),
    ]

    total_reward = 0
    for action, action_name in actions:
        next_state, reward, done = env.step(None, action)
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

    next_state, reward, done = env.step((0, 0), GridWorld.ACTION_DOWN)
    print(f"다음 상태: {next_state}")
    print(f"보상: {reward:.2f}")
    print(f"종료: {done}")

    # 벽 테스트
    print("\n\n=== 벽 테스트 ===")
    print("(0, 0)에서 위로 이동 (벽):")
    next_state, reward, done = env.step((0, 0), GridWorld.ACTION_UP)
    print(f"다음 상태: {next_state} (제자리)")
    print(f"보상: {reward:.2f}")

    print("\n(0, 0)에서 왼쪽으로 이동 (벽):")
    next_state, reward, done = env.step((0, 0), GridWorld.ACTION_LEFT)
    print(f"다음 상태: {next_state} (제자리)")
    print(f"보상: {reward:.2f}")


def test_policy_visualization():
    """정책 시각화 테스트"""
    print("\n\n=== 정책 시각화 테스트 ===")

    env = GridWorld()

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
            # 간단한 가치: 목표까지의 거리 기반
            pos = (r, c)
            if pos == env.goal:
                values[pos] = 1.0
            elif pos in env.obstacles:
                values[pos] = -1.0
            else:
                # 목표까지의 맨해튼 거리
                dist = abs(r - env.goal[0]) + abs(c - env.goal[1])
                values[pos] = 1.0 - dist * 0.1

    env.render_values(values)


if __name__ == "__main__":
    test_gridworld()
    test_policy_visualization()

    print("\n\n=== GridWorld 환경 준비 완료! ===")
    print("이제 value_iteration.py와 policy_iteration.py에서 사용할 수 있습니다.")
