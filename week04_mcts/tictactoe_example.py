#!/usr/bin/env python3
"""
틱택토 MCTS 예제

간단한 게임으로 MCTS를 이해하기 위한 예제입니다.
ATAXX MCTS를 구현하기 전에 이 예제를 먼저 공부하세요.
"""

import random
import math
from copy import deepcopy


class TicTacToe:
    """틱택토 게임 (3x3)"""

    def __init__(self):
        """보드 초기화"""
        self.board = [[0] * 3 for _ in range(3)]
        self.current_player = 1  # 1 = X, 2 = O

    def copy(self):
        """보드 복사"""
        new_game = TicTacToe()
        new_game.board = deepcopy(self.board)
        new_game.current_player = self.current_player
        return new_game

    def get_legal_moves(self):
        """
        가능한 수 반환

        Returns:
            [(r, c), ...]: 빈 칸의 좌표 리스트
        """
        moves = []
        for r in range(3):
            for c in range(3):
                if self.board[r][c] == 0:
                    moves.append((r, c))
        return moves

    def apply_move(self, move):
        """
        수를 적용한 새 게임 상태 반환

        Args:
            move: (r, c) 좌표

        Returns:
            새 TicTacToe 인스턴스
        """
        new_game = self.copy()
        r, c = move
        new_game.board[r][c] = new_game.current_player
        new_game.current_player = 3 - new_game.current_player  # 턴 전환
        return new_game

    def is_terminal(self):
        """
        게임 종료 확인

        Returns:
            True if 게임이 끝남
        """
        # 승리 조건 확인
        if self._check_winner() != 0:
            return True

        # 빈 칸이 없으면 무승부
        if not self.get_legal_moves():
            return True

        return False

    def get_result(self):
        """
        현재 플레이어 관점의 게임 결과

        Returns:
            1.0: 현재 플레이어 승리
            0.0: 현재 플레이어 패배
            0.5: 무승부
        """
        winner = self._check_winner()

        if winner == 0:
            return 0.5  # 무승부

        # 현재 플레이어가 승자인가?
        return 1.0 if winner == self.current_player else 0.0

    def _check_winner(self):
        """
        승자 확인

        Returns:
            0: 승자 없음
            1: X(1번) 승리
            2: O(2번) 승리
        """
        # 가로 확인
        for r in range(3):
            if self.board[r][0] == self.board[r][1] == self.board[r][2] != 0:
                return self.board[r][0]

        # 세로 확인
        for c in range(3):
            if self.board[0][c] == self.board[1][c] == self.board[2][c] != 0:
                return self.board[0][c]

        # 대각선 확인
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != 0:
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != 0:
            return self.board[0][2]

        return 0  # 승자 없음

    def __str__(self):
        """보드 출력"""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = []
        for row in self.board:
            lines.append(' '.join(symbols[cell] for cell in row))
        return '\n'.join(lines)


class MCTSNode:
    """Monte Carlo Tree Search 노드"""

    def __init__(self, state, parent=None, move=None):
        """
        Args:
            state: TicTacToe 게임 상태
            parent: 부모 노드
            move: 부모에서 이 노드로 오는 수
        """
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried_moves = state.get_legal_moves()

    def is_fully_expanded(self):
        """모든 자식이 확장되었는가?"""
        return len(self.untried_moves) == 0

    def is_terminal(self):
        """터미널 노드인가?"""
        return self.state.is_terminal()

    def ucb1(self, c=1.414):
        """
        UCB1 값 계산

        UCB1 = exploitation + exploration
             = w/n + c * sqrt(ln(N)/n)

        Args:
            c: 탐험 상수 (exploration constant)

        Returns:
            UCB1 값
        """
        if self.visits == 0:
            return float('inf')  # 미방문 노드는 최우선 선택

        exploitation = self.wins / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def select_child(self):
        """
        UCB1 값이 가장 큰 자식 선택

        Returns:
            최선의 자식 노드
        """
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        """
        미탐색 수 중 하나를 선택해서 자식 노드 추가

        Returns:
            새로 추가된 자식 노드
        """
        # 미탐색 수 중 하나 선택
        move = self.untried_moves.pop()

        # 수를 적용한 새 상태 생성
        next_state = self.state.apply_move(move)

        # 자식 노드 생성
        child = MCTSNode(next_state, parent=self, move=move)
        self.children.append(child)

        return child

    def rollout(self):
        """
        현재 상태에서 게임 끝까지 랜덤 플레이

        Returns:
            게임 결과 (현재 플레이어 관점)
        """
        state = self.state.copy()

        # 게임이 끝날 때까지 랜덤으로 진행
        while not state.is_terminal():
            legal_moves = state.get_legal_moves()
            move = random.choice(legal_moves)
            state = state.apply_move(move)

        # 결과를 원래 플레이어 관점으로 변환
        # state.current_player는 게임 종료 후 플레이어
        # self.state.current_player는 시뮬레이션 시작 시 플레이어

        winner = state._check_winner()

        if winner == 0:
            return 0.5  # 무승부

        # 원래 플레이어가 승자인가?
        return 1.0 if winner == self.state.current_player else 0.0

    def backpropagate(self, result):
        """
        결과를 루트까지 역전파

        Args:
            result: 시뮬레이션 결과
        """
        node = self
        while node is not None:
            node.visits += 1
            node.wins += result
            result = 1.0 - result  # 관점 전환 (부모는 상대편)
            node = node.parent

    def best_child(self, c=0):
        """
        최선의 자식 선택

        Args:
            c: 0이면 방문 횟수 기준, 0 아니면 UCB1 기준

        Returns:
            최선의 자식 노드
        """
        if c == 0:
            # 방문 횟수가 가장 많은 자식
            return max(self.children, key=lambda child: child.visits)
        else:
            # UCB1 값이 가장 큰 자식
            return max(self.children, key=lambda child: child.ucb1(c))


def mcts(root_state, iterations=1000):
    """
    Monte Carlo Tree Search

    Args:
        root_state: 초기 게임 상태
        iterations: 시뮬레이션 반복 횟수

    Returns:
        최선의 수
    """
    root = MCTSNode(root_state)

    for i in range(iterations):
        node = root

        # 1. Selection: 리프 노드까지 내려가기
        while not node.is_terminal() and node.is_fully_expanded():
            node = node.select_child()

        # 2. Expansion: 가능하면 확장
        if not node.is_terminal() and not node.is_fully_expanded():
            node = node.expand()

        # 3. Simulation: 게임 끝까지 랜덤 플레이
        result = node.rollout()

        # 4. Backpropagation: 결과 역전파
        node.backpropagate(result)

    # 최종 수 선택: 가장 많이 방문된 자식
    if root.children:
        best = root.best_child(c=0)
        print(f"선택된 수: {best.move}, 방문: {best.visits}, 승률: {best.wins/best.visits:.2%}")
        return best.move
    else:
        # 가능한 수가 없음
        return None


def play_game_mcts_vs_random():
    """MCTS vs 랜덤 플레이어"""
    print("=" * 60)
    print("틱택토: MCTS(X) vs 랜덤(O)")
    print("=" * 60)

    game = TicTacToe()

    while not game.is_terminal():
        print(f"\n현재 보드 (차례: {'X' if game.current_player == 1 else 'O'}):")
        print(game)
        print()

        if game.current_player == 1:
            # MCTS 차례
            print("MCTS 생각 중...")
            move = mcts(game, iterations=1000)
        else:
            # 랜덤 차례
            moves = game.get_legal_moves()
            move = random.choice(moves)
            print(f"랜덤 선택: {move}")

        game = game.apply_move(move)

    # 최종 결과
    print("\n최종 보드:")
    print(game)
    print()

    winner = game._check_winner()
    if winner == 0:
        print("결과: 무승부")
    elif winner == 1:
        print("결과: MCTS(X) 승리!")
    else:
        print("결과: 랜덤(O) 승리!")


def play_game_mcts_vs_mcts():
    """MCTS vs MCTS"""
    print("=" * 60)
    print("틱택토: MCTS(X) vs MCTS(O)")
    print("=" * 60)

    game = TicTacToe()

    while not game.is_terminal():
        print(f"\n현재 보드 (차례: {'X' if game.current_player == 1 else 'O'}):")
        print(game)
        print()

        print("MCTS 생각 중...")
        move = mcts(game, iterations=500)

        game = game.apply_move(move)

    # 최종 결과
    print("\n최종 보드:")
    print(game)
    print()

    winner = game._check_winner()
    if winner == 0:
        print("결과: 무승부")
    elif winner == 1:
        print("결과: MCTS(X) 승리!")
    else:
        print("결과: MCTS(O) 승리!")


def test_iterations():
    """시뮬레이션 횟수에 따른 성능 테스트"""
    print("=" * 60)
    print("성능 테스트: 시뮬레이션 횟수 vs 승률")
    print("=" * 60)

    iteration_counts = [10, 50, 100, 500, 1000]
    games_per_setting = 20

    for iterations in iteration_counts:
        wins = 0
        draws = 0
        losses = 0

        print(f"\nMCTS({iterations} iterations) vs 랜덤: ", end="", flush=True)

        for game_num in range(games_per_setting):
            game = TicTacToe()

            while not game.is_terminal():
                if game.current_player == 1:
                    # MCTS
                    root = MCTSNode(game)
                    for _ in range(iterations):
                        node = root
                        while not node.is_terminal() and node.is_fully_expanded():
                            node = node.select_child()
                        if not node.is_terminal() and not node.is_fully_expanded():
                            node = node.expand()
                        result = node.rollout()
                        node.backpropagate(result)
                    if root.children:
                        best = root.best_child(c=0)
                        move = best.move
                    else:
                        move = None
                else:
                    # 랜덤
                    moves = game.get_legal_moves()
                    move = random.choice(moves)

                if move:
                    game = game.apply_move(move)
                else:
                    break

            # 결과 집계
            winner = game._check_winner()
            if winner == 1:
                wins += 1
                print("W", end="", flush=True)
            elif winner == 2:
                losses += 1
                print("L", end="", flush=True)
            else:
                draws += 1
                print("D", end="", flush=True)

        print(f"\n  승: {wins}/{games_per_setting} ({wins/games_per_setting*100:.1f}%)")
        print(f"  무: {draws}/{games_per_setting} ({draws/games_per_setting*100:.1f}%)")
        print(f"  패: {losses}/{games_per_setting} ({losses/games_per_setting*100:.1f}%)")


def main():
    """메인 함수"""
    print("\n틱택토 MCTS 예제\n")
    print("1. MCTS vs 랜덤 (1게임)")
    print("2. MCTS vs MCTS (1게임)")
    print("3. 성능 테스트 (시뮬레이션 횟수별)")
    print("4. 모두 실행")
    print()

    choice = input("선택 (1-4): ").strip()

    if choice == '1':
        play_game_mcts_vs_random()
    elif choice == '2':
        play_game_mcts_vs_mcts()
    elif choice == '3':
        test_iterations()
    elif choice == '4':
        play_game_mcts_vs_random()
        print("\n" + "=" * 60 + "\n")
        play_game_mcts_vs_mcts()
        print("\n" + "=" * 60 + "\n")
        test_iterations()
    else:
        print("잘못된 선택입니다.")


if __name__ == "__main__":
    main()
