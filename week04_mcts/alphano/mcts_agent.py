#!/usr/bin/env python3
"""
ALPHANO ATAXX MCTS Agent
Week 4: Monte Carlo Tree Search

ALPHANO 프로토콜:
- READY FIRST/SECOND → print("OK")
- TURN my_time opp_time → print("MOVE x1 y1 x2 y2") (1-indexed)
- OPP x1 y1 x2 y2 → 보드 업데이트
- FINISH → 종료
"""

import sys
import time
import random
import math
from copy import deepcopy


class AtaxxBoard:
    """ATAXX 7x7 보드"""

    def __init__(self):
        """보드 초기화"""
        self.board = [[0] * 7 for _ in range(7)]
        # 초기 배치 (0-indexed)
        self.board[0][0] = 1  # FIRST
        self.board[6][6] = 1
        self.board[0][6] = 2  # SECOND
        self.board[6][0] = 2
        self.current_player = 1  # 1=FIRST, 2=SECOND

    def copy(self):
        """보드 복사"""
        new_board = AtaxxBoard()
        new_board.board = deepcopy(self.board)
        new_board.current_player = self.current_player
        return new_board

    def get_legal_moves(self):
        """가능한 모든 수 반환 (1-indexed)"""
        moves = []
        player = self.current_player

        for r in range(7):
            for c in range(7):
                if self.board[r][c] == player:
                    # Split (거리 1): 8방향
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 7 and 0 <= nc < 7 and self.board[nr][nc] == 0:
                                # (r, c)에서 (nr, nc)로 Split
                                moves.append((r + 1, c + 1, nr + 1, nc + 1))

                    # Jump (거리 2): 8방향
                    for dr in [-2, -1, 0, 1, 2]:
                        for dc in [-2, -1, 0, 1, 2]:
                            if abs(dr) <= 1 and abs(dc) <= 1:
                                continue  # 거리 1은 이미 처리
                            if max(abs(dr), abs(dc)) > 2:
                                continue  # 거리 2 초과
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < 7 and 0 <= nc < 7 and self.board[nr][nc] == 0:
                                # (r, c)에서 (nr, nc)로 Jump
                                moves.append((r + 1, c + 1, nr + 1, nc + 1))

        # 중복 제거
        moves = list(set(moves))

        # 가능한 수가 없으면 PASS
        if not moves:
            moves = [None]

        return moves

    def apply_move(self, move):
        """수를 적용 (원본 변경하지 않고 새 보드 반환)"""
        new_board = self.copy()
        new_board._apply_move_inplace(move)
        return new_board

    def _apply_move_inplace(self, move):
        """수를 적용 (원본 변경)"""
        if move is None:
            # PASS
            self.current_player = 3 - self.current_player
            return

        r1, c1, r2, c2 = move
        r1 -= 1
        c1 -= 1
        r2 -= 1
        c2 -= 1  # 0-indexed로 변환

        player = self.current_player

        # 거리 계산
        dist = max(abs(r2 - r1), abs(c2 - c1))

        if dist == 1:
            # Split: 복제
            self.board[r2][c2] = player
        elif dist == 2:
            # Jump: 이동
            self.board[r1][c1] = 0
            self.board[r2][c2] = player
        else:
            raise ValueError(f"Invalid move distance: {dist}")

        # 인접 8방향 감염
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r2 + dr, c2 + dc
                if 0 <= nr < 7 and 0 <= nc < 7:
                    if self.board[nr][nc] == (3 - player):
                        self.board[nr][nc] = player

        # 턴 전환
        self.current_player = 3 - self.current_player

    def is_terminal(self):
        """게임 종료 확인"""
        # 돌이 하나도 없으면 종료
        count1 = sum(row.count(1) for row in self.board)
        count2 = sum(row.count(2) for row in self.board)

        if count1 == 0 or count2 == 0:
            return True

        # 양쪽 다 수를 둘 수 없으면 종료
        moves = self.get_legal_moves()
        if moves == [None]:
            # 상대도 확인
            self.current_player = 3 - self.current_player
            opp_moves = self.get_legal_moves()
            self.current_player = 3 - self.current_player

            if opp_moves == [None]:
                return True

        # 보드가 꽉 차면 종료
        empty = sum(row.count(0) for row in self.board)
        if empty == 0:
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
        count1 = sum(row.count(1) for row in self.board)
        count2 = sum(row.count(2) for row in self.board)

        if count1 > count2:
            winner = 1
        elif count2 > count1:
            winner = 2
        else:
            return 0.5  # 무승부

        return 1.0 if winner == self.current_player else 0.0

    def count_pieces(self):
        """돌 개수 세기"""
        count1 = sum(row.count(1) for row in self.board)
        count2 = sum(row.count(2) for row in self.board)
        return count1, count2


class MCTSNode:
    """Monte Carlo Tree Search 노드"""

    def __init__(self, state, parent=None, move=None):
        """
        Args:
            state: AtaxxBoard 상태
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

        Args:
            c: 탐험 상수 (exploration constant)

        Returns:
            UCB1 값
        """
        if self.visits == 0:
            return float('inf')  # 미방문 노드는 최우선

        # Exploitation: 평균 승률
        exploitation = self.wins / self.visits

        # Exploration: 불확실성 보너스
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)

        return exploitation + exploration

    def select_child(self):
        """UCB1 값이 가장 큰 자식 선택"""
        return max(self.children, key=lambda child: child.ucb1())

    def expand(self):
        """
        미탐색 수 중 하나를 선택해서 자식 노드 추가

        Returns:
            새로 추가된 자식 노드
        """
        move = self.untried_moves.pop()
        next_state = self.state.apply_move(move)
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

        # 게임이 끝날 때까지 랜덤 플레이
        while not state.is_terminal():
            legal_moves = state.get_legal_moves()
            move = random.choice(legal_moves)
            state._apply_move_inplace(move)

        # 결과를 원래 플레이어 관점으로 변환
        # state.current_player는 게임 종료 후 플레이어
        # self.state.current_player는 롤아웃 시작 시 플레이어

        count1, count2 = state.count_pieces()

        if count1 > count2:
            winner = 1
        elif count2 > count1:
            winner = 2
        else:
            return 0.5  # 무승부

        # 원래 플레이어 관점으로 결과 반환
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


def mcts_search(board, time_limit_ms):
    """
    Monte Carlo Tree Search로 최선의 수 찾기

    Args:
        board: AtaxxBoard 상태
        time_limit_ms: 시간 제한 (밀리초)

    Returns:
        최선의 수 (1-indexed tuple) 또는 None (PASS)
    """
    root = MCTSNode(board)
    start_time = time.time()
    end_time = start_time + time_limit_ms / 1000.0

    iterations = 0

    while time.time() < end_time:
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

        iterations += 1

    # 디버깅 정보 출력 (stderr로 출력하면 ALPHANO가 무시)
    print(f"MCTS: {iterations} iterations in {time_limit_ms}ms", file=sys.stderr)

    # 최종 수 선택: 가장 많이 방문된 자식
    if root.children:
        best = root.best_child(c=0)
        print(f"Best move: {best.move}, visits: {best.visits}, winrate: {best.wins / best.visits:.2%}", file=sys.stderr)
        return best.move
    else:
        # 가능한 수가 없음 (PASS)
        return None


def main():
    """ALPHANO 프로토콜 메인 루프"""
    board = AtaxxBoard()
    my_color = None

    while True:
        try:
            line = input().strip()
            parts = line.split()

            if not parts:
                continue

            command = parts[0]

            if command == "READY":
                # READY FIRST/SECOND
                color = parts[1]
                if color == "FIRST":
                    my_color = 1
                elif color == "SECOND":
                    my_color = 2
                print("OK")
                sys.stdout.flush()

            elif command == "TURN":
                # TURN my_time opp_time
                my_time = int(parts[1])
                opp_time = int(parts[2])

                # 시간 관리: 남은 시간에 따라 조절
                if my_time > 1000:
                    time_limit = 150  # 1초 이상 남았으면 150ms 사용
                else:
                    time_limit = 10   # 시간이 없으면 10ms만 사용

                # MCTS로 수 찾기
                best_move = mcts_search(board, time_limit)

                if best_move is None:
                    # PASS (가능한 수가 없음)
                    print("PASS")
                else:
                    # MOVE x1 y1 x2 y2
                    print(f"MOVE {best_move[0]} {best_move[1]} {best_move[2]} {best_move[3]}")

                sys.stdout.flush()

                # 보드 업데이트
                if best_move is not None:
                    board._apply_move_inplace(best_move)

            elif command == "OPP":
                # OPP x1 y1 x2 y2 (상대방 수)
                if len(parts) == 5:
                    x1, y1, x2, y2 = map(int, parts[1:5])
                    move = (x1, y1, x2, y2)
                    board._apply_move_inplace(move)
                elif parts[1] == "PASS":
                    # 상대방 PASS
                    board.current_player = 3 - board.current_player

            elif command == "FINISH":
                # 게임 종료
                break

        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            break


if __name__ == "__main__":
    main()
