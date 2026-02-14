#!/usr/bin/env python3
"""
ALPHANO Alpha-Beta Pruning Agent
Week 2: Alpha-Beta Pruning & Agent Performance Validation

ATAXX 게임 규칙:
- 7x7 보드
- 거리 1: Split (복제, 원본 유지)
- 거리 2: Jump (이동, 원본 제거)
- 이동 후 인접 8방향 적 말 변환
- 좌표는 1-indexed (x=열, y=행)

ALPHANO 프로토콜:
- READY FIRST/SECOND → print("OK")
- TURN my_time opp_time → print("MOVE x1 y1 x2 y2")
- OPP x1 y1 x2 y2 → update board
- FINISH → exit
"""

import sys
import time

# 상수 정의
BOARD_SIZE = 7
EMPTY = 0
PLAYER1 = 1  # 선공
PLAYER2 = 2  # 후공
INF = float('inf')

# 방향 벡터 (8방향)
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

# 위치 가중치 (코너 중시)
POSITION_WEIGHTS = [
    [100, -20,  10,   5,  10, -20, 100],
    [-20, -40,  -5,  -5,  -5, -40, -20],
    [ 10,  -5,  10,   5,  10,  -5,  10],
    [  5,  -5,   5,   0,   5,  -5,   5],
    [ 10,  -5,  10,   5,  10,  -5,  10],
    [-20, -40,  -5,  -5,  -5, -40, -20],
    [100, -20,  10,   5,  10, -20, 100],
]


class AtaxxBoard:
    """ATAXX 게임 보드"""

    def __init__(self):
        self.board = [[EMPTY] * BOARD_SIZE for _ in range(BOARD_SIZE)]
        # 초기 배치: 코너 4곳
        self.board[0][0] = PLAYER2
        self.board[0][6] = PLAYER1
        self.board[6][0] = PLAYER1
        self.board[6][6] = PLAYER2

    def copy(self):
        """보드 복사"""
        new_board = AtaxxBoard()
        new_board.board = [row[:] for row in self.board]
        return new_board

    def get_opponent(self, player):
        """상대 플레이어"""
        return PLAYER1 if player == PLAYER2 else PLAYER2

    def is_valid_pos(self, row, col):
        """유효한 좌표인지 확인"""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def manhattan_distance(self, r1, c1, r2, c2):
        """맨해튼 거리"""
        return abs(r1 - r2) + abs(c1 - c2)

    def chebyshev_distance(self, r1, c1, r2, c2):
        """체비셰프 거리 (ATAXX는 대각선 이동 가능)"""
        return max(abs(r1 - r2), abs(c1 - c2))

    def get_possible_moves(self, player):
        """가능한 모든 수 반환 [(r1, c1, r2, c2), ...]"""
        moves = []

        for r1 in range(BOARD_SIZE):
            for c1 in range(BOARD_SIZE):
                if self.board[r1][c1] != player:
                    continue

                # 거리 1 (Split) 및 거리 2 (Jump) 탐색
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue

                        r2, c2 = r1 + dr, c1 + dc

                        # 목적지가 비어있고 유효한 위치인지 확인
                        if self.is_valid_pos(r2, c2) and self.board[r2][c2] == EMPTY:
                            dist = self.chebyshev_distance(r1, c1, r2, c2)
                            if dist == 1 or dist == 2:
                                moves.append((r1, c1, r2, c2))

        return moves

    def make_move(self, move, player):
        """수를 둔 후의 새로운 보드 반환"""
        r1, c1, r2, c2 = move
        new_board = self.copy()

        dist = self.chebyshev_distance(r1, c1, r2, c2)

        if dist == 1:
            # Split: 복제
            new_board.board[r2][c2] = player
        elif dist == 2:
            # Jump: 이동
            new_board.board[r1][c1] = EMPTY
            new_board.board[r2][c2] = player
        else:
            raise ValueError(f"Invalid move distance: {dist}")

        # 인접한 적 말 감염
        opponent = self.get_opponent(player)
        for dr, dc in DIRECTIONS:
            nr, nc = r2 + dr, c2 + dc
            if new_board.is_valid_pos(nr, nc) and new_board.board[nr][nc] == opponent:
                new_board.board[nr][nc] = player

        return new_board

    def count_pieces(self, player):
        """플레이어의 돌 개수"""
        return sum(row.count(player) for row in self.board)

    def is_game_over(self):
        """게임 종료 확인"""
        # 둘 다 수를 둘 수 없으면 종료
        return not self.get_possible_moves(PLAYER1) and not self.get_possible_moves(PLAYER2)

    def get_winner(self):
        """승자 반환 (게임 종료 시)"""
        p1_count = self.count_pieces(PLAYER1)
        p2_count = self.count_pieces(PLAYER2)

        if p1_count > p2_count:
            return PLAYER1
        elif p2_count > p1_count:
            return PLAYER2
        else:
            return 0  # 무승부

    def evaluate_advanced(self, player):
        """개선된 평가 함수"""
        # 게임 종료 확인
        if self.is_game_over():
            winner = self.get_winner()
            if winner == player:
                return 10000
            elif winner == self.get_opponent(player):
                return -10000
            else:
                return 0

        opponent = self.get_opponent(player)

        # 1. 돌 개수
        my_pieces = self.count_pieces(player)
        opp_pieces = self.count_pieces(opponent)
        piece_score = my_pieces - opp_pieces

        # 2. 위치 가치
        my_position = 0
        opp_position = 0
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board[r][c] == player:
                    my_position += POSITION_WEIGHTS[r][c]
                elif self.board[r][c] == opponent:
                    opp_position += POSITION_WEIGHTS[r][c]
        position_score = my_position - opp_position

        # 3. 이동성 (가능한 수의 개수)
        my_mobility = len(self.get_possible_moves(player))
        opp_mobility = len(self.get_possible_moves(opponent))
        mobility_score = my_mobility - opp_mobility

        # 4. 게임 단계별 가중치
        total_pieces = my_pieces + opp_pieces

        if total_pieces < 20:  # 초반: 이동성과 위치 중요
            weight_piece = 1
            weight_position = 3
            weight_mobility = 5
        elif total_pieces < 35:  # 중반: 균형
            weight_piece = 2
            weight_position = 2
            weight_mobility = 3
        else:  # 후반: 돌 개수가 가장 중요
            weight_piece = 5
            weight_position = 1
            weight_mobility = 1

        return (weight_piece * piece_score +
                weight_position * position_score +
                weight_mobility * mobility_score)


def negamax_alpha_beta(board, depth, alpha, beta, player):
    """
    Negamax with Alpha-Beta Pruning

    Args:
        board: AtaxxBoard 객체
        depth: 남은 탐색 깊이
        alpha: 현재까지 MAX가 보장받은 최소 점수
        beta: 현재까지 MIN이 허용할 최대 점수
        player: 현재 플레이어

    Returns:
        (score, best_move): 평가 점수와 최선의 수
    """
    # 기저 조건: 깊이 0 또는 게임 종료
    if depth == 0 or board.is_game_over():
        return board.evaluate_advanced(player), None

    # 가능한 수 탐색
    moves = board.get_possible_moves(player)

    # 수가 없으면 패스 (상대방 차례)
    if not moves:
        opponent = board.get_opponent(player)
        score, _ = negamax_alpha_beta(board, depth - 1, -beta, -alpha, opponent)
        return -score, None

    best_move = None

    # Move Ordering: 중앙/코너 우선 (간단한 휴리스틱)
    def move_priority(move):
        r1, c1, r2, c2 = move
        # 목적지의 위치 가중치를 우선순위로 사용
        return -POSITION_WEIGHTS[r2][c2]

    moves.sort(key=move_priority)

    # 모든 수에 대해 탐색
    for move in moves:
        new_board = board.make_move(move, player)
        opponent = board.get_opponent(player)

        # 재귀 호출: 부호 반전 및 alpha-beta 순서 반전
        score, _ = negamax_alpha_beta(new_board, depth - 1, -beta, -alpha, opponent)
        score = -score

        # Alpha 업데이트
        if score > alpha:
            alpha = score
            best_move = move

        # Beta Cutoff: 가지치기!
        if alpha >= beta:
            break

    return alpha, best_move


class AlphaBetaAgent:
    """Alpha-Beta Pruning 에이전트"""

    def __init__(self, depth=4):
        self.depth = depth
        self.board = AtaxxBoard()
        self.my_player = None

    def get_best_move(self):
        """Alpha-Beta로 최선의 수 찾기"""
        score, move = negamax_alpha_beta(self.board, self.depth, -INF, INF, self.my_player)
        return move

    def apply_move(self, move, player):
        """보드에 수 적용"""
        self.board = self.board.make_move(move, player)

    def coord_to_1indexed(self, r, c):
        """0-indexed → 1-indexed 변환"""
        return c + 1, r + 1

    def coord_to_0indexed(self, x, y):
        """1-indexed → 0-indexed 변환"""
        return y - 1, x - 1

    def run(self):
        """ALPHANO 프로토콜 실행"""
        for line in sys.stdin:
            line = line.strip()
            parts = line.split()

            if not parts:
                continue

            command = parts[0]

            if command == "READY":
                # READY FIRST/SECOND
                position = parts[1]
                self.my_player = PLAYER1 if position == "FIRST" else PLAYER2
                print("OK")
                sys.stdout.flush()

            elif command == "TURN":
                # TURN my_time opp_time
                # 최선의 수 찾기
                move = self.get_best_move()

                if move:
                    r1, c1, r2, c2 = move
                    x1, y1 = self.coord_to_1indexed(r1, c1)
                    x2, y2 = self.coord_to_1indexed(r2, c2)

                    # 수 적용
                    self.apply_move(move, self.my_player)

                    # 출력
                    print(f"MOVE {x1} {y1} {x2} {y2}")
                    sys.stdout.flush()
                else:
                    # 둘 수 없으면 패스 (실제로는 발생하지 않아야 함)
                    print("PASS")
                    sys.stdout.flush()

            elif command == "OPP":
                # OPP x1 y1 x2 y2
                x1, y1, x2, y2 = map(int, parts[1:5])
                r1, c1 = self.coord_to_0indexed(x1, y1)
                r2, c2 = self.coord_to_0indexed(x2, y2)

                opponent = self.board.get_opponent(self.my_player)
                move = (r1, c1, r2, c2)
                self.apply_move(move, opponent)

            elif command == "FINISH":
                # 게임 종료
                break


if __name__ == "__main__":
    # depth=4: Alpha-Beta 덕분에 실용적인 성능
    # depth=5로 올리면 더 강하지만 약간 느릴 수 있음
    agent = AlphaBetaAgent(depth=4)
    agent.run()
