"""
ATAXX (세균 전쟁) - Minimax Agent
ALPHANO 프로토콜을 따르는 Minimax/Negamax 에이전트

전략:
- Negamax 알고리즘 구현 (Minimax의 간소화 버전)
- 깊이 제한 탐색 (기본 depth=3)
- 평가 함수: (내 말 개수 - 상대 말 개수)

Minimax/Negamax 개념:
- Minimax: 플레이어가 최선의 수를 두고, 상대방도 최선의 수를 둔다고 가정
- Negamax: Minimax의 간소화 버전, 평가값을 부정하여 한 함수로 구현
- 깊이 제한: 시간 제한을 위해 일정 깊이까지만 탐색
- 평가 함수: 말판 상태를 점수화하여 어느 쪽이 유리한지 판단
"""

import sys


class AtaxxBoard:
    """ATAXX 게임 보드를 표현하는 클래스"""

    def __init__(self):
        # 7x7 보드 초기화 (0-indexed 내부 사용)
        self.board = [['' for _ in range(7)] for _ in range(7)]
        self.my_symbol = ''  # 'O' 또는 'X'
        self.opp_symbol = ''

        # 초기 배치
        self.board[0][0] = 'O'  # (1,1) in 1-indexed
        self.board[6][6] = 'O'  # (7,7) in 1-indexed
        self.board[0][6] = 'X'  # (1,7) in 1-indexed
        self.board[6][0] = 'X'  # (7,1) in 1-indexed

    def set_player(self, is_first):
        """플레이어 심볼 설정"""
        if is_first:
            self.my_symbol = 'O'
            self.opp_symbol = 'X'
        else:
            self.my_symbol = 'X'
            self.opp_symbol = 'O'

    def get_legal_moves(self, symbol):
        """
        특정 플레이어의 모든 합법적인 이동 반환
        반환값: [(x1, y1, x2, y2), ...] (1-indexed 좌표)
        """
        moves = []

        # 해당 플레이어의 모든 말 찾기
        for i in range(7):
            for j in range(7):
                if self.board[i][j] == symbol:
                    # 이 말로부터 가능한 모든 이동 생성
                    moves.extend(self._get_moves_from(i, j))

        return moves

    def _get_moves_from(self, row, col):
        """
        특정 위치(row, col)에서 가능한 모든 이동 생성
        - 거리 1: Split (분열) - 원본 유지
        - 거리 2: Jump (점프) - 원본 이동
        """
        moves = []

        # 8방향 + 거리 1, 2 탐색
        for dr in [-2, -1, 0, 1, 2]:
            for dc in [-2, -1, 0, 1, 2]:
                if dr == 0 and dc == 0:
                    continue

                # Chebyshev 거리 계산 (max(|dx|, |dy|))
                distance = max(abs(dr), abs(dc))
                if distance > 2:
                    continue

                new_row = row + dr
                new_col = col + dc

                # 보드 범위 확인
                if 0 <= new_row < 7 and 0 <= new_col < 7:
                    # 목적지가 비어있어야 함
                    if self.board[new_row][new_col] == '':
                        # 1-indexed로 변환하여 추가
                        moves.append((col + 1, row + 1, new_col + 1, new_row + 1))

        return moves

    def make_move(self, x1, y1, x2, y2, symbol):
        """
        이동 실행 (1-indexed 좌표)
        x1, y1: 출발 좌표
        x2, y2: 도착 좌표
        """
        # 0-indexed로 변환
        col1, row1 = x1 - 1, y1 - 1
        col2, row2 = x2 - 1, y2 - 1

        # 거리 계산
        distance = max(abs(row2 - row1), abs(col2 - col1))

        # 도착지에 말 배치
        self.board[row2][col2] = symbol

        # Jump (거리 2)인 경우 원본 제거
        if distance == 2:
            self.board[row1][col1] = ''
        # Split (거리 1)인 경우 원본 유지

        # 인접한 8방향의 적 말을 아군으로 변환
        self._convert_adjacent(row2, col2, symbol)

    def _convert_adjacent(self, row, col, symbol):
        """도착지 주변 8방향의 적 말을 아군으로 변환"""
        opponent = 'X' if symbol == 'O' else 'O'

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                new_row = row + dr
                new_col = col + dc

                if 0 <= new_row < 7 and 0 <= new_col < 7:
                    if self.board[new_row][new_col] == opponent:
                        self.board[new_row][new_col] = symbol

    def copy(self):
        """보드 상태 복사 (시뮬레이션용)"""
        new_board = AtaxxBoard()
        new_board.board = [row[:] for row in self.board]
        new_board.my_symbol = self.my_symbol
        new_board.opp_symbol = self.opp_symbol
        return new_board

    def count_pieces(self, symbol):
        """특정 심볼의 말 개수 세기"""
        count = 0
        for i in range(7):
            for j in range(7):
                if self.board[i][j] == symbol:
                    count += 1
        return count

    def evaluate(self, player_symbol):
        """
        현재 보드 상태를 player_symbol 관점에서 평가
        평가 함수: (player의 말 개수 - 상대의 말 개수)

        Args:
            player_symbol: 평가 대상 플레이어의 심볼

        Returns:
            평가 점수 (int)
        """
        opponent_symbol = 'X' if player_symbol == 'O' else 'O'

        my_count = self.count_pieces(player_symbol)
        opp_count = self.count_pieces(opponent_symbol)

        return my_count - opp_count


class MinimaxAgent:
    """
    Minimax/Negamax 알고리즘을 사용하는 에이전트

    Negamax는 Minimax의 간소화 버전으로:
    - Max와 Min을 하나의 함수로 통합
    - 평가값을 부정(negate)하여 관점을 전환
    - 코드가 간결하고 이해하기 쉬움
    """

    def __init__(self, depth=3):
        """
        Args:
            depth: Negamax 탐색 깊이 (기본값: 3)
        """
        self.board = AtaxxBoard()
        self.depth = depth

    def run(self):
        """ALPHANO 프로토콜에 따라 게임 진행"""
        while True:
            line = sys.stdin.readline().strip()

            if not line:
                continue

            parts = line.split()
            command = parts[0]

            if command == "READY":
                # 선공/후공 설정
                is_first = (parts[1] == "FIRST")
                self.board.set_player(is_first)
                print("OK")
                sys.stdout.flush()

            elif command == "TURN":
                # 내 차례 - 이동 결정
                my_time = int(parts[1])
                opp_time = int(parts[2])

                move = self.select_move()

                if move:
                    x1, y1, x2, y2 = move
                    print(f"MOVE {x1} {y1} {x2} {y2}")
                    # 보드 업데이트
                    self.board.make_move(x1, y1, x2, y2, self.board.my_symbol)
                else:
                    # 가능한 수가 없으면 PASS
                    print("MOVE -1 -1 -1 -1")

                sys.stdout.flush()

            elif command == "OPP":
                # 상대방 이동 - 보드 업데이트
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])

                if x1 != -1:  # PASS가 아닌 경우
                    self.board.make_move(x1, y1, x2, y2, self.board.opp_symbol)

            elif command == "FINISH":
                # 게임 종료
                break

    def select_move(self):
        """
        Negamax 알고리즘을 사용하여 최선의 수 선택

        Returns:
            최선의 수 (x1, y1, x2, y2) 또는 None
        """
        legal_moves = self.board.get_legal_moves(self.board.my_symbol)

        if not legal_moves:
            return None

        best_move = None
        best_score = float('-inf')

        # 루트 레벨에서 모든 가능한 수를 평가
        for move in legal_moves:
            # 이동 시뮬레이션
            simulated_board = self.board.copy()
            x1, y1, x2, y2 = move
            simulated_board.make_move(x1, y1, x2, y2, self.board.my_symbol)

            # Negamax 재귀 호출 (상대방 턴, 깊이 감소)
            # 상대방 관점의 점수를 부정하여 내 관점의 점수로 변환
            score = -self.negamax(simulated_board, self.depth - 1, self.board.opp_symbol)

            # 더 좋은 수를 발견하면 업데이트
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def negamax(self, board, depth, current_player):
        """
        Negamax 알고리즘 구현

        Negamax는 현재 플레이어가 항상 최대화한다고 가정하고,
        상대방의 점수를 부정하여 관점을 전환하는 방식

        Args:
            board: 현재 보드 상태
            depth: 남은 탐색 깊이
            current_player: 현재 턴의 플레이어 심볼

        Returns:
            현재 플레이어 관점에서의 최선 평가값
        """
        # 기저 사례 1: 깊이 제한에 도달
        if depth == 0:
            return board.evaluate(current_player)

        # 현재 플레이어의 합법적인 수 생성
        legal_moves = board.get_legal_moves(current_player)

        # 기저 사례 2: 더 이상 둘 수 없음 (게임 종료 또는 PASS)
        if not legal_moves:
            # 현재 상태를 평가하여 반환
            return board.evaluate(current_player)

        # 상대 플레이어 심볼
        opponent = 'X' if current_player == 'O' else 'O'

        # 최선의 점수 초기화
        best_score = float('-inf')

        # 모든 가능한 수를 탐색
        for move in legal_moves:
            # 이동 시뮬레이션
            simulated_board = board.copy()
            x1, y1, x2, y2 = move
            simulated_board.make_move(x1, y1, x2, y2, current_player)

            # 재귀 호출: 상대방 턴
            # 상대방의 최선 점수를 부정하면 현재 플레이어의 점수가 됨
            score = -self.negamax(simulated_board, depth - 1, opponent)

            # 최선의 점수 업데이트
            best_score = max(best_score, score)

        return best_score


if __name__ == "__main__":
    # 기본 탐색 깊이 3으로 에이전트 생성
    agent = MinimaxAgent(depth=3)
    agent.run()
