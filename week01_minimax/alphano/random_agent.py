"""
ATAXX (세균 전쟁) - Random Agent
ALPHANO 프로토콜을 따르는 랜덤 에이전트

게임 규칙:
- 7x7 보드
- O 플레이어: (1,1), (7,7) 시작
- X 플레이어: (1,7), (7,1) 시작
- 이동 종류:
  * Split (분열): 거리 1 이동 (원본 유지, 복제)
  * Jump (점프): 거리 2 이동 (원본 이동)
- 이동 후 인접한 8방향 적 말을 아군으로 변환
- 승리 조건: 상대 말 제거, 빈 칸 없음, 400턴 도달 시 많은 쪽 승리
"""

import sys
import random


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

    def get_legal_moves(self):
        """
        현재 플레이어의 모든 합법적인 이동 반환
        반환값: [(x1, y1, x2, y2), ...] (1-indexed 좌표)
        """
        moves = []

        # 내 모든 말의 위치 찾기
        for i in range(7):
            for j in range(7):
                if self.board[i][j] == self.my_symbol:
                    # 이 말로부터 가능한 모든 이동 생성
                    moves.extend(self._get_moves_from(i, j))

        return moves

    def _get_moves_from(self, row, col):
        """
        특정 위치(row, col)에서 가능한 모든 이동 생성
        - 거리 1: Split (분열)
        - 거리 2: Jump (점프)
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


class RandomAgent:
    """랜덤하게 합법적인 수를 선택하는 에이전트"""

    def __init__(self):
        self.board = AtaxxBoard()

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
        """랜덤하게 합법적인 수 선택"""
        legal_moves = self.board.get_legal_moves()

        if not legal_moves:
            return None

        # 모든 합법적인 수 중 무작위로 선택
        return random.choice(legal_moves)


if __name__ == "__main__":
    agent = RandomAgent()
    agent.run()
