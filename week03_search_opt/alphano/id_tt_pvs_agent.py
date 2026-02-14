#!/usr/bin/env python3
"""
ALPHANO ATAXX Agent - Week 3
탐색 최적화: Iterative Deepening + Transposition Table + PVS

프로토콜:
- READY FIRST/SECOND → print("OK")
- TURN my_time opp_time → print("MOVE x1 y1 x2 y2") (1-indexed)
- OPP x1 y1 x2 y2 → update board
- FINISH → exit

ATAXX 규칙:
- 7x7 보드, 1-indexed 좌표
- Split (거리 1): 복사하여 배치
- Jump (거리 2): 이동
- 인접 8방향 감염
- 이동 불가 시 자동 패스
"""

import sys
import time

# ============================================================================
# 상수 정의
# ============================================================================

EMPTY = 0
BLACK = 1
WHITE = 2
WALL = 3

DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

INF = float('inf')

# TT Flag
PV_NODE = 0   # Exact value
CUT_NODE = 1  # Lower bound (beta cutoff)
ALL_NODE = 2  # Upper bound (alpha cutoff)

# ============================================================================
# Zobrist Hashing
# ============================================================================

def xorshift64(x):
    """64비트 XORShift 의사난수 생성기"""
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7)
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    return x

def init_zobrist():
    """Zobrist 해시 테이블 초기화"""
    zobrist = {}
    seed = 987654321
    for x in range(7):
        for y in range(7):
            for piece in range(4):  # EMPTY, BLACK, WHITE, WALL
                seed = xorshift64(seed)
                zobrist[(x, y, piece)] = seed
    return zobrist

ZOBRIST = init_zobrist()

# ============================================================================
# Transposition Table Entry
# ============================================================================

class TTEntry:
    """Transposition Table 엔트리"""
    __slots__ = ['best_move', 'flag', 'depth', 'value']

    def __init__(self, best_move, flag, depth, value):
        self.best_move = best_move
        self.flag = flag
        self.depth = depth
        self.value = value

# 전역 TT
tt = {}

# ============================================================================
# ATAXX Board
# ============================================================================

class ATAXXBoard:
    """ATAXX 게임 보드"""

    def __init__(self):
        # 7x7 보드 초기화
        self.board = [[EMPTY] * 7 for _ in range(7)]

        # 초기 돌 배치 (0-indexed)
        self.board[0][0] = WHITE
        self.board[0][6] = BLACK
        self.board[6][0] = BLACK
        self.board[6][6] = WHITE

        # 벽 배치 (중앙)
        self.board[3][3] = WALL

        self.current_player = BLACK  # 흑이 선공
        self.hash_value = self.compute_hash()
        self.move_history = []

    def compute_hash(self):
        """전체 보드의 해시값 계산"""
        h = 0
        for x in range(7):
            for y in range(7):
                h ^= ZOBRIST[(x, y, self.board[x][y])]
        return h

    def hash(self):
        """현재 보드의 해시값 반환"""
        return self.hash_value

    def opponent(self, player):
        """상대 플레이어 반환"""
        return WHITE if player == BLACK else BLACK

    def is_valid(self, x, y):
        """좌표가 보드 내부이고 벽이 아닌지 확인"""
        return 0 <= x < 7 and 0 <= y < 7 and self.board[x][y] != WALL

    def legal_moves(self):
        """현재 플레이어의 합법적인 수 목록"""
        moves = []
        player = self.current_player

        for x in range(7):
            for y in range(7):
                if self.board[x][y] != player:
                    continue

                # Split (거리 1) 및 Jump (거리 2)
                for dx in range(-2, 3):
                    for dy in range(-2, 3):
                        if dx == 0 and dy == 0:
                            continue
                        dist = max(abs(dx), abs(dy))
                        if dist > 2:
                            continue

                        nx, ny = x + dx, y + dy
                        if self.is_valid(nx, ny) and self.board[nx][ny] == EMPTY:
                            moves.append((x, y, nx, ny))

        return moves

    def make_move(self, move):
        """수를 실행하고 해시 업데이트"""
        x1, y1, x2, y2 = move
        player = self.current_player
        opponent = self.opponent(player)

        dist = max(abs(x2 - x1), abs(y2 - y1))

        # 이전 상태 해시 제거
        self.hash_value ^= ZOBRIST[(x2, y2, EMPTY)]
        self.hash_value ^= ZOBRIST[(x2, y2, player)]

        # 목적지에 돌 배치
        self.board[x2][y2] = player

        # Jump이면 출발지 비우기
        if dist == 2:
            self.hash_value ^= ZOBRIST[(x1, y1, player)]
            self.hash_value ^= ZOBRIST[(x1, y1, EMPTY)]
            self.board[x1][y1] = EMPTY

        # 인접 8방향 감염
        infected = []
        for dx, dy in DIRECTIONS:
            nx, ny = x2 + dx, y2 + dy
            if self.is_valid(nx, ny) and self.board[nx][ny] == opponent:
                infected.append((nx, ny))
                self.hash_value ^= ZOBRIST[(nx, ny, opponent)]
                self.hash_value ^= ZOBRIST[(nx, ny, player)]
                self.board[nx][ny] = player

        # Undo를 위한 기록
        self.move_history.append((move, dist, infected))
        self.current_player = opponent

    def undo_move(self):
        """마지막 수를 되돌림"""
        move, dist, infected = self.move_history.pop()
        x1, y1, x2, y2 = move

        # 플레이어 복원
        self.current_player = self.opponent(self.current_player)
        player = self.current_player
        opponent = self.opponent(player)

        # 목적지 비우기
        self.hash_value ^= ZOBRIST[(x2, y2, player)]
        self.hash_value ^= ZOBRIST[(x2, y2, EMPTY)]
        self.board[x2][y2] = EMPTY

        # Jump이었으면 출발지 복원
        if dist == 2:
            self.hash_value ^= ZOBRIST[(x1, y1, EMPTY)]
            self.hash_value ^= ZOBRIST[(x1, y1, player)]
            self.board[x1][y1] = player

        # 감염 되돌리기
        for nx, ny in infected:
            self.hash_value ^= ZOBRIST[(nx, ny, player)]
            self.hash_value ^= ZOBRIST[(nx, ny, opponent)]
            self.board[nx][ny] = opponent

    def count_pieces(self):
        """돌 개수 세기"""
        black_count = 0
        white_count = 0
        for row in self.board:
            for cell in row:
                if cell == BLACK:
                    black_count += 1
                elif cell == WHITE:
                    white_count += 1
        return black_count, white_count

    def evaluate(self):
        """평가 함수: 돌 수 차이 + 이동성"""
        black_count, white_count = self.count_pieces()
        mobility = len(self.legal_moves())

        # 현재 플레이어 기준 평가
        if self.current_player == BLACK:
            piece_diff = black_count - white_count
        else:
            piece_diff = white_count - black_count

        return piece_diff + mobility * 0.1

    def is_terminal(self):
        """게임 종료 확인"""
        black_count, white_count = self.count_pieces()
        if black_count == 0 or white_count == 0:
            return True
        if black_count + white_count == 49 - 1:  # 49칸 중 벽 1칸 제외
            return True
        # 양쪽 모두 수가 없으면 종료 (여기서는 간단히 생략)
        return False

# ============================================================================
# Principal Variation Search with Transposition Table
# ============================================================================

def pvs(board, depth, alpha, beta, start_time, time_limit):
    """
    Principal Variation Search with Transposition Table

    Args:
        board: ATAXXBoard
        depth: 탐색 깊이
        alpha: Alpha 값
        beta: Beta 값
        start_time: 탐색 시작 시간
        time_limit: 시간 제한 (ms)

    Returns:
        (value, best_move)
    """
    # 시간 체크
    elapsed = (time.time() - start_time) * 1000
    if elapsed > time_limit * 0.95:
        return board.evaluate(), None

    alpha_original = alpha

    # TT 조회
    board_hash = board.hash()
    tt_move = None
    if board_hash in tt:
        entry = tt[board_hash]
        tt_move = entry.best_move

        # TT Cutoff (선택사항, 얕은 탐색에서는 오버헤드)
        # ATAXX에서는 효과가 미미하므로 주석 처리 가능
        # if entry.depth >= depth:
        #     if entry.flag == PV_NODE:
        #         return entry.value, entry.best_move
        #     elif entry.flag == CUT_NODE and entry.value >= beta:
        #         return entry.value, entry.best_move
        #     elif entry.flag == ALL_NODE and entry.value <= alpha:
        #         return entry.value, entry.best_move

    # 종료 조건
    if depth == 0 or board.is_terminal():
        return board.evaluate(), None

    # 수 생성
    moves = board.legal_moves()

    # 이동 불가 시 패스 (평가만 반환)
    if not moves:
        return board.evaluate(), None

    # Move Ordering: TT move를 맨 앞으로
    if tt_move and tt_move in moves:
        moves.remove(tt_move)
        moves.insert(0, tt_move)

    # 탐색
    best_value = -INF
    best_move = None

    for i, move in enumerate(moves):
        board.make_move(move)

        if i == 0:
            # 첫 번째 자식: Full window
            value, _ = pvs(board, depth - 1, -beta, -alpha, start_time, time_limit)
            value = -value
        else:
            # 나머지: Null window로 빠르게 확인
            value, _ = pvs(board, depth - 1, -alpha - 1, -alpha, start_time, time_limit)
            value = -value

            # Null window 실패 시 재탐색
            if alpha < value < beta:
                value, _ = pvs(board, depth - 1, -beta, -value, start_time, time_limit)
                value = -value

        board.undo_move()

        if value > best_value:
            best_value = value
            best_move = move

        alpha = max(alpha, value)
        if alpha >= beta:
            break  # Beta cutoff

    # TT 저장
    if best_value <= alpha_original:
        flag = ALL_NODE
    elif best_value >= beta:
        flag = CUT_NODE
    else:
        flag = PV_NODE

    # Depth-preferred replace
    if board_hash not in tt or tt[board_hash].depth <= depth:
        tt[board_hash] = TTEntry(best_move, flag, depth, best_value)

    return best_value, best_move

# ============================================================================
# Iterative Deepening
# ============================================================================

def iterative_deepening(board, time_limit):
    """
    Iterative Deepening with PVS and TT

    Args:
        board: ATAXXBoard
        time_limit: 시간 제한 (ms)

    Returns:
        best_move
    """
    start_time = time.time()
    best_move = None

    for depth in range(1, 50):
        elapsed = (time.time() - start_time) * 1000
        if elapsed > time_limit * 0.85:  # 85%에서 중단
            break

        value, move = pvs(board, depth, -INF, INF, start_time, time_limit)

        if move:
            best_move = move
        else:
            # 시간 초과로 완료하지 못함
            break

    return best_move

# ============================================================================
# 시간 관리
# ============================================================================

def calculate_time_limit(my_time):
    """
    턴당 시간 배분 계산

    Args:
        my_time: 남은 시간 (ms)

    Returns:
        턴당 시간 제한 (ms)
    """
    if my_time > 60000:  # 60초 이상
        return 50
    elif my_time > 20000:  # 20초 이상
        return 150
    else:
        return 10  # 안전 마진

# ============================================================================
# ALPHANO 프로토콜
# ============================================================================

def main():
    board = ATAXXBoard()
    my_color = None

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        tokens = line.split()
        command = tokens[0]

        if command == "READY":
            # READY FIRST 또는 READY SECOND
            position = tokens[1]
            if position == "FIRST":
                my_color = BLACK
            else:
                my_color = WHITE
            print("OK")
            sys.stdout.flush()

        elif command == "TURN":
            # TURN my_time opp_time
            my_time = int(tokens[1])
            opp_time = int(tokens[2])

            # 시간 배분 계산
            time_limit = calculate_time_limit(my_time)

            # Iterative Deepening으로 최선의 수 탐색
            best_move = iterative_deepening(board, time_limit)

            if best_move:
                x1, y1, x2, y2 = best_move
                # 1-indexed로 변환
                print(f"MOVE {x1+1} {y1+1} {x2+1} {y2+1}")
                sys.stdout.flush()

                # 보드 업데이트
                board.make_move(best_move)
            else:
                # 이동 불가 (패스)
                print("PASS")
                sys.stdout.flush()

        elif command == "OPP":
            # OPP x1 y1 x2 y2 (1-indexed)
            x1 = int(tokens[1]) - 1
            y1 = int(tokens[2]) - 1
            x2 = int(tokens[3]) - 1
            y2 = int(tokens[4]) - 1

            # 상대 수 업데이트
            board.make_move((x1, y1, x2, y2))

        elif command == "FINISH":
            # 게임 종료
            break

if __name__ == "__main__":
    main()
