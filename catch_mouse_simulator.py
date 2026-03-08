#!/usr/bin/env python3
"""
Catch The Mouse (쥐를 잡자) Game Simulator for ALPHANO Problem #2

Runs two C++ agents against each other using the ALPHANO protocol.

Board: 7x11 (rows 1-7, cols 1-11)
  - Row 1: 11 mice (FIRST player)
  - Row 6: 4 cats + 1 nadori (SECOND player)
  - Mice move down 1 row to empty cells
  - Cats slide like chess queens (rows 2-6, empty cells only)
  - Nadori moves like chess king (rows 2-6, can capture mice)
  - Mice win by reaching row 7
  - Cats win by capturing/blocking all mice

Usage:
    python3 catch_mouse_simulator.py agent1.cpp agent2.cpp
    python3 catch_mouse_simulator.py agent1.cpp agent2.cpp -n 10 --swap
    python3 catch_mouse_simulator.py agent1.cpp agent2.cpp -q
"""

import subprocess
import sys
import os
import time
import argparse
import threading
import shutil
from enum import IntEnum

# ============================================================
# Constants
# ============================================================

ROWS = 7
COLS = 11
MAX_TURNS_PER_PLAYER = 200
DEFAULT_TIME_MS = 60000
READY_TIMEOUT_S = 3.0

# 8 directions: right, up-right, up, up-left, left, down-left, down, down-right
DX = [0, -1, -1, -1, 0, 1, 1, 1]
DY = [1, 1, 0, -1, -1, -1, 0, 1]


class Piece(IntEnum):
    EMPTY = 0
    MOUSE = 1   # M - FIRST player
    CAT = 2     # C - SECOND player (queen-like)
    NADORI = 3  # N - SECOND player (king-like, can capture mice)


SYMBOLS = {Piece.EMPTY: '.', Piece.MOUSE: 'M', Piece.CAT: 'C', Piece.NADORI: 'N'}

FIRST = 1   # Mouse player
SECOND = 2  # Cat+Nadori player

# ANSI Colors
C_RESET = '\033[0m'
C_MOUSE = '\033[92m'   # Green for mice
C_CAT = '\033[91m'     # Red for cats
C_NADORI = '\033[95m'  # Magenta for nadori
C_DIM = '\033[90m'
C_BOLD = '\033[1m'
C_YELLOW = '\033[93m'
C_CYAN = '\033[96m'
C_BG_YELLOW = '\033[43m'


# ============================================================
# Board
# ============================================================

class Board:
    def __init__(self):
        # 1-indexed: grid[1..7][1..11]
        self.grid = [[Piece.EMPTY] * (COLS + 1) for _ in range(ROWS + 1)]
        # Initial setup
        for j in range(1, 12):
            self.grid[1][j] = Piece.MOUSE  # 11 mice at row 1
        self.grid[6][4] = Piece.CAT
        self.grid[6][5] = Piece.CAT
        self.grid[6][7] = Piece.CAT
        self.grid[6][8] = Piece.CAT
        self.grid[6][6] = Piece.NADORI

    def count_mice(self):
        cnt = 0
        for i in range(1, ROWS + 1):
            for j in range(1, COLS + 1):
                if self.grid[i][j] == Piece.MOUSE:
                    cnt += 1
        return cnt

    def count_cats(self):
        cnt = 0
        for i in range(1, ROWS + 1):
            for j in range(1, COLS + 1):
                if self.grid[i][j] == Piece.CAT:
                    cnt += 1
        return cnt

    def has_nadori(self):
        for i in range(1, ROWS + 1):
            for j in range(1, COLS + 1):
                if self.grid[i][j] == Piece.NADORI:
                    return True
        return False

    def mice_can_move(self):
        """쥐가 이동 가능한 수가 있는지"""
        for i in range(1, 7):  # rows 1-6
            for j in range(1, COLS + 1):
                if self.grid[i][j] == Piece.MOUSE and self.grid[i + 1][j] == Piece.EMPTY:
                    return True
        return False

    def cats_can_move(self):
        """고양이+까치가 이동 가능한 수가 있는지"""
        for i in range(2, 7):  # rows 2-6
            for j in range(1, COLS + 1):
                if self.grid[i][j] == Piece.CAT:
                    for d in range(8):
                        x, y = i + DX[d], j + DY[d]
                        if 2 <= x <= 6 and 1 <= y <= COLS and self.grid[x][y] == Piece.EMPTY:
                            return True
                elif self.grid[i][j] == Piece.NADORI:
                    for d in range(8):
                        x, y = i + DX[d], j + DY[d]
                        if 2 <= x <= 6 and 1 <= y <= COLS and self.grid[x][y] != Piece.CAT:
                            return True
        return False

    def mouse_on_row7(self):
        """row 7에 도달한 쥐가 있는지"""
        for j in range(1, COLS + 1):
            if self.grid[7][j] == Piece.MOUSE:
                return True
        return False

    # ---- Move Validation ----

    def is_valid_mouse_move(self, x1, y1, x2, y2):
        """쥐 수 합법성 검증"""
        if not (1 <= x1 <= 6 and 1 <= y1 <= COLS):
            return False, "출발지 범위 초과"
        if self.grid[x1][y1] != Piece.MOUSE:
            return False, f"출발지({x1},{y1})에 쥐 없음 (값={self.grid[x1][y1]})"
        if x2 != x1 + 1 or y2 != y1:
            return False, f"쥐는 아래로 1칸만 이동 가능 ({x1},{y1})->({x2},{y2})"
        if not (1 <= x2 <= 7 and 1 <= y2 <= COLS):
            return False, "도착지 범위 초과"
        if self.grid[x2][y2] != Piece.EMPTY:
            return False, f"도착지({x2},{y2}) 비어있지 않음 (값={self.grid[x2][y2]})"
        return True, ""

    def is_valid_cat_move(self, x1, y1, x2, y2):
        """고양이/까치 수 합법성 검증"""
        if not (2 <= x1 <= 6 and 1 <= y1 <= COLS):
            return False, "출발지 범위 초과"
        if not (2 <= x2 <= 6 and 1 <= y2 <= COLS):
            return False, f"도착지 범위 초과 ({x2},{y2}), 고양이/까치는 rows 2-6만 가능"

        piece = self.grid[x1][y1]

        if piece == Piece.CAT:
            # Cat: queen-like slide, empty cells only
            if x1 == x2 and y1 == y2:
                return False, "제자리 이동 불가"
            dx = 0 if x1 == x2 else (1 if x2 > x1 else -1)
            dy = 0 if y1 == y2 else (1 if y2 > y1 else -1)
            # 대각/직선 방향인지 확인
            if dx != 0 and dy != 0 and abs(x2 - x1) != abs(y2 - y1):
                return False, "고양이는 직선/대각선으로만 이동"
            if dx == 0 and dy == 0:
                return False, "제자리 이동 불가"
            # 경로 검증
            x, y = x1, y1
            while True:
                x += dx
                y += dy
                if x < 2 or x > 6 or y < 1 or y > COLS:
                    return False, f"고양이 경로 범위 초과"
                if x == x2 and y == y2:
                    if self.grid[x][y] != Piece.EMPTY:
                        return False, f"도착지({x2},{y2}) 비어있지 않음"
                    return True, ""
                if self.grid[x][y] != Piece.EMPTY:
                    return False, f"경로상 장애물 ({x},{y})={self.grid[x][y]}"
            return False, "도착지에 도달 못함"

        elif piece == Piece.NADORI:
            # Nadori: king-like, 1 step
            d = max(abs(x1 - x2), abs(y1 - y2))
            if d != 1:
                return False, f"까치는 1칸만 이동 (거리={d})"
            if self.grid[x2][y2] == Piece.CAT:
                return False, "까치는 고양이 위치로 이동 불가"
            return True, ""

        else:
            return False, f"출발지({x1},{y1})에 고양이/까치 없음 (값={piece})"

    def is_valid_move(self, x1, y1, x2, y2, side):
        if side == FIRST:
            return self.is_valid_mouse_move(x1, y1, x2, y2)
        else:
            return self.is_valid_cat_move(x1, y1, x2, y2)

    # ---- Apply Move ----

    def apply_move(self, x1, y1, x2, y2, side):
        """수를 적용. 반환: (captured_mouse, description)"""
        if side == FIRST:
            # Mouse moves down
            self.grid[x1][y1] = Piece.EMPTY
            self.grid[x2][y2] = Piece.MOUSE
            if x2 == 7:
                return False, "Mouse reached row 7!"
            return False, "Mouse advance"
        else:
            piece = self.grid[x1][y1]
            captured = self.grid[x2][y2] == Piece.MOUSE
            self.grid[x1][y1] = Piece.EMPTY
            self.grid[x2][y2] = piece
            if captured:
                return True, f"{'Nadori' if piece == Piece.NADORI else 'Cat'} captured mouse!"
            piece_name = "Nadori" if piece == Piece.NADORI else "Cat"
            return False, f"{piece_name} move"

    # ---- Display ----

    def display(self, use_color=True, last_move=None):
        # Header
        cols_str = ' '.join(f"{j:2d}" for j in range(1, COLS + 1))
        if use_color:
            print(f"   {C_DIM}{cols_str}{C_RESET}")
        else:
            print(f"   {cols_str}")

        for i in range(1, ROWS + 1):
            if use_color:
                row = f"{C_DIM}{i:2d}{C_RESET} "
            else:
                row = f"{i:2d} "

            for j in range(1, COLS + 1):
                cell = self.grid[i][j]
                sym = SYMBOLS[cell]
                highlight = last_move and (i, j) == (last_move[2], last_move[3])

                if use_color:
                    if cell == Piece.MOUSE:
                        if highlight:
                            row += f"{C_BOLD}{C_BG_YELLOW}{C_MOUSE}[M]{C_RESET}"
                        else:
                            row += f" {C_MOUSE}{sym}{C_RESET}"
                    elif cell == Piece.CAT:
                        if highlight:
                            row += f"{C_BOLD}{C_BG_YELLOW}{C_CAT}[C]{C_RESET}"
                        else:
                            row += f" {C_CAT}{sym}{C_RESET}"
                    elif cell == Piece.NADORI:
                        if highlight:
                            row += f"{C_BOLD}{C_BG_YELLOW}{C_NADORI}[N]{C_RESET}"
                        else:
                            row += f" {C_NADORI}{sym}{C_RESET}"
                    else:
                        row += f" {C_DIM}.{C_RESET}"
                else:
                    if highlight:
                        row += f"[{sym}]"
                    else:
                        row += f" {sym} "[0:2]  # pad to 2
            print(row)

        # Goal line indicator
        if use_color:
            print(f"   {C_DIM}{'---' * COLS} (row 7 = goal){C_RESET}")
        else:
            print(f"   {'---' * COLS} (row 7 = goal)")

        mc = self.count_mice()
        cc = self.count_cats()
        nd = "Y" if self.has_nadori() else "N"
        if use_color:
            print(f"   {C_MOUSE}Mice: {mc}{C_RESET}  "
                  f"{C_CAT}Cats: {cc}{C_RESET}  "
                  f"{C_NADORI}Nadori: {nd}{C_RESET}")
        else:
            print(f"   Mice: {mc}  Cats: {cc}  Nadori: {nd}")


# ============================================================
# Agent Process (shared with ataxx_simulator)
# ============================================================

class AgentProcess:
    def __init__(self, executable, name):
        self.name = name
        self.executable = executable
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            [self.executable],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
        )

    def send(self, message):
        try:
            self.process.stdin.write(message + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def recv(self, timeout_s):
        result = [None]
        error = [None]

        def reader():
            try:
                result[0] = self.process.stdout.readline()
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        thread.join(timeout=timeout_s)

        if thread.is_alive():
            return None
        if error[0] is not None or result[0] is None:
            return None
        return result[0].strip()

    def close(self):
        if self.process:
            try:
                self.process.stdin.close()
            except:
                pass
            try:
                self.process.terminate()
                self.process.wait(timeout=2)
            except:
                try:
                    self.process.kill()
                except:
                    pass


# ============================================================
# Compiler
# ============================================================

def find_gpp():
    import glob as globmod
    candidates = sorted(globmod.glob('/opt/homebrew/Cellar/gcc/*/bin/g++-*'), reverse=True)
    candidates += sorted(globmod.glob('/usr/local/Cellar/gcc/*/bin/g++-*'), reverse=True)
    for ver in range(20, 9, -1):
        path = shutil.which(f'g++-{ver}')
        if path:
            candidates.append(path)
    default = shutil.which('g++')
    if default:
        candidates.append(default)
    for cand in candidates:
        try:
            out = subprocess.run([cand, '--version'], capture_output=True, text=True, timeout=5)
            if 'clang' not in out.stdout.lower():
                return cand
        except:
            continue
    return 'g++'


def compile_agent(source_path, output_dir=None):
    if not os.path.exists(source_path):
        print(f"Error: {source_path} not found")
        return None
    base = os.path.splitext(os.path.basename(source_path))[0]
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(source_path))
    output_path = os.path.join(output_dir, base)
    gpp = find_gpp()
    print(f"  Compiling {source_path} -> {output_path} ({os.path.basename(gpp)}) ... ", end='', flush=True)
    result = subprocess.run(
        [gpp, '-std=c++17', '-O2', '-o', output_path, source_path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("FAILED")
        print(f"  Compiler error:\n{result.stderr}")
        return None
    print("OK")
    return output_path


def resolve_agent(path, no_compile=False, build_dir=None):
    if path.endswith('.cpp'):
        if no_compile:
            base = os.path.splitext(path)[0]
            if os.path.exists(base):
                return base
            print(f"Error: {base} not found. Compile first or remove --no-compile.")
            return None
        return compile_agent(path, build_dir)
    else:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            return None
        if not os.access(path, os.X_OK):
            print(f"Warning: {path} may not be executable")
        return path


# ============================================================
# Game Referee
# ============================================================

class GameResult:
    def __init__(self):
        self.winner = None       # FIRST (mice) or SECOND (cats) or None
        self.reason = ""
        self.mice_remaining = 0
        self.total_turns = 0
        self.mice_captured = 0
        self.moves = []
        self.first_time_used = 0
        self.second_time_used = 0
        self.error_player = None
        self.error_msg = ""


def run_game(agent1_path, agent2_path, time_ms=DEFAULT_TIME_MS,
             use_color=True, verbose=True, delay=0.0):
    """두 에이전트 간 한 게임을 실행 (FIRST=쥐, SECOND=고양이+까치)"""
    board = Board()
    result = GameResult()

    agents = {
        FIRST: AgentProcess(agent1_path, "FIRST(Mice)"),
        SECOND: AgentProcess(agent2_path, "SECOND(Cats)"),
    }
    remaining_time = {FIRST: time_ms, SECOND: time_ms}
    turn_count = {FIRST: 0, SECOND: 0}
    initial_mice = board.count_mice()  # 11
    mice_captured = 0

    try:
        for agent in agents.values():
            agent.start()

        # --- READY ---
        for player, agent in agents.items():
            role = "FIRST" if player == FIRST else "SECOND"
            agent.send(f"READY {role}")
            response = agent.recv(READY_TIMEOUT_S)
            if response is None:
                result.error_player = player
                result.error_msg = "READY 응답 타임아웃"
                result.winner = SECOND if player == FIRST else FIRST
                result.reason = f"{agent.name} READY 타임아웃"
                return result
            if response != "OK":
                result.error_player = player
                result.error_msg = f"잘못된 READY 응답: '{response}'"
                result.winner = SECOND if player == FIRST else FIRST
                result.reason = f"{agent.name} 잘못된 READY 응답"
                return result

        if verbose:
            print(f"\n{'='*55}")
            print(f"  Catch The Mouse - Game Start")
            print(f"  FIRST(Mice):       {os.path.basename(agent1_path)}")
            print(f"  SECOND(Cats+N):    {os.path.basename(agent2_path)}")
            print(f"  Time: {time_ms}ms per player")
            print(f"{'='*55}")
            board.display(use_color)
            print()

        # --- Main Game Loop ---
        current = FIRST
        total_turns = 0
        game_over = False

        while not game_over:
            agent = agents[current]
            opp = SECOND if current == FIRST else FIRST
            opp_agent = agents[opp]

            # 턴 제한
            if turn_count[current] >= MAX_TURNS_PER_PLAYER:
                if turn_count[opp] >= MAX_TURNS_PER_PLAYER:
                    break
                current = opp
                continue

            # 이동 가능 여부 사전 체크
            if current == FIRST and not board.mice_can_move():
                # 쥐가 이동 불가 → 고양이 승
                result.winner = SECOND
                result.reason = "쥐 이동 불가 (모두 막힘)"
                if verbose:
                    print(f"  {C_CAT if use_color else ''}!! 쥐 이동 불가 - 고양이 승리 !!{C_RESET if use_color else ''}")
                break
            if current == SECOND and not board.cats_can_move():
                # 고양이 이동 불가 (매우 드묾)
                result.winner = FIRST
                result.reason = "고양이 이동 불가"
                if verbose:
                    print(f"  !! 고양이 이동 불가 - 쥐 승리 !!")
                break

            # --- TURN ---
            my_time = remaining_time[current]
            opp_time = remaining_time[opp]
            agent.send(f"TURN {my_time} {opp_time}")

            timeout_s = (my_time / 1000.0) + 0.5
            start_time = time.time()
            response = agent.recv(timeout_s)
            elapsed_ms = int((time.time() - start_time) * 1000)

            remaining_time[current] -= elapsed_ms

            # TLE
            if response is None or remaining_time[current] < -100:
                result.error_player = current
                result.error_msg = "시간 초과 (TLE)"
                result.winner = opp
                result.reason = f"{agent.name} 시간 초과"
                if verbose:
                    tag = C_YELLOW if use_color else ''
                    print(f"\n  {tag}!! {agent.name} 시간 초과 !!{C_RESET if use_color else ''}")
                break

            # 응답 파싱
            if not response.startswith("MOVE "):
                result.error_player = current
                result.error_msg = f"잘못된 응답: '{response}'"
                result.winner = opp
                result.reason = f"{agent.name} 잘못된 응답"
                if verbose:
                    print(f"\n  !! {agent.name} 잘못된 응답: '{response}' !!")
                break

            try:
                parts = response.split()
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except (IndexError, ValueError):
                result.error_player = current
                result.error_msg = f"파싱 실패: '{response}'"
                result.winner = opp
                result.reason = f"{agent.name} 응답 파싱 실패"
                break

            # 합법성 검증
            valid, err_msg = board.is_valid_move(x1, y1, x2, y2, current)
            if not valid:
                result.error_player = current
                result.error_msg = f"비합법 수: ({x1},{y1})->({x2},{y2}) - {err_msg}"
                result.winner = opp
                result.reason = f"{agent.name} 비합법 수 (RE)"
                if verbose:
                    print(f"\n  !! {agent.name} 비합법 수: ({x1},{y1})->({x2},{y2})")
                    print(f"     사유: {err_msg} !!")
                break

            # 수 적용
            captured, desc = board.apply_move(x1, y1, x2, y2, current)
            if captured:
                mice_captured += 1
            turn_count[current] += 1
            total_turns += 1
            result.moves.append((current, x1, y1, x2, y2, elapsed_ms))

            # OPP 전송
            opp_agent.send(f"OPP {x1} {y1} {x2} {y2} {elapsed_ms}")

            # 표시
            if verbose:
                side_str = "Mice" if current == FIRST else "Cats"
                color = C_MOUSE if current == FIRST else C_CAT
                if use_color:
                    print(f"  {color}Turn {total_turns} [{side_str}]{C_RESET}", end='')
                else:
                    print(f"  Turn {total_turns} [{side_str}]", end='')

                piece_at_src = "M" if current == FIRST else ("C" if board.grid[x2][y2] == Piece.CAT or (board.grid[x2][y2] == Piece.EMPTY and captured) else "N")
                # Better: use what's now at destination
                dest_piece = board.grid[x2][y2]
                piece_name = SYMBOLS.get(dest_piece, '?')
                print(f" {piece_name}({x1},{y1})->({x2},{y2})", end='')
                if captured:
                    if use_color:
                        print(f" {C_BOLD}{C_YELLOW}CAPTURE!{C_RESET}", end='')
                    else:
                        print(f" CAPTURE!", end='')
                if current == FIRST and x2 == 7:
                    if use_color:
                        print(f" {C_BOLD}{C_MOUSE}GOAL!{C_RESET}", end='')
                    else:
                        print(f" GOAL!", end='')

                print(f"  ({elapsed_ms}ms, {remaining_time[current]}ms left)")
                board.display(use_color, last_move=(x1, y1, x2, y2))
                print()

            if delay > 0:
                time.sleep(delay)

            # 종료 조건 체크
            # 1) 쥐가 row 7 도달
            if board.mouse_on_row7():
                result.winner = FIRST
                result.reason = "쥐가 row 7 도달 (탈출 성공)"
                game_over = True
                if verbose:
                    tag = f"{C_BOLD}{C_MOUSE}" if use_color else ''
                    print(f"  {tag}** 쥐 탈출 성공! 쥐 승리! **{C_RESET if use_color else ''}")
                break

            # 2) 쥐 전멸
            if board.count_mice() == 0:
                result.winner = SECOND
                result.reason = "쥐 전멸 (모두 포획)"
                game_over = True
                if verbose:
                    tag = f"{C_BOLD}{C_CAT}" if use_color else ''
                    print(f"  {tag}** 쥐 전멸! 고양이 승리! **{C_RESET if use_color else ''}")
                break

            # 3) 턴 제한
            if turn_count[FIRST] >= MAX_TURNS_PER_PLAYER and \
               turn_count[SECOND] >= MAX_TURNS_PER_PLAYER:
                break

            # 다음 턴
            current = opp

        # --- FINISH ---
        for agent in agents.values():
            agent.send("FINISH")

        # --- 결과 집계 ---
        result.mice_remaining = board.count_mice()
        result.mice_captured = mice_captured
        result.total_turns = total_turns
        result.first_time_used = time_ms - remaining_time[FIRST]
        result.second_time_used = time_ms - remaining_time[SECOND]

        # 턴 제한으로 끝난 경우 승자 결정
        if result.winner is None:
            # 쥐가 아직 살아있으면 → 고양이가 막는 데 성공 → 고양이 승
            # (실제 ALPHANO 규칙에 따라 다를 수 있음, 여기선 쥐 탈출 못하면 고양이 승)
            if board.count_mice() > 0:
                result.winner = SECOND
                result.reason = f"턴 제한 종료 (쥐 {result.mice_remaining}마리 생존, 탈출 실패)"
            else:
                result.winner = SECOND
                result.reason = "쥐 전멸"

    finally:
        for agent in agents.values():
            agent.close()

    return result


# ============================================================
# Output
# ============================================================

def print_result(result, agent1_name, agent2_name, use_color=True):
    print(f"{'='*55}")
    print(f"  Game Result")
    print(f"{'='*55}")
    print(f"  FIRST(Mice):    {agent1_name}")
    print(f"  SECOND(Cats):   {agent2_name}")
    print(f"  Total Turns:    {result.total_turns}")
    print(f"  Mice Remaining: {result.mice_remaining}")
    print(f"  Mice Captured:  {result.mice_captured}")
    print(f"  Time Used:      Mice {result.first_time_used}ms / Cats {result.second_time_used}ms")

    if result.error_player:
        print(f"  Error: {result.error_msg}")

    if result.winner == FIRST:
        winner_str = f"Mice ({agent1_name})"
        color = C_MOUSE
    elif result.winner == SECOND:
        winner_str = f"Cats ({agent2_name})"
        color = C_CAT
    else:
        winner_str = "DRAW"
        color = C_YELLOW

    if use_color:
        print(f"\n  {C_BOLD}{color}Winner: {winner_str}{C_RESET}")
    else:
        print(f"\n  Winner: {winner_str}")

    print(f"  Reason: {result.reason}")
    print(f"{'='*55}")


def print_batch_summary(results, agent1_name, agent2_name, use_color=True):
    total = len(results)
    mice_wins = sum(1 for r in results if r.winner == FIRST)
    cats_wins = sum(1 for r in results if r.winner == SECOND)
    errors = sum(1 for r in results if r.error_player is not None)
    avg_mice = sum(r.mice_remaining for r in results) / total
    avg_captured = sum(r.mice_captured for r in results) / total
    avg_turns = sum(r.total_turns for r in results) / total

    print(f"\n{'='*60}")
    print(f"  Batch Results ({total} games)")
    print(f"{'='*60}")
    print(f"  FIRST(Mice):    {agent1_name}")
    print(f"  SECOND(Cats):   {agent2_name}")
    print(f"{'─'*60}")

    if use_color:
        print(f"  {C_MOUSE}Mice Wins: {mice_wins:3d} ({mice_wins/total*100:5.1f}%){C_RESET}")
        print(f"  {C_CAT}Cats Wins: {cats_wins:3d} ({cats_wins/total*100:5.1f}%){C_RESET}")
    else:
        print(f"  Mice Wins: {mice_wins:3d} ({mice_wins/total*100:5.1f}%)")
        print(f"  Cats Wins: {cats_wins:3d} ({cats_wins/total*100:5.1f}%)")

    if errors > 0:
        print(f"  Errors: {errors}")

    print(f"{'─'*60}")
    print(f"  Avg Turns: {avg_turns:.1f}")
    print(f"  Avg Mice Remaining: {avg_mice:.1f}")
    print(f"  Avg Mice Captured: {avg_captured:.1f}")

    print(f"{'─'*60}")
    print(f"  {'#':>3}  {'Winner':^8}  {'Turns':>5}  {'Mice':>4}  {'Cap':>3}  {'Reason'}")
    print(f"  {'─'*56}")
    for i, r in enumerate(results, 1):
        w = "Mice" if r.winner == FIRST else "Cats"
        reason = r.reason[:28]
        print(f"  {i:3d}  {w:^8}  {r.total_turns:5d}  {r.mice_remaining:4d}  {r.mice_captured:3d}  {reason}")

    print(f"{'='*60}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Catch The Mouse (쥐를 잡자) Simulator for ALPHANO Problem #2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s mice.cpp cats.cpp                   # 1판 (FIRST=쥐, SECOND=고양이)
  %(prog)s mice.cpp cats.cpp -n 10             # 10판
  %(prog)s mice.cpp cats.cpp -n 10 --swap      # 10판, 선후공 교대
  %(prog)s mice.cpp cats.cpp -q                # 조용한 모드
  %(prog)s mice.cpp cats.cpp --delay 0.3       # 관전 모드
        """)

    parser.add_argument('agent1', help='첫 번째 에이전트 = FIRST(Mice) (.cpp 또는 실행파일)')
    parser.add_argument('agent2', help='두 번째 에이전트 = SECOND(Cats+Nadori) (.cpp 또는 실행파일)')
    parser.add_argument('-n', '--games', type=int, default=1,
                        help='게임 횟수 (기본: 1)')
    parser.add_argument('-t', '--time', type=int, default=DEFAULT_TIME_MS,
                        help=f'플레이어당 시간(ms) (기본: {DEFAULT_TIME_MS})')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='조용한 모드')
    parser.add_argument('--no-compile', action='store_true',
                        help='컴파일 건너뛰기')
    parser.add_argument('--no-color', action='store_true',
                        help='색상 비활성화')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='턴 간 딜레이 (초)')
    parser.add_argument('--swap', action='store_true',
                        help='배치에서 선후공 교대')
    parser.add_argument('--build-dir', type=str, default=None,
                        help='컴파일 출력 디렉토리')

    args = parser.parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    print("Preparing agents...")
    agent1_exec = resolve_agent(args.agent1, args.no_compile, args.build_dir)
    agent2_exec = resolve_agent(args.agent2, args.no_compile, args.build_dir)

    if not agent1_exec or not agent2_exec:
        print("Error: Failed to prepare agents.")
        sys.exit(1)

    agent1_name = os.path.basename(args.agent1)
    agent2_name = os.path.basename(args.agent2)
    print(f"  Agent 1 (Mice): {agent1_name} ({agent1_exec})")
    print(f"  Agent 2 (Cats): {agent2_name} ({agent2_exec})")
    print()

    results = []
    for game_num in range(1, args.games + 1):
        if args.swap and game_num % 2 == 0:
            first_exec, second_exec = agent2_exec, agent1_exec
            first_name, second_name = agent2_name, agent1_name
            swapped = True
        else:
            first_exec, second_exec = agent1_exec, agent2_exec
            first_name, second_name = agent1_name, agent2_name
            swapped = False

        if args.games > 1:
            swap_tag = " (swapped)" if swapped else ""
            if args.quiet:
                print(f"Game {game_num}/{args.games}{swap_tag}... ", end='', flush=True)
            else:
                print(f"\n{'#'*55}")
                print(f"  Game {game_num}/{args.games}{swap_tag}")
                print(f"  FIRST(Mice): {first_name}")
                print(f"  SECOND(Cats): {second_name}")
                print(f"{'#'*55}")

        game_result = run_game(
            first_exec, second_exec,
            time_ms=args.time,
            use_color=use_color,
            verbose=not args.quiet,
            delay=args.delay,
        )

        # swap 보정
        if swapped:
            orig = GameResult()
            orig.total_turns = game_result.total_turns
            orig.mice_remaining = game_result.mice_remaining
            orig.mice_captured = game_result.mice_captured
            orig.moves = game_result.moves
            orig.error_player = game_result.error_player
            orig.error_msg = game_result.error_msg
            orig.first_time_used = game_result.second_time_used
            orig.second_time_used = game_result.first_time_used
            # swap 시 agent1=cats, agent2=mice → 승자 뒤집기
            if game_result.winner == FIRST:
                orig.winner = SECOND  # agent2가 mice로 이김 → agent2 승
            elif game_result.winner == SECOND:
                orig.winner = FIRST   # agent1이 cats로 이김 → agent1 승
            else:
                orig.winner = None
            orig.reason = game_result.reason + " (swapped)"
            results.append(orig)
        else:
            results.append(game_result)

        if args.quiet:
            r = results[-1]
            w = f"Mice ({agent1_name})" if r.winner == FIRST else f"Cats ({agent2_name})"
            print(f"Mice left: {r.mice_remaining}  Captured: {r.mice_captured}  "
                  f"Turns: {r.total_turns}  Winner: {w}")
        elif args.games == 1:
            print_result(game_result, first_name, second_name, use_color)

    if args.games > 1:
        print_batch_summary(results, agent1_name, agent2_name, use_color)


if __name__ == '__main__':
    main()
