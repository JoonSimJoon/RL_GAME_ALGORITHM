#!/usr/bin/env python3
"""
Ataxx (세균전쟁) Game Simulator for ALPHANO Problem #1

Runs two C++ agents against each other using the ALPHANO protocol.
Supports auto-compilation, visual display, time tracking, and batch mode.

Usage:
    python3 ataxx_simulator.py agent1.cpp agent2.cpp
    python3 ataxx_simulator.py agent1.cpp agent2.cpp --games 10
    python3 ataxx_simulator.py ./agent1 ./agent2 --no-compile
    python3 ataxx_simulator.py agent1.cpp agent2.cpp --time 60000 --quiet
"""

import subprocess
import sys
import os
import time
import argparse
import threading
import tempfile
import shutil
from enum import IntEnum

# ============================================================
# Constants
# ============================================================

BOARD_SIZE = 7
MAX_TURNS_PER_PLAYER = 200  # 각 플레이어 최대 200턴 (총 400턴)
DEFAULT_TIME_MS = 60000     # 기본 60초
READY_TIMEOUT_S = 3.0       # READY 응답 제한 3초

DX = [-1, -1, -1, 0, 0, 1, 1, 1]
DY = [-1, 0, 1, -1, 1, -1, 0, 1]


class Piece(IntEnum):
    EMPTY = 0
    FIRST = 1   # O
    SECOND = 2  # X


SYMBOLS = {Piece.EMPTY: '.', Piece.FIRST: 'O', Piece.SECOND: 'X'}

# ANSI Colors
COLOR_RESET = '\033[0m'
COLOR_FIRST = '\033[94m'   # Blue for O
COLOR_SECOND = '\033[91m'  # Red for X
COLOR_DIM = '\033[90m'     # Gray for dots
COLOR_BOLD = '\033[1m'
COLOR_GREEN = '\033[92m'
COLOR_YELLOW = '\033[93m'
COLOR_CYAN = '\033[96m'


# ============================================================
# Board
# ============================================================

class Board:
    def __init__(self):
        # 1-indexed board[1..7][1..7]
        self.grid = [[Piece.EMPTY] * (BOARD_SIZE + 1) for _ in range(BOARD_SIZE + 1)]
        self.grid[1][1] = Piece.FIRST
        self.grid[7][7] = Piece.FIRST
        self.grid[1][7] = Piece.SECOND
        self.grid[7][1] = Piece.SECOND

    def count(self, player):
        cnt = 0
        for i in range(1, 8):
            for j in range(1, 8):
                if self.grid[i][j] == player:
                    cnt += 1
        return cnt

    def empty_count(self):
        cnt = 0
        for i in range(1, 8):
            for j in range(1, 8):
                if self.grid[i][j] == Piece.EMPTY:
                    cnt += 1
        return cnt

    def has_moves(self, player):
        for x1 in range(1, 8):
            for y1 in range(1, 8):
                if self.grid[x1][y1] != player:
                    continue
                for x2 in range(x1 - 2, x1 + 3):
                    if x2 < 1 or x2 > 7:
                        continue
                    for y2 in range(y1 - 2, y1 + 3):
                        if y2 < 1 or y2 > 7:
                            continue
                        if x2 == x1 and y2 == y1:
                            continue
                        if self.grid[x2][y2] == Piece.EMPTY:
                            return True
        return False

    def is_valid_move(self, x1, y1, x2, y2, player):
        """합법적인 수인지 검증"""
        # PASS
        if x1 == -1 and y1 == -1 and x2 == -1 and y2 == -1:
            return not self.has_moves(player)

        # 좌표 범위
        if not (1 <= x1 <= 7 and 1 <= y1 <= 7):
            return False
        if not (1 <= x2 <= 7 and 1 <= y2 <= 7):
            return False
        # 출발지에 내 말이 있어야 함
        if self.grid[x1][y1] != player:
            return False
        # 도착지가 비어있어야 함
        if self.grid[x2][y2] != Piece.EMPTY:
            return False
        # 거리 1 또는 2
        dist = max(abs(x1 - x2), abs(y1 - y2))
        if dist < 1 or dist > 2:
            return False
        return True

    def apply_move(self, x1, y1, x2, y2, player):
        """수를 적용하고 감염된 말 수를 반환"""
        if x1 == -1 and y1 == -1 and x2 == -1 and y2 == -1:
            return 0  # PASS

        dist = max(abs(x1 - x2), abs(y1 - y2))
        if dist == 2:
            self.grid[x1][y1] = Piece.EMPTY  # jump: 원래 자리 비움
        # split(d=1): 원래 자리 유지, 새 자리에도 생성
        self.grid[x2][y2] = player

        # 감염: 도착지 인접 8방향의 상대 말 변환
        opp = Piece.FIRST if player == Piece.SECOND else Piece.SECOND
        infected = 0
        for d in range(8):
            nx, ny = x2 + DX[d], y2 + DY[d]
            if 1 <= nx <= 7 and 1 <= ny <= 7:
                if self.grid[nx][ny] == opp:
                    self.grid[nx][ny] = player
                    infected += 1
        return infected

    def display(self, use_color=True, last_move=None):
        """보드를 시각적으로 출력"""
        if use_color:
            header = f"  {COLOR_DIM}1 2 3 4 5 6 7{COLOR_RESET}"
        else:
            header = "  1 2 3 4 5 6 7"
        print(header)

        for i in range(1, 8):
            if use_color:
                row = f"{COLOR_DIM}{i}{COLOR_RESET} "
            else:
                row = f"{i} "
            for j in range(1, 8):
                cell = self.grid[i][j]
                sym = SYMBOLS[cell]

                # 마지막 수의 도착지 하이라이트
                highlight = last_move and (i, j) == (last_move[2], last_move[3])

                if use_color:
                    if cell == Piece.FIRST:
                        if highlight:
                            row += f"{COLOR_BOLD}{COLOR_FIRST}[O]{COLOR_RESET}"
                        else:
                            row += f"{COLOR_FIRST}{sym}{COLOR_RESET} "
                    elif cell == Piece.SECOND:
                        if highlight:
                            row += f"{COLOR_BOLD}{COLOR_SECOND}[X]{COLOR_RESET}"
                        else:
                            row += f"{COLOR_SECOND}{sym}{COLOR_RESET} "
                    else:
                        row += f"{COLOR_DIM}{sym}{COLOR_RESET} "
                else:
                    row += f"{sym} "
            print(row)

        fc = self.count(Piece.FIRST)
        sc = self.count(Piece.SECOND)
        if use_color:
            print(f"  {COLOR_FIRST}O(FIRST): {fc}{COLOR_RESET}  "
                  f"{COLOR_SECOND}X(SECOND): {sc}{COLOR_RESET}  "
                  f"{COLOR_DIM}Empty: {self.empty_count()}{COLOR_RESET}")
        else:
            print(f"  O(FIRST): {fc}  X(SECOND): {sc}  Empty: {self.empty_count()}")

    def copy(self):
        b = Board.__new__(Board)
        b.grid = [row[:] for row in self.grid]
        return b


# ============================================================
# Agent Process
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
            bufsize=0,  # unbuffered
        )

    def send(self, message):
        try:
            self.process.stdin.write(message + '\n')
            self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            return False
        return True

    def recv(self, timeout_s):
        """한 줄 읽기 (타임아웃 포함)"""
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
            return None  # 타임아웃
        if error[0] is not None:
            return None
        if result[0] is None:
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
    """bits/stdc++.h를 지원하는 g++ 경로를 찾음 (GCC 우선, clang은 미지원)"""
    import glob as globmod

    # 1) Homebrew GCC (macOS)
    candidates = sorted(globmod.glob('/opt/homebrew/Cellar/gcc/*/bin/g++-*'), reverse=True)
    candidates += sorted(globmod.glob('/usr/local/Cellar/gcc/*/bin/g++-*'), reverse=True)

    # 2) 시스템 g++-XX
    for ver in range(20, 9, -1):
        path = shutil.which(f'g++-{ver}')
        if path:
            candidates.append(path)

    # 3) 기본 g++ (GCC인 경우만)
    default = shutil.which('g++')
    if default:
        candidates.append(default)

    for cand in candidates:
        try:
            out = subprocess.run([cand, '--version'], capture_output=True, text=True, timeout=5)
            # clang은 bits/stdc++.h 미지원 → GCC만 사용
            if 'clang' not in out.stdout.lower():
                return cand
        except:
            continue

    return 'g++'  # fallback


def compile_agent(source_path, output_dir=None):
    """C++ 소스를 컴파일하여 실행 파일 경로를 반환"""
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


# ============================================================
# Game Referee
# ============================================================

class GameResult:
    def __init__(self):
        self.winner = None       # Piece.FIRST, Piece.SECOND, or None (draw)
        self.reason = ""         # 승리 사유
        self.first_score = 0
        self.second_score = 0
        self.total_turns = 0
        self.moves = []          # 전체 수 기록
        self.first_time_used = 0
        self.second_time_used = 0
        self.error_player = None # 에러 발생 플레이어
        self.error_msg = ""      # 에러 메시지


def run_game(agent1_path, agent2_path, time_ms=DEFAULT_TIME_MS,
             use_color=True, verbose=True, delay=0.0):
    """두 에이전트 간 한 게임을 실행"""
    board = Board()
    result = GameResult()

    # 에이전트 프로세스 시작
    agents = {
        Piece.FIRST: AgentProcess(agent1_path, "FIRST(O)"),
        Piece.SECOND: AgentProcess(agent2_path, "SECOND(X)"),
    }
    remaining_time = {Piece.FIRST: time_ms, Piece.SECOND: time_ms}
    turn_count = {Piece.FIRST: 0, Piece.SECOND: 0}

    try:
        for agent in agents.values():
            agent.start()

        # --- READY ---
        for player, agent in agents.items():
            role = "FIRST" if player == Piece.FIRST else "SECOND"
            agent.send(f"READY {role}")
            response = agent.recv(READY_TIMEOUT_S)
            if response is None:
                result.error_player = player
                result.error_msg = "READY 응답 타임아웃"
                result.winner = Piece.SECOND if player == Piece.FIRST else Piece.FIRST
                result.reason = f"{agent.name} READY 타임아웃"
                return result
            if response != "OK":
                result.error_player = player
                result.error_msg = f"잘못된 READY 응답: '{response}'"
                result.winner = Piece.SECOND if player == Piece.FIRST else Piece.FIRST
                result.reason = f"{agent.name} 잘못된 READY 응답"
                return result

        if verbose:
            print(f"\n{'='*45}")
            print(f"  Game Start")
            print(f"  FIRST(O): {os.path.basename(agent1_path)}")
            print(f"  SECOND(X): {os.path.basename(agent2_path)}")
            print(f"  Time: {time_ms}ms per player")
            print(f"{'='*45}")
            board.display(use_color)
            print()

        # --- Main Game Loop ---
        current = Piece.FIRST
        total_turns = 0
        consecutive_pass = 0

        while True:
            agent = agents[current]
            opp_player = Piece.SECOND if current == Piece.FIRST else Piece.FIRST
            opp_agent = agents[opp_player]

            # 턴 제한 확인
            if turn_count[current] >= MAX_TURNS_PER_PLAYER:
                # 양쪽 200턴 다 사용한 경우
                if turn_count[opp_player] >= MAX_TURNS_PER_PLAYER:
                    break
                # 한쪽만 다 사용한 경우 상대 턴으로
                current = opp_player
                continue

            # --- TURN ---
            my_time = remaining_time[current]
            opp_time = remaining_time[opp_player]
            agent.send(f"TURN {my_time} {opp_time}")

            # 응답 대기 (남은 시간 + 여유 0.5초)
            timeout_s = (my_time / 1000.0) + 0.5
            start_time = time.time()
            response = agent.recv(timeout_s)
            elapsed_ms = int((time.time() - start_time) * 1000)

            # 시간 차감
            remaining_time[current] -= elapsed_ms

            # TLE 체크
            if response is None or remaining_time[current] < -100:
                result.error_player = current
                result.error_msg = "시간 초과 (TLE)"
                result.winner = opp_player
                result.reason = f"{agent.name} 시간 초과"
                if verbose:
                    print(f"\n  {COLOR_YELLOW if use_color else ''}!! {agent.name} 시간 초과 !!{COLOR_RESET if use_color else ''}")
                break

            # 응답 파싱
            if not response.startswith("MOVE "):
                result.error_player = current
                result.error_msg = f"잘못된 응답 형식: '{response}'"
                result.winner = opp_player
                result.reason = f"{agent.name} 잘못된 응답"
                if verbose:
                    print(f"\n  !! {agent.name} 잘못된 응답: '{response}' !!")
                break

            try:
                parts = response.split()
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
            except (IndexError, ValueError):
                result.error_player = current
                result.error_msg = f"응답 파싱 실패: '{response}'"
                result.winner = opp_player
                result.reason = f"{agent.name} 응답 파싱 실패"
                break

            # 합법성 검증
            if not board.is_valid_move(x1, y1, x2, y2, current):
                result.error_player = current
                result.error_msg = f"비합법 수: {x1} {y1} {x2} {y2}"
                result.winner = opp_player
                result.reason = f"{agent.name} 비합법 수 (RE)"
                if verbose:
                    print(f"\n  !! {agent.name} 비합법 수: {x1} {y1} -> {x2} {y2} !!")
                break

            # 수 적용
            is_pass = (x1 == -1)
            infected = board.apply_move(x1, y1, x2, y2, current)
            turn_count[current] += 1
            total_turns += 1
            result.moves.append((current, x1, y1, x2, y2, elapsed_ms))

            # 상대에게 OPP 전송
            opp_agent.send(f"OPP {x1} {y1} {x2} {y2} {elapsed_ms}")

            # 표시
            if verbose:
                role = "O" if current == Piece.FIRST else "X"
                color = COLOR_FIRST if current == Piece.FIRST else COLOR_SECOND
                if use_color:
                    print(f"  {color}Turn {total_turns} [{role}]{COLOR_RESET}", end='')
                else:
                    print(f"  Turn {total_turns} [{role}]", end='')

                if is_pass:
                    print(" PASS", end='')
                else:
                    dist = max(abs(x1 - x2), abs(y1 - y2))
                    move_type = "Split" if dist == 1 else "Jump"
                    print(f" ({x1},{y1})->({x2},{y2}) [{move_type}]", end='')
                    if infected > 0:
                        print(f" +{infected} infected", end='')

                print(f"  ({elapsed_ms}ms, {remaining_time[current]}ms left)")
                board.display(use_color, last_move=None if is_pass else (x1, y1, x2, y2))
                print()

            if delay > 0:
                time.sleep(delay)

            # PASS 연속 체크
            if is_pass:
                consecutive_pass += 1
                if consecutive_pass >= 2:
                    break  # 양쪽 모두 PASS
            else:
                consecutive_pass = 0

            # 게임 종료 조건
            fc = board.count(Piece.FIRST)
            sc = board.count(Piece.SECOND)

            if fc == 0 or sc == 0:
                break  # 한 쪽 전멸
            if board.empty_count() == 0:
                break  # 보드 가득 참
            if turn_count[Piece.FIRST] >= MAX_TURNS_PER_PLAYER and \
               turn_count[Piece.SECOND] >= MAX_TURNS_PER_PLAYER:
                break  # 400턴 소진

            # 다음 턴
            current = opp_player

        # --- FINISH ---
        for agent in agents.values():
            agent.send("FINISH")

        # --- 결과 집계 ---
        result.first_score = board.count(Piece.FIRST)
        result.second_score = board.count(Piece.SECOND)
        result.total_turns = total_turns
        result.first_time_used = time_ms - remaining_time[Piece.FIRST]
        result.second_time_used = time_ms - remaining_time[Piece.SECOND]

        if result.winner is None:  # 에러로 승부 결정 안 된 경우
            if result.first_score > result.second_score:
                result.winner = Piece.FIRST
                result.reason = f"O가 더 많음 ({result.first_score} vs {result.second_score})"
            elif result.second_score > result.first_score:
                result.winner = Piece.SECOND
                result.reason = f"X가 더 많음 ({result.second_score} vs {result.first_score})"
            else:
                result.winner = None
                result.reason = f"무승부 ({result.first_score} vs {result.second_score})"

    finally:
        for agent in agents.values():
            agent.close()

    return result


# ============================================================
# Main
# ============================================================

def print_result(result, agent1_name, agent2_name, use_color=True):
    """게임 결과를 출력"""
    print(f"{'='*45}")
    print(f"  Game Result")
    print(f"{'='*45}")
    print(f"  FIRST(O):  {agent1_name}")
    print(f"  SECOND(X): {agent2_name}")
    print(f"  Total Turns: {result.total_turns}")
    print(f"  Score: O {result.first_score} - {result.second_score} X")
    print(f"  Time Used: O {result.first_time_used}ms / X {result.second_time_used}ms")

    if result.error_player:
        print(f"  Error: {result.error_msg}")

    if result.winner == Piece.FIRST:
        winner_str = f"O ({agent1_name})"
        color = COLOR_FIRST
    elif result.winner == Piece.SECOND:
        winner_str = f"X ({agent2_name})"
        color = COLOR_SECOND
    else:
        winner_str = "DRAW"
        color = COLOR_YELLOW

    if use_color:
        print(f"\n  {COLOR_BOLD}{color}Winner: {winner_str}{COLOR_RESET}")
    else:
        print(f"\n  Winner: {winner_str}")

    print(f"  Reason: {result.reason}")
    print(f"{'='*45}")


def print_batch_summary(results, agent1_name, agent2_name, use_color=True):
    """배치 결과 요약 출력"""
    total = len(results)
    first_wins = sum(1 for r in results if r.winner == Piece.FIRST)
    second_wins = sum(1 for r in results if r.winner == Piece.SECOND)
    draws = sum(1 for r in results if r.winner is None)
    errors = sum(1 for r in results if r.error_player is not None)

    avg_first_score = sum(r.first_score for r in results) / total
    avg_second_score = sum(r.second_score for r in results) / total
    avg_turns = sum(r.total_turns for r in results) / total

    print(f"\n{'='*50}")
    print(f"  Batch Results ({total} games)")
    print(f"{'='*50}")
    print(f"  FIRST(O):  {agent1_name}")
    print(f"  SECOND(X): {agent2_name}")
    print(f"{'─'*50}")

    if use_color:
        print(f"  {COLOR_FIRST}O Wins: {first_wins:3d} ({first_wins/total*100:5.1f}%){COLOR_RESET}")
        print(f"  {COLOR_SECOND}X Wins: {second_wins:3d} ({second_wins/total*100:5.1f}%){COLOR_RESET}")
        print(f"  {COLOR_YELLOW}Draws:  {draws:3d} ({draws/total*100:5.1f}%){COLOR_RESET}")
    else:
        print(f"  O Wins: {first_wins:3d} ({first_wins/total*100:5.1f}%)")
        print(f"  X Wins: {second_wins:3d} ({second_wins/total*100:5.1f}%)")
        print(f"  Draws:  {draws:3d} ({draws/total*100:5.1f}%)")

    if errors > 0:
        print(f"  Errors: {errors}")

    print(f"{'─'*50}")
    print(f"  Avg Score: O {avg_first_score:.1f} - {avg_second_score:.1f} X")
    print(f"  Avg Turns: {avg_turns:.1f}")

    # 개별 결과 테이블
    print(f"{'─'*50}")
    print(f"  {'#':>3}  {'Winner':^8}  {'Score':^9}  {'Turns':>5}  {'Reason'}")
    print(f"  {'─'*46}")
    for i, r in enumerate(results, 1):
        if r.winner == Piece.FIRST:
            w = "O"
        elif r.winner == Piece.SECOND:
            w = "X"
        else:
            w = "Draw"
        score = f"{r.first_score}-{r.second_score}"
        reason = r.reason[:25]
        print(f"  {i:3d}  {w:^8}  {score:^9}  {r.total_turns:5d}  {reason}")

    print(f"{'='*50}")


def resolve_agent(path, no_compile=False, build_dir=None):
    """에이전트 경로 해석: .cpp면 컴파일, 아니면 그대로 반환"""
    if path.endswith('.cpp'):
        if no_compile:
            # .cpp인데 --no-compile이면 이미 컴파일된 것으로 간주
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


def main():
    parser = argparse.ArgumentParser(
        description='Ataxx (세균전쟁) Game Simulator for ALPHANO Problem #1',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s agent1.cpp agent2.cpp              # 컴파일 후 1판
  %(prog)s agent1.cpp agent2.cpp -n 10        # 10판 (배치)
  %(prog)s agent1.cpp agent2.cpp -n 10 --swap # 10판, 선후공 교대
  %(prog)s ./a1 ./a2 --no-compile             # 이미 컴파일된 에이전트
  %(prog)s agent1.cpp agent2.cpp -q           # 조용한 모드 (결과만)
  %(prog)s agent1.cpp agent2.cpp --delay 0.5  # 0.5초 딜레이
  %(prog)s agent1.cpp agent2.cpp --no-color   # 색상 없이
        """)

    parser.add_argument('agent1', help='첫 번째 에이전트 (.cpp 또는 실행파일)')
    parser.add_argument('agent2', help='두 번째 에이전트 (.cpp 또는 실행파일)')
    parser.add_argument('-n', '--games', type=int, default=1,
                        help='게임 횟수 (기본: 1)')
    parser.add_argument('-t', '--time', type=int, default=DEFAULT_TIME_MS,
                        help=f'플레이어당 시간(ms) (기본: {DEFAULT_TIME_MS})')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='조용한 모드 (결과만 출력)')
    parser.add_argument('--no-compile', action='store_true',
                        help='C++ 컴파일 건너뛰기')
    parser.add_argument('--no-color', action='store_true',
                        help='ANSI 색상 비활성화')
    parser.add_argument('--delay', type=float, default=0.0,
                        help='턴 간 딜레이 (초)')
    parser.add_argument('--swap', action='store_true',
                        help='배치 모드에서 선후공 교대')
    parser.add_argument('--build-dir', type=str, default=None,
                        help='컴파일 출력 디렉토리')

    args = parser.parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    # 컴파일 / 경로 해석
    print("Preparing agents...")
    agent1_exec = resolve_agent(args.agent1, args.no_compile, args.build_dir)
    agent2_exec = resolve_agent(args.agent2, args.no_compile, args.build_dir)

    if not agent1_exec or not agent2_exec:
        print("Error: Failed to prepare agents.")
        sys.exit(1)

    agent1_name = os.path.basename(args.agent1)
    agent2_name = os.path.basename(args.agent2)
    print(f"  Agent 1: {agent1_name} ({agent1_exec})")
    print(f"  Agent 2: {agent2_name} ({agent2_exec})")
    print()

    # 게임 실행
    results = []
    for game_num in range(1, args.games + 1):
        # 선후공 결정
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
                print(f"\n{'#'*45}")
                print(f"  Game {game_num}/{args.games}{swap_tag}")
                print(f"  FIRST(O): {first_name}")
                print(f"  SECOND(X): {second_name}")
                print(f"{'#'*45}")

        result = run_game(
            first_exec, second_exec,
            time_ms=args.time,
            use_color=use_color,
            verbose=not args.quiet,
            delay=args.delay,
        )

        # swap된 경우 결과 보정 (항상 agent1 기준으로 기록)
        if swapped:
            orig_result = GameResult()
            orig_result.first_score = result.second_score
            orig_result.second_score = result.first_score
            orig_result.total_turns = result.total_turns
            orig_result.first_time_used = result.second_time_used
            orig_result.second_time_used = result.first_time_used
            orig_result.moves = result.moves
            orig_result.error_player = result.error_player
            orig_result.error_msg = result.error_msg
            if result.winner == Piece.FIRST:
                orig_result.winner = Piece.SECOND
            elif result.winner == Piece.SECOND:
                orig_result.winner = Piece.FIRST
            else:
                orig_result.winner = None
            # reason을 agent1/agent2 기준으로 재생성
            fs, ss = orig_result.first_score, orig_result.second_score
            if orig_result.winner == Piece.FIRST:
                orig_result.reason = f"O가 더 많음 ({fs} vs {ss})"
            elif orig_result.winner == Piece.SECOND:
                orig_result.reason = f"X가 더 많음 ({ss} vs {fs})"
            else:
                orig_result.reason = f"무승부 ({fs} vs {ss})"
            if orig_result.error_player:
                orig_result.reason = result.reason + " (swapped)"
            results.append(orig_result)
        else:
            results.append(result)

        if args.quiet:
            r = results[-1]
            if r.winner == Piece.FIRST:
                w = f"O ({agent1_name})"
            elif r.winner == Piece.SECOND:
                w = f"X ({agent2_name})"
            else:
                w = "Draw"
            print(f"{r.first_score}-{r.second_score}  Winner: {w}")
        elif args.games == 1:
            print_result(result, first_name, second_name, use_color)

    # 배치 결과 요약
    if args.games > 1:
        print_batch_summary(results, agent1_name, agent2_name, use_color)


if __name__ == '__main__':
    main()
