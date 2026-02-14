#!/usr/bin/env python3
"""
백준 16571: 알파 틱택토 (Gold III)

문제:
- 3×3 틱택토 게임
- 보드 상태가 주어짐 (0=빈칸, 1=선공돌, 2=후공돌)
- 선공이 이기면 W, 후공이 이기면 L, 무승부면 D 출력

풀이:
- Minimax 알고리즘으로 최적 플레이 시뮬레이션
- 메모이제이션으로 중복 계산 방지

상태 인코딩:
- 3×3 보드를 3진수로 인코딩 (0, 1, 2)
- state = Σ board[i][j] * 3^(i*3 + j)

Minimax:
- 현재 플레이어가 이길 수 있으면 1
- 무승부면 0
- 질 수밖에 없으면 -1

반환값:
- 1: 현재 플레이어 승리
- 0: 무승부
- -1: 현재 플레이어 패배
"""

import sys

def solve():
    board = []
    for _ in range(3):
        row = list(map(int, sys.stdin.readline().split()))
        board.append(row)

    # 현재 누구 차례인지 판단
    count1 = sum(row.count(1) for row in board)
    count2 = sum(row.count(2) for row in board)

    # 선공(1)이 먼저 두므로 count1 == count2이면 선공 차례
    current_player = 1 if count1 == count2 else 2

    memo = {}

    def encode_state(board):
        """보드를 정수로 인코딩"""
        state = 0
        for i in range(3):
            for j in range(3):
                state = state * 3 + board[i][j]
        return state

    def check_winner(board):
        """승자 확인 (0=진행중, 1=선공승, 2=후공승, 3=무승부)"""
        # 가로
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != 0:
                return board[i][0]

        # 세로
        for j in range(3):
            if board[0][j] == board[1][j] == board[2][j] != 0:
                return board[0][j]

        # 대각선
        if board[0][0] == board[1][1] == board[2][2] != 0:
            return board[0][0]
        if board[0][2] == board[1][1] == board[2][0] != 0:
            return board[0][2]

        # 빈 칸이 있으면 진행중
        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    return 0

        # 무승부
        return 3

    def minimax(board, player):
        """
        Minimax with memoization
        반환값: 1=현재 플레이어 승, 0=무승부, -1=현재 플레이어 패
        """
        state = encode_state(board)
        if state in memo:
            return memo[state]

        winner = check_winner(board)
        if winner != 0:
            if winner == 3:  # 무승부
                return 0
            elif winner == player:  # 현재 플레이어 승리
                return 1
            else:  # 현재 플레이어 패배
                return -1

        # 가능한 수 탐색
        opponent = 3 - player  # 1 <-> 2
        best_score = -2  # -1보다 작은 값

        for i in range(3):
            for j in range(3):
                if board[i][j] == 0:
                    # 수를 둠
                    board[i][j] = player

                    # 상대방 입장에서 minimax
                    score = -minimax(board, opponent)

                    # 원복
                    board[i][j] = 0

                    best_score = max(best_score, score)

                    # Pruning: 이미 승리를 찾았으면 더 볼 필요 없음
                    if best_score == 1:
                        break
            if best_score == 1:
                break

        memo[state] = best_score
        return best_score

    result = minimax(board, current_player)

    if result == 1:
        print("W")  # 선공 승 (현재 플레이어가 선공)
    elif result == -1:
        print("L")  # 후공 승
    else:
        print("D")  # 무승부


if __name__ == "__main__":
    solve()
