# 백준 2615번: 오목
# 난이도: Silver I
# 분류: 구현 / 브루트포스
#
# 풀이 핵심:
# - 19x19 바둑판에서 5개가 연속으로 놓인 경우 찾기
# - 6개 이상 연속은 승리가 아님 (육목 규칙)
# - 4방향 체크: 가로, 세로, 대각선(↘), 대각선(↗)
# - 가장 왼쪽 위의 돌 좌표를 출력해야 하므로 방향 고려

def solve():
    # 19x19 바둑판 입력
    board = []
    for _ in range(19):
        board.append(list(map(int, input().split())))

    # 4방향: 가로, 세로, 대각선(↘), 대각선(↗)
    directions = [
        (0, 1),   # 가로 →
        (1, 0),   # 세로 ↓
        (1, 1),   # 대각선 ↘
        (1, -1)   # 대각선 ↗
    ]

    # 모든 위치에서 시작하여 체크
    for i in range(19):
        for j in range(19):
            stone = board[i][j]
            if stone == 0:  # 빈 칸은 건너뛰기
                continue

            # 4방향 체크
            for dx, dy in directions:
                # 이전 칸이 같은 색이면 건너뛰기 (중복 체크 방지)
                prev_x, prev_y = i - dx, j - dy
                if 0 <= prev_x < 19 and 0 <= prev_y < 19:
                    if board[prev_x][prev_y] == stone:
                        continue

                # 현재 방향으로 연속된 돌 개수 세기
                count = 1
                nx, ny = i + dx, j + dy

                while 0 <= nx < 19 and 0 <= ny < 19 and board[nx][ny] == stone:
                    count += 1
                    nx += dx
                    ny += dy

                # 정확히 5개인 경우 승리
                if count == 5:
                    # 승자와 가장 왼쪽 위의 돌 좌표 출력 (1-indexed)
                    print(stone)
                    print(i + 1, j + 1)
                    return

    # 승부가 나지 않은 경우
    print(0)

solve()
