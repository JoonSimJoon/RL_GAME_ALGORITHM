"""
백준 2600: 구슬게임 (Gold IV)
https://www.acmicpc.net/problem/2600

문제:
- 두 병에 각각 k1, k2개의 구슬
- 각 턴마다 다음 중 하나 선택:
  1. 첫 번째 병에서 b1개, 두 번째 병에서 b2개 제거
  2. 첫 번째 병에서 b3개, 두 번째 병에서 b4개 제거
- 구슬을 가져갈 수 없는 사람이 패배
- 5개의 게임에 대해 승패 판정

풀이:
- 2차원 게임 DP
- dp[i][j] = (첫 번째 병에 i개, 두 번째 병에 j개 남았을 때 현재 플레이어의 승패)
- dp[i][j] = True (승리) if 한 수라도 상대를 패배 상태로 보낼 수 있음
- dp[i][j] = False (패배) if 모든 수가 상대를 승리 상태로 보냄

점화식:
- dp[0][0] = False (구슬이 없으면 패배)
- dp[i][j] = True if 다음 중 하나라도 True:
  - i >= b1 and j >= b2 and not dp[i-b1][j-b2]
  - i >= b3 and j >= b4 and not dp[i-b3][j-b4]
"""

def solve_game(b1, b2, b3, b4, k1, k2, dp):
    """
    게임 승패 판정

    Args:
        b1, b2: 첫 번째 선택지 (b1개, b2개 제거)
        b3, b4: 두 번째 선택지 (b3개, b4개 제거)
        k1, k2: 초기 구슬 개수
        dp: 메모이제이션 테이블

    Returns:
        현재 플레이어가 승리하면 True, 아니면 False
    """
    # 이미 계산된 경우
    if dp[k1][k2] != -1:
        return dp[k1][k2]

    # 기저 사례: 구슬이 없으면 패배
    if k1 == 0 and k2 == 0:
        dp[k1][k2] = False
        return False

    # 가능한 수를 시도
    can_win = False

    # 선택지 1: (b1, b2) 제거
    if k1 >= b1 and k2 >= b2:
        if not solve_game(b1, b2, b3, b4, k1 - b1, k2 - b2, dp):
            can_win = True

    # 선택지 2: (b3, b4) 제거
    if k1 >= b3 and k2 >= b4:
        if not solve_game(b1, b2, b3, b4, k1 - b3, k2 - b4, dp):
            can_win = True

    dp[k1][k2] = can_win
    return can_win

def main():
    """메인 함수"""
    # 입력: b1, b2, b3, b4
    b1, b2, b3, b4 = map(int, input().split())

    # 최대 구슬 개수 (문제에서 최대 500)
    MAX_K = 501

    # DP 테이블 초기화 (-1: 미계산)
    dp = [[-1] * MAX_K for _ in range(MAX_K)]

    # 모든 상태에 대해 DP 계산
    for i in range(MAX_K):
        for j in range(MAX_K):
            solve_game(b1, b2, b3, b4, i, j, dp)

    # 5개의 게임 판정
    for _ in range(5):
        k1, k2 = map(int, input().split())
        if dp[k1][k2]:
            print("A")  # 선공 승리
        else:
            print("B")  # 후공 승리

if __name__ == "__main__":
    main()
