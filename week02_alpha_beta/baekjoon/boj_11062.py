#!/usr/bin/env python3
"""
백준 11062: 카드 게임 (Gold III)

문제:
- N개의 카드가 일렬로 놓여있고, 각 카드에 점수가 적혀있음
- 두 사람이 번갈아가며 카드를 가져감
- 양쪽 끝의 카드만 가져갈 수 있음
- 근우가 선공, 두 사람 모두 최적으로 플레이
- 근우가 얻을 수 있는 최대 점수는?

입력:
- T: 테스트 케이스 개수
- N: 카드 개수 (짝수)
- 카드 점수들

풀이:
- 구간 DP + Minimax
- dp[i][j][turn] = 카드 i~j가 남았을 때, turn 플레이어가 얻을 수 있는 최대 점수
  turn=0: 근우 차례 (MAX)
  turn=1: 명우 차례 (MIN)

상태 전이:
- 근우 차례 (MAX):
  dp[i][j][0] = max(
      cards[i] + dp[i+1][j][1],  # 왼쪽 카드 선택
      cards[j] + dp[i][j-1][1]   # 오른쪽 카드 선택
  )

- 명우 차례 (MIN):
  명우도 자신의 점수를 최대화하므로:
  dp[i][j][1] = max(
      dp[i+1][j][0],  # 왼쪽 카드 선택 (근우에게는 cards[i] 안 줌)
      dp[i][j-1][0]   # 오른쪽 카드 선택
  )

  하지만 우리는 근우의 점수만 추적하므로, 명우가 선택한 후 근우가 얻을 점수:
  dp[i][j][1] = min(
      dp[i+1][j][0],  # 명우가 왼쪽 선택 → 근우는 나머지에서 최대
      dp[i][j-1][0]   # 명우가 오른쪽 선택
  )

최적화:
- turn을 없애고 하나의 dp로 통합
- dp[i][j] = 카드 i~j가 남았을 때 현재 플레이어가 얻을 수 있는 최대 점수
- 하지만 두 플레이어의 점수를 따로 추적해야 함

더 간단한 방법:
- dp[i][j] = 카드 i~j가 남았을 때 선공이 얻을 수 있는 최대 점수
- total_sum[i][j] = 카드 i~j의 합
- 선공의 점수 + 후공의 점수 = total_sum[i][j]

상태 전이:
dp[i][j] = max(
    cards[i] + (total[i+1][j] - dp[i+1][j]),  # 왼쪽 선택
    cards[j] + (total[i][j-1] - dp[i][j-1])   # 오른쪽 선택
)

설명:
- 왼쪽 카드 선택: cards[i] + (나머지 합 - 상대가 나머지에서 얻을 점수)
- 상대가 나머지에서 얻을 점수 = dp[i+1][j] (상대가 선공으로 플레이)
"""

import sys

def solve():
    T = int(sys.stdin.readline().strip())

    for _ in range(T):
        N = int(sys.stdin.readline().strip())
        cards = list(map(int, sys.stdin.readline().split()))

        # 구간 합 계산
        prefix_sum = [0] * (N + 1)
        for i in range(N):
            prefix_sum[i + 1] = prefix_sum[i] + cards[i]

        def get_sum(i, j):
            """카드 i~j의 합"""
            return prefix_sum[j + 1] - prefix_sum[i]

        # DP 테이블
        dp = [[0] * N for _ in range(N)]

        # 기저: 카드 1개 남음
        for i in range(N):
            dp[i][i] = cards[i]

        # 구간 길이를 늘려가며 DP
        for length in range(2, N + 1):
            for i in range(N - length + 1):
                j = i + length - 1

                # 왼쪽 선택
                left_choice = cards[i]
                if i + 1 <= j:
                    left_choice += get_sum(i + 1, j) - dp[i + 1][j]

                # 오른쪽 선택
                right_choice = cards[j]
                if i <= j - 1:
                    right_choice += get_sum(i, j - 1) - dp[i][j - 1]

                dp[i][j] = max(left_choice, right_choice)

        print(dp[0][N - 1])


if __name__ == "__main__":
    solve()
