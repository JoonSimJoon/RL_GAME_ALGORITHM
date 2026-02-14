#!/usr/bin/env python3
"""
백준 9660: 돌 게임 6 (Silver V)

문제:
- N개의 돌이 있고, 상근이와 창영이가 번갈아가며 게임
- 한 번에 1개, 3개, 또는 4개의 돌을 가져갈 수 있음
- 마지막 돌을 가져가는 사람이 승리
- 상근이가 선공, 둘 다 완벽하게 플레이할 때 승자는?

풀이:
- N이 매우 클 수 있음 (10^18 이하)
- DP로는 메모리 초과 → 패턴 찾기

패턴 분석 (작은 N부터):
N=1: 상근 1개 → SK
N=2: 상근 1개 → 창영 1개 → CY
N=3: 상근 3개 → SK
N=4: 상근 4개 → SK
N=5: 상근 1개 or 3개 or 4개 → 모두 창영이 이김 → CY
N=6: 상근 1개 → N=5(CY) → SK
N=7: 상근 3개 → N=4(SK) 또는 4개 → N=3(SK) → CY
...

DP로 계산:
win[0] = False (0개 남으면 이전 사람이 이김)
win[i] = not win[i-1] or not win[i-3] or not win[i-4]
       = 적어도 하나의 수로 상대를 지게 만들 수 있으면 승리

결과:
1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 ...
SK CY SK SK CY SK CY SK CY SK CY SK SK CY SK CY ...

주기: 7
패턴: SK SK SK SK CY SK CY (인덱스 0~6)
      1  2  3  4  5  6  0

규칙:
- N % 7 == 0 or N % 7 == 2: CY
- 나머지: SK
"""

import sys

def solve():
    N = int(sys.stdin.readline().strip())

    remainder = N % 7

    if remainder == 0 or remainder == 2:
        print("CY")
    else:
        print("SK")


if __name__ == "__main__":
    solve()
