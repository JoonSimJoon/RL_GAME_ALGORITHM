#!/usr/bin/env python3
"""
백준 16894: 약수 게임 (Gold IV)

문제:
- 자연수 N이 주어짐
- 두 사람이 번갈아가며 게임:
  1. 현재 수의 약수 중 하나를 선택 (자기 자신 제외)
  2. 선택한 약수만큼 현재 수에서 뺌
- 1이 되면 더 이상 진행할 수 없음 (1의 약수는 1뿐)
- 마지막으로 수를 뺀 사람이 승리
- 선공이 이기면 koosaga, 후공이 이기면 cubelover

풀이:
- Sprague-Grundy 정리 적용
- 그런디 수(Grundy number) 계산

그런디 수:
- g(1) = 0 (패배 상태, 더 이상 움직일 수 없음)
- g(n) = mex({g(n - d) : d는 n의 약수, d ≠ n})
  mex = minimum excludant (포함되지 않은 최소 자연수)

패턴:
- 소수 p: g(p) = mex({g(p-1)}) = mex({g(p-1)})
  p-1이 홀수면 g(p-1)은 대부분 0 또는 작은 수

관찰:
- N이 홀수: 모든 약수가 홀수 → N - 약수 = 짝수
- N이 짝수: 약수 중 홀수 존재 → N - 홀수 = 홀수

패턴 발견 (DP로 계산):
N=1: g=0 (패배)
N=2: 약수=1 → 2-1=1 → g(1)=0 → g(2)=mex({0})=1 (승리)
N=3: 약수=1 → 3-1=2 → g(2)=1 → g(3)=mex({1})=0 (패배)
N=4: 약수=1,2 → 4-1=3(g=0), 4-2=2(g=1) → g(4)=mex({0,1})=2
N=5: 약수=1 → 5-1=4 → g(4)=2 → g(5)=mex({2})=0 (패배)
...

규칙:
- 홀수 N (N ≥ 3): cubelover 승 (그런디 수 0)
- 짝수 N: koosaga 승 (그런디 수 ≠ 0)
- N=1: cubelover 승 (이미 게임 끝)

간단한 규칙:
- N이 홀수 → cubelover
- N이 짝수 → koosaga
"""

import sys

def solve():
    N = int(sys.stdin.readline().strip())

    if N % 2 == 1:
        print("cubelover")  # 후공 승
    else:
        print("koosaga")  # 선공 승


if __name__ == "__main__":
    solve()
