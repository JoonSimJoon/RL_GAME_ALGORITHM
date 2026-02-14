"""
백준 16877: 핌버 (Gold II)
https://www.acmicpc.net/problem/16877

문제:
- 여러 개의 돌 더미
- 각 턴마다 한 더미에서 피보나치 수만큼 돌을 제거
- 피보나치 수: 1, 2, 3, 5, 8, 13, 21, 34, ...
- 돌을 가져갈 수 없는 사람이 패배

풀이:
- Nim 게임의 변형 (Fibonacci Nim)
- Sprague-Grundy 정리 사용
- 각 돌 더미의 그런디 수(Grundy Number) 계산
- 모든 더미의 그런디 수를 XOR한 값이 0이 아니면 선공 승리

핵심 개념:
1. 그런디 수: 게임 상태의 "가치"를 나타내는 음이 아닌 정수
2. mex (minimum excludant): 집합에 없는 가장 작은 음이 아닌 정수
   - mex({0, 1, 3, 4}) = 2
   - mex({1, 2, 3}) = 0
3. 그런디 수 계산: grundy(n) = mex({grundy(n-f) for f in Fibonacci if f <= n})
4. 여러 게임의 합: 각 게임의 그런디 수를 XOR
   - XOR 결과가 0이면 후공 승리, 아니면 선공 승리
"""

def calculate_fibonacci(max_n):
    """최대 max_n까지의 피보나치 수 계산"""
    fib = [1, 2]
    while fib[-1] < max_n:
        fib.append(fib[-1] + fib[-2])
    return fib

def calculate_grundy(max_n):
    """
    각 돌 개수에 대한 그런디 수 계산

    grundy[n] = mex({grundy[n-f] for f in Fibonacci if f <= n})
    """
    fib = calculate_fibonacci(max_n)
    grundy = [0] * (max_n + 1)

    for n in range(1, max_n + 1):
        # n개의 돌에서 피보나치 수만큼 제거한 후의 상태들
        reachable = set()
        for f in fib:
            if f > n:
                break
            reachable.add(grundy[n - f])

        # mex 계산: reachable에 없는 가장 작은 음이 아닌 정수
        mex = 0
        while mex in reachable:
            mex += 1

        grundy[n] = mex

    return grundy

def solve():
    """핌버 문제 풀이"""
    # 입력
    n = int(input())  # 돌 더미 개수
    piles = list(map(int, input().split()))  # 각 더미의 돌 개수

    # 최대값 찾기
    max_pile = max(piles)

    # 그런디 수 계산
    grundy = calculate_grundy(max_pile)

    # 모든 더미의 그런디 수를 XOR
    xor_sum = 0
    for pile in piles:
        xor_sum ^= grundy[pile]

    # XOR 결과가 0이 아니면 선공 승리
    if xor_sum != 0:
        print("koosaga")  # 선공
    else:
        print("cubelover")  # 후공

if __name__ == "__main__":
    solve()
