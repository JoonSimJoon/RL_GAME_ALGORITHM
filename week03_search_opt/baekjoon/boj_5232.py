"""
백준 5232: Grid Nim (Gold I)
https://www.acmicpc.net/problem/5232

문제:
- N x M 격자
- 각 칸에 돌이 있음
- 두 명이 번갈아가며 한 칸을 선택하여 그 칸의 돌을 일부 또는 전부 제거
- 돌을 가져갈 수 없는 사람이 패배

풀이:
- Nim 게임의 2D 확장
- Sprague-Grundy 정리 적용
- 각 칸을 독립적인 게임으로 간주
- 모든 칸의 그런디 수를 XOR

핵심:
- 2D Grid Nim은 일반 Nim과 동일
- 각 칸의 돌 개수 자체가 그런디 수
- 모든 칸의 돌 개수를 XOR한 값이 0이 아니면 선공 승리

증명:
- 각 칸에서 돌을 제거하는 것은 독립적인 Nim 게임
- Nim 게임의 그런디 수는 돌의 개수
- 여러 게임의 합: 각 게임의 그런디 수를 XOR
"""

def solve():
    """Grid Nim 풀이"""
    # 여러 테스트 케이스
    while True:
        try:
            # 입력
            line = input().strip()
            if not line:
                break

            n, m = map(int, line.split())

            # N과 M이 모두 0이면 종료
            if n == 0 and m == 0:
                break

            # 격자 입력
            grid = []
            for _ in range(n):
                row = list(map(int, input().split()))
                grid.append(row)

            # 모든 칸의 XOR 계산
            xor_sum = 0
            for row in grid:
                for stones in row:
                    xor_sum ^= stones

            # XOR 결과가 0이 아니면 선공 승리
            if xor_sum != 0:
                print("Yes")  # 선공 승리
            else:
                print("No")   # 후공 승리

        except EOFError:
            break

if __name__ == "__main__":
    solve()
