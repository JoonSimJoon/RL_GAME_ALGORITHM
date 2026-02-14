"""
백준 4370: 곱셈 게임 (Gold IV)
https://www.acmicpc.net/problem/4370

문제:
- 초기값 p (1 이하의 실수)
- 각 턴마다 현재 수에 2~9 중 하나를 곱함
- n 이상이 되면 게임 종료
- n 이상으로 만든 사람이 승리

풀이:
- 게임 DP
- 상태: 현재 수 (p에 몇을 곱했는지로 표현)
- dp[x] = (현재 x일 때 현재 플레이어의 승패)
- x >= n이면 이전 플레이어가 승리 (현재 플레이어는 턴을 못 가짐)

전략:
- 역으로 생각: n에서 시작하여 p까지 거슬러 올라감
- dp[x] = True (승리) if 한 수라도 상대를 패배 상태로 보낼 수 있음

주의:
- p와 n이 실수이므로 직접 계산하지 않고 비율로 처리
- p * k >= n인지 확인 → k >= n/p
"""

def solve(p, n):
    """
    곱셈 게임 승패 판정

    Args:
        p: 초기값
        n: 목표값

    Returns:
        선공이 승리하면 True, 아니면 False
    """
    # n/p까지의 승패 계산
    # 상태를 정수로 표현: 1 * multiplier가 현재 값
    # p * multiplier >= n인지 확인

    # 최대 상태: n/p를 넘지 않는 범위
    ratio = n / p

    # DP 테이블
    # dp[k] = p * k 상태에서 현재 플레이어의 승패
    max_states = int(ratio * 10) + 100  # 충분히 큰 크기

    dp = {}

    def can_win(current):
        """
        현재 값이 current일 때 현재 플레이어가 승리하는가?

        Args:
            current: 현재 값

        Returns:
            승리 여부
        """
        # 이미 n 이상이면 이전 플레이어가 승리 (현재는 패배)
        if current >= n:
            return False

        # 메모이제이션
        if current in dp:
            return dp[current]

        # 2~9를 곱한 후 상대가 패배하는 경우가 있는가?
        for mult in range(2, 10):
            next_val = current * mult
            if not can_win(next_val):
                dp[current] = True
                return True

        dp[current] = False
        return False

    return can_win(p)

def main():
    """메인 함수"""
    while True:
        try:
            line = input().strip()
            if not line:
                break

            p, n = map(float, line.split())

            if solve(p, n):
                print("Stan wins.")
            else:
                print("Ollie wins.")

        except EOFError:
            break

if __name__ == "__main__":
    main()
