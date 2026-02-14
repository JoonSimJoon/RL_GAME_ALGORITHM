"""
백준 16882: 카드 게임 (Gold III)
https://www.acmicpc.net/problem/16882

문제:
- N장의 카드 (각 카드에 숫자)
- 두 명이 번갈아가며 카드를 가져감
- 규칙:
  1. 카드를 1장 또는 2장 가져갈 수 있음
  2. 2장 가져가는 경우, 두 카드의 숫자가 같아야 함
- 가져간 카드의 숫자 합이 점수
- 마지막에 점수가 높은 사람이 승리

풀이:
- Nim 게임의 변형
- 각 카드 그룹(같은 숫자)을 독립적인 게임으로 간주
- Sprague-Grundy 정리 사용

전략:
1. 카드를 숫자별로 그룹화
2. 각 그룹의 그런디 수 계산
   - 1장 그룹: grundy = 1
   - 2장 그룹: grundy = 2 (2장을 한 번에 가져갈 수 있음)
   - 3장 이상: 복잡한 계산 필요
3. 모든 그룹의 그런디 수를 XOR
4. XOR 결과가 0이 아니면 선공 승리

그런디 수 계산 (카드 n장):
- n = 0: grundy = 0
- n = 1: grundy = 1
- n = 2: grundy = 2
- n >= 3: grundy = mex(가능한 후속 상태들의 grundy)

단, 이 문제는 점수 합을 비교하는 게임이므로
실제로는 더 복잡한 분석이 필요합니다.

간단한 전략:
- 선공이 최선을 다할 때 얻을 수 있는 최대 점수를 계산
- 후공이 최선을 다할 때 얻을 수 있는 최대 점수를 계산
- DP로 해결
"""

def solve():
    """카드 게임 풀이"""
    # 입력
    n = int(input())
    cards = list(map(int, input().split()))

    # 카드를 숫자별로 개수 세기
    from collections import Counter
    card_count = Counter(cards)

    # 총 점수 계산
    total_score = sum(cards)

    # 상태: 남은 카드 상황에서 현재 플레이어가 얻을 수 있는 최대 점수
    # 메모이제이션을 위해 튜플로 상태 표현
    memo = {}

    def max_score(state):
        """
        현재 상태에서 현재 플레이어가 얻을 수 있는 최대 점수

        Args:
            state: 튜플 (num1의 개수, num2의 개수, ...)

        Returns:
            현재 플레이어의 최대 점수
        """
        if state in memo:
            return memo[state]

        # 남은 카드가 없으면 0점
        if sum(state) == 0:
            return 0

        best = 0

        # 상태를 리스트로 변환
        state_list = list(state)

        # 카드 종류별로 시도
        for i in range(len(state_list)):
            if state_list[i] > 0:
                # 1장 가져가기
                new_state = state_list[:]
                new_state[i] -= 1
                card_value = sorted(card_count.keys())[i]
                score = card_value + (sum(sorted(card_count.keys())[j] * new_state[j] for j in range(len(new_state))) - max_score(tuple(new_state)))
                best = max(best, score)

                # 2장 가져가기 (같은 숫자만 가능)
                if new_state[i] > 0:
                    new_state2 = new_state[:]
                    new_state2[i] -= 1
                    score2 = 2 * card_value + (sum(sorted(card_count.keys())[j] * new_state2[j] for j in range(len(new_state2))) - max_score(tuple(new_state2)))
                    best = max(best, score2)

        memo[state] = best
        return best

    # 초기 상태
    sorted_nums = sorted(card_count.keys())
    initial_state = tuple(card_count[num] for num in sorted_nums)

    # 선공의 최대 점수
    first_player_score = max_score(initial_state)

    # 후공의 점수
    second_player_score = total_score - first_player_score

    # 승자 판정
    if first_player_score > second_player_score:
        print("koosaga")
    elif first_player_score < second_player_score:
        print("cubelover")
    else:
        print("draw")

if __name__ == "__main__":
    solve()
