"""
백준 16890: 창업 (Gold I)
https://www.acmicpc.net/problem/16890

문제:
- 두 문자열 A, B가 주어짐 (길이 같음)
- 구데기(A)와 큐브러버(B)가 번갈아가며 자신의 문자열에서 문자 선택
- 선택한 문자들을 순서대로 이어붙여 최종 문자열 생성
- 구데기는 사전순으로 작은 문자열을 원함
- 큐브러버는 사전순으로 큰 문자열을 원함
- 최선을 다할 때 최종 문자열은?

풀이:
- 그리디 + 게임 이론
- 양쪽이 최선을 다하므로 다음 전략:
  1. 구데기: 가능한 한 작은 문자를 앞쪽에 배치
  2. 큐브러버: 가능한 한 큰 문자를 앞쪽에 배치

전략:
- 각자의 문자를 정렬
- 구데기: 오름차순 정렬 후 앞/뒤에서 선택
- 큐브러버: 내림차순 정렬 후 앞/뒤에서 선택
- 현재 턴에서 최선의 선택:
  - 구데기 턴: A의 가장 작은 문자 vs 큐브러버가 다음에 놓을 가장 큰 문자 비교
  - 큐브러버 턴: B의 가장 큰 문자 vs 구데기가 다음에 놓을 가장 작은 문자 비교

구현:
- 덱(deque) 사용하여 양쪽 끝에서 선택
- 위치 추적: 앞에서부터 채울지 뒤에서부터 채울지 결정
"""

from collections import deque

def solve():
    """창업 문제 풀이"""
    # 입력
    A = input().strip()
    B = input().strip()

    n = len(A)

    # 문자 정렬
    A_sorted = deque(sorted(A))  # 구데기: 오름차순
    B_sorted = deque(sorted(B, reverse=True))  # 큐브러버: 내림차순

    # 결과 문자열 (앞/뒤에서 채움)
    result = [''] * n
    left = 0
    right = n - 1

    # 턴 (True: 구데기, False: 큐브러버)
    turn = True  # 구데기 선공

    for _ in range(n):
        if turn:
            # 구데기 턴: 작은 문자를 앞에 배치하려 함
            # 하지만 큐브러버가 다음에 큰 문자를 앞에 놓을 수 있으면 방어적으로 뒤에 배치

            if left <= right:
                # 앞에 놓을지 뒤에 놓을지 결정
                # 구데기의 가장 작은 문자 vs 큐브러버의 다음 큰 문자
                if B_sorted and A_sorted[0] < B_sorted[0]:
                    # 앞에 배치 (유리)
                    result[left] = A_sorted.popleft()
                    left += 1
                else:
                    # 뒤에 배치 (방어)
                    result[right] = A_sorted.pop()
                    right -= 1
        else:
            # 큐브러버 턴: 큰 문자를 앞에 배치하려 함
            # 하지만 구데기가 다음에 작은 문자를 앞에 놓을 수 있으면 방어적으로 뒤에 배치

            if left <= right:
                # 앞에 놓을지 뒤에 놓을지 결정
                # 큐브러버의 가장 큰 문자 vs 구데기의 다음 작은 문자
                if A_sorted and B_sorted[0] > A_sorted[0]:
                    # 앞에 배치 (유리)
                    result[left] = B_sorted.popleft()
                    left += 1
                else:
                    # 뒤에 배치 (방어)
                    result[right] = B_sorted.pop()
                    right -= 1

        # 턴 교체
        turn = not turn

    # 결과 출력
    print(''.join(result))

if __name__ == "__main__":
    solve()
