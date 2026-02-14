# 백준 28472번: Minimax Tree
# 난이도: Gold V
# 분류: 트리 / Minimax / 재귀
#
# 풀이 핵심:
# - 완전 이진 트리가 주어지고, 리프 노드의 값이 주어짐
# - 루트는 MAX 노드, 각 레벨마다 MIN/MAX가 교대로 나타남
# - Minimax 알고리즘을 사용하여 루트 노드의 값을 계산
#
# Minimax 규칙:
# - MAX 노드: 자식 노드들 중 최댓값 선택
# - MIN 노드: 자식 노드들 중 최솟값 선택
#
# 구조:
# - 레벨 0 (루트): MAX
# - 레벨 1: MIN
# - 레벨 2: MAX
# - 레벨 3: MIN
# - ...

def solve():
    # 입력: 트리의 높이 H
    H = int(input())

    # 리프 노드는 레벨 H에 있음
    # 리프 노드 개수 = 2^H
    leaf_count = 2 ** H

    # 리프 노드 값 입력
    leaves = list(map(int, input().split()))

    # Minimax 트리 계산
    # 현재 레벨의 노드 값들
    current_level = leaves

    # 레벨 H부터 0까지 역순으로 계산
    for level in range(H - 1, -1, -1):
        next_level = []

        # 레벨 0이 MAX이므로, 홀수 레벨은 MIN, 짝수 레벨은 MAX
        is_max_level = (level % 2 == 0)

        # 두 자식씩 묶어서 부모 노드 값 계산
        for i in range(0, len(current_level), 2):
            left_child = current_level[i]
            right_child = current_level[i + 1]

            if is_max_level:
                # MAX 노드: 최댓값 선택
                parent_value = max(left_child, right_child)
            else:
                # MIN 노드: 최솟값 선택
                parent_value = min(left_child, right_child)

            next_level.append(parent_value)

        current_level = next_level

    # 루트 노드의 값 출력
    print(current_level[0])

solve()
