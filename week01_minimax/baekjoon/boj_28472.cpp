// 백준 28472번: Minimax Tree
// 난이도: Gold V
// 분류: 트리 / Minimax / 재귀
//
// 풀이 핵심:
// - 완전 이진 트리가 주어지고, 리프 노드의 값이 주어짐
// - 루트는 MAX 노드, 각 레벨마다 MIN/MAX가 교대로 나타남
// - Minimax 알고리즘을 사용하여 루트 노드의 값을 계산
//
// Minimax 규칙:
// - MAX 노드: 자식 노드들 중 최댓값 선택
// - MIN 노드: 자식 노드들 중 최솟값 선택
//
// 구조:
// - 레벨 0 (루트): MAX
// - 레벨 1: MIN
// - 레벨 2: MAX
// - 레벨 3: MIN
// - ...

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int H;
    cin >> H;

    // 리프 노드는 레벨 H에 있음
    // 리프 노드 개수 = 2^H
    int leaf_count = (1 << H);  // 2^H

    // 리프 노드 값 입력
    vector<int> current_level(leaf_count);
    for (int i = 0; i < leaf_count; i++) {
        cin >> current_level[i];
    }

    // Minimax 트리 계산
    // 레벨 H부터 0까지 역순으로 계산
    for (int level = H - 1; level >= 0; level--) {
        vector<int> next_level;

        // 레벨 0이 MAX이므로, 홀수 레벨은 MIN, 짝수 레벨은 MAX
        bool is_max_level = (level % 2 == 0);

        // 두 자식씩 묶어서 부모 노드 값 계산
        for (int i = 0; i < current_level.size(); i += 2) {
            int left_child = current_level[i];
            int right_child = current_level[i + 1];

            int parent_value;
            if (is_max_level) {
                // MAX 노드: 최댓값 선택
                parent_value = max(left_child, right_child);
            } else {
                // MIN 노드: 최솟값 선택
                parent_value = min(left_child, right_child);
            }

            next_level.push_back(parent_value);
        }

        current_level = next_level;
    }

    // 루트 노드의 값 출력
    cout << current_level[0] << '\n';

    return 0;
}
