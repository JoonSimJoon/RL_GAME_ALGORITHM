/*
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
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n, m;
    while (cin >> n >> m) {
        if (n == 0 && m == 0) break;

        // 격자 입력
        int xor_sum = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                int stones;
                cin >> stones;
                xor_sum ^= stones;
            }
        }

        // XOR 결과가 0이 아니면 선공 승리
        if (xor_sum != 0) {
            cout << "Yes" << endl;  // 선공 승리
        } else {
            cout << "No" << endl;   // 후공 승리
        }
    }

    return 0;
}
