/*
백준 16895: 님 게임 3 (Gold III)
https://www.acmicpc.net/problem/16895

문제:
- N개의 돌 더미
- 각 턴마다 한 더미에서 원하는 만큼 돌을 제거
- 돌을 가져갈 수 없는 사람이 패배
- 선공이 이기기 위한 첫 수의 가짓수를 구하라

풀이:
- 표준 Nim 게임
- Nim 게임의 승리 조건: 모든 더미의 XOR 값이 0이 아님
- 선공이 이기려면: 현재 XOR 값을 0으로 만드는 수를 찾기

전략:
1. 현재 XOR 값 계산: xor_sum
2. 만약 xor_sum == 0이면 선공 패배 (경우의 수 0)
3. 각 더미에 대해:
   - 해당 더미에서 제거 후 전체 XOR이 0이 되는지 확인
   - pile XOR xor_sum < pile이면 가능
     (xor_sum을 0으로 만들려면 pile을 pile XOR xor_sum으로 만들어야 함)
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<int> piles(n);
    int xor_sum = 0;

    for (int i = 0; i < n; i++) {
        cin >> piles[i];
        xor_sum ^= piles[i];
    }

    // XOR이 0이면 선공 패배
    if (xor_sum == 0) {
        cout << 0 << endl;
        return 0;
    }

    // 선공이 이기는 경우의 수
    int count = 0;

    for (int pile : piles) {
        // 이 더미에서 제거하여 XOR을 0으로 만들 수 있는가?
        int target = pile ^ xor_sum;

        // target < pile이면 가능
        if (target < pile) {
            count++;
        }
    }

    cout << count << endl;

    return 0;
}
