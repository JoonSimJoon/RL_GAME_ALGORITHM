/*
백준 11694: 님 게임 (Silver I)

문제:
- N개의 돌 더미가 있음
- 두 사람이 번갈아가며 한 더미를 선택해 원하는 만큼 돌을 제거
- 마지막 돌을 가져가는 사람이 패배 (Misère Nim)
- 선공이 이기면 koosaga, 후공이 이기면 cubelover 출력

풀이:
- Misère Nim: 일반 Nim과 반대 (마지막 돌 가져가면 패배)

규칙:
1. 모든 더미가 1 이하일 때:
   - 더미 개수가 홀수 → 선공이 마지막 돌 가져감 → 선공 패배
   - 더미 개수가 짝수 → 후공이 마지막 돌 가져감 → 후공 패배

2. 적어도 하나의 더미가 2 이상일 때:
   - XOR ≠ 0 → 선공 승리
   - XOR = 0 → 후공 승리
   (일반 Nim과 동일)

증명:
- 더미가 모두 1 이하가 되기 전까지는 일반 Nim과 동일
- 마지막 단계에서만 승패가 반전
- 승리 전략: 상대방이 마지막 돌을 가져가도록 유도
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    cin >> N;

    vector<int> piles(N);
    int xor_sum = 0;
    bool has_large_pile = false;

    for (int i = 0; i < N; i++) {
        cin >> piles[i];
        xor_sum ^= piles[i];
        if (piles[i] >= 2) {
            has_large_pile = true;
        }
    }

    if (has_large_pile) {
        // 일반 Nim과 동일
        if (xor_sum != 0) {
            cout << "koosaga" << endl;  // 선공 승
        } else {
            cout << "cubelover" << endl;  // 후공 승
        }
    } else {
        // 모든 더미가 1 이하
        // 더미 개수(= 남은 돌 개수)의 홀짝성으로 판단
        int count = 0;
        for (int pile : piles) {
            count += pile;
        }
        if (count % 2 == 1) {
            cout << "cubelover" << endl;  // 선공이 마지막 돌 가져감 → 선공 패배
        } else {
            cout << "koosaga" << endl;  // 후공이 마지막 돌 가져감 → 후공 패배
        }
    }

    return 0;
}
