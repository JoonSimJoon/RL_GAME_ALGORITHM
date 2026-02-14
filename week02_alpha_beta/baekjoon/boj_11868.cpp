/*
백준 11868: 님 게임 2 (Silver II)

문제:
- N개의 돌 더미가 있음
- 두 사람이 번갈아가며 한 더미를 선택해 원하는 만큼 돌을 제거
- 마지막 돌을 가져가는 사람이 승리
- 선공이 이기면 koosaga, 후공이 이기면 cubelover 출력

풀이:
- 전형적인 님 게임 (Nim Game)
- 모든 더미의 XOR 연산 결과가 0이 아니면 선공 승리
- XOR = 0이면 후공 승리

증명:
- XOR = 0인 상태는 패배 상태 (둘 수 없는 승리 전략)
- XOR ≠ 0인 상태는 승리 상태 (XOR = 0으로 만드는 수가 존재)

Sprague-Grundy 정리:
- 각 더미의 돌 개수를 그런디 수(Grundy number)로 봄
- 전체 게임의 그런디 수 = 모든 더미의 XOR
- 그런디 수가 0이면 패배 상태 (P-position)
- 그런디 수가 0이 아니면 승리 상태 (N-position)
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    cin >> N;

    int xor_sum = 0;
    for (int i = 0; i < N; i++) {
        int pile;
        cin >> pile;
        xor_sum ^= pile;
    }

    if (xor_sum != 0) {
        cout << "koosaga" << endl;  // 선공 승
    } else {
        cout << "cubelover" << endl;  // 후공 승
    }

    return 0;
}
