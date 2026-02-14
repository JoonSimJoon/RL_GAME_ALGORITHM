/*
백준 2600: 구슬게임 (Gold IV)
https://www.acmicpc.net/problem/2600

문제:
- 두 병에 각각 k1, k2개의 구슬
- 각 턴마다 다음 중 하나 선택:
  1. 첫 번째 병에서 b1개, 두 번째 병에서 b2개 제거
  2. 첫 번째 병에서 b3개, 두 번째 병에서 b4개 제거
- 구슬을 가져갈 수 없는 사람이 패배
- 5개의 게임에 대해 승패 판정

풀이:
- 2차원 게임 DP
- dp[i][j] = (첫 번째 병에 i개, 두 번째 병에 j개 남았을 때 현재 플레이어의 승패)
- dp[i][j] = True (승리) if 한 수라도 상대를 패배 상태로 보낼 수 있음
- dp[i][j] = False (패배) if 모든 수가 상대를 승리 상태로 보냄

점화식:
- dp[0][0] = False (구슬이 없으면 패배)
- dp[i][j] = True if 다음 중 하나라도 True:
  - i >= b1 and j >= b2 and not dp[i-b1][j-b2]
  - i >= b3 and j >= b4 and not dp[i-b3][j-b4]
*/

#include <bits/stdc++.h>
using namespace std;

const int MAX_K = 501;
int dp[MAX_K][MAX_K];
int b1, b2, b3, b4;

bool solve_game(int k1, int k2) {
    // 이미 계산된 경우
    if (dp[k1][k2] != -1) {
        return dp[k1][k2];
    }

    // 기저 사례: 구슬이 없으면 패배
    if (k1 == 0 && k2 == 0) {
        dp[k1][k2] = 0;
        return false;
    }

    // 가능한 수를 시도
    bool can_win = false;

    // 선택지 1: (b1, b2) 제거
    if (k1 >= b1 && k2 >= b2) {
        if (!solve_game(k1 - b1, k2 - b2)) {
            can_win = true;
        }
    }

    // 선택지 2: (b3, b4) 제거
    if (k1 >= b3 && k2 >= b4) {
        if (!solve_game(k1 - b3, k2 - b4)) {
            can_win = true;
        }
    }

    dp[k1][k2] = can_win;
    return can_win;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    cin >> b1 >> b2 >> b3 >> b4;

    // DP 테이블 초기화
    memset(dp, -1, sizeof(dp));

    // 모든 상태에 대해 DP 계산
    for (int i = 0; i < MAX_K; i++) {
        for (int j = 0; j < MAX_K; j++) {
            solve_game(i, j);
        }
    }

    // 5개의 게임 판정
    for (int i = 0; i < 5; i++) {
        int k1, k2;
        cin >> k1 >> k2;

        if (dp[k1][k2]) {
            cout << "A" << endl;  // 선공 승리
        } else {
            cout << "B" << endl;  // 후공 승리
        }
    }

    return 0;
}
