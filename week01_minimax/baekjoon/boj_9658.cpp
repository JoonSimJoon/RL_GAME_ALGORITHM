// 백준 9658번: 돌 게임 4
// 난이도: Silver II
// 분류: 게임 이론 / DP
//
// 풀이 핵심:
// - N개의 돌, 한 턴에 1개, 3개, 또는 4개 가져갈 수 있음
// - 마지막 돌을 가져가는 사람이 패배
// - 상근이가 먼저 시작
//
// DP 접근:
// - dp[i] = true: i개 남았을 때 현재 차례 플레이어가 승리
// - dp[i] = false: i개 남았을 때 현재 차례 플레이어가 패배
//
// 점화식 (마지막 돌 = 패배):
// - dp[i] = true if (dp[i-1] == false or dp[i-3] == false or dp[i-4] == false)
// - 상대방을 패배 상태로 만들 수 있는 선택지가 하나라도 있으면 승리

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int N;
    cin >> N;

    // DP 테이블 초기화
    vector<bool> dp(N + 5, false);

    // 기저 사례 (마지막 돌을 가져가면 패배)
    dp[1] = false;  // 1개 남음 → 1개 가져가서 패배
    dp[2] = true;   // 2개 남음 → 1개 가져감 → 상대가 1개 가져가서 패배 → 승리
    dp[3] = false;  // 3개 남음 → 무엇을 가져가든 상대가 승리
    dp[4] = false;  // 4개 남음 → 무엇을 가져가든 상대가 승리

    // DP 테이블 채우기
    for (int i = 5; i <= N; i++) {
        // 1개, 3개, 또는 4개 가져갔을 때 상대방이 패배 상태가 되면 승리
        if (!dp[i-1] || !dp[i-3] || !dp[i-4]) {
            dp[i] = true;
        } else {
            dp[i] = false;
        }
    }

    // 결과 출력
    if (dp[N]) {
        cout << "SK" << '\n';  // 상근 승리
    } else {
        cout << "CY" << '\n';  // 창영 승리
    }

    return 0;
}
