/*
백준 2040: 수 게임 (Gold IV)

문제:
- N개의 수가 일렬로 놓여있음
- 두 사람이 번갈아가며 수를 선택
- 각 턴마다 양쪽 끝 중 하나를 선택
- 각 플레이어는 자신이 선택한 수들의 합을 최대화
- 홍익이가 선공, 두 사람 모두 최적으로 플레이
- 홍익이가 얻을 수 있는 최대 점수 - 상대방 점수

입력:
- N: 수의 개수
- N개의 수

풀이:
- boj_11062와 유사한 구간 DP 문제
- dp[i][j] = 수 i~j가 남았을 때 현재 플레이어가 얻을 수 있는 최대 점수

차이점:
- 출력이 "홍익 점수 - 상대 점수"
- 전체 합을 S라 하면:
  홍익 점수 = H
  상대 점수 = S - H
  H - (S - H) = 2H - S

따라서:
1. dp[0][N-1]로 홍익이의 최대 점수 H를 구함
2. 2 * H - total_sum 출력

DP 점화식:
dp[i][j] = max(
    nums[i] + (sum[i+1][j] - dp[i+1][j]),
    nums[j] + (sum[i][j-1] - dp[i][j-1])
)
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    int N;
    cin >> N;

    vector<int> nums(N);
    for (int i = 0; i < N; i++) {
        cin >> nums[i];
    }

    // 전체 합
    long long total_sum = 0;
    for (int num : nums) {
        total_sum += num;
    }

    // 구간 합 계산
    vector<long long> prefix_sum(N + 1, 0);
    for (int i = 0; i < N; i++) {
        prefix_sum[i + 1] = prefix_sum[i] + nums[i];
    }

    auto get_sum = [&](int i, int j) {
        return prefix_sum[j + 1] - prefix_sum[i];
    };

    // DP 테이블
    vector<vector<long long>> dp(N, vector<long long>(N, 0));

    // 기저: 수 1개 남음
    for (int i = 0; i < N; i++) {
        dp[i][i] = nums[i];
    }

    // 구간 길이를 늘려가며 DP
    for (int length = 2; length <= N; length++) {
        for (int i = 0; i <= N - length; i++) {
            int j = i + length - 1;

            // 왼쪽 선택
            long long left_choice = nums[i];
            if (i + 1 <= j) {
                left_choice += get_sum(i + 1, j) - dp[i + 1][j];
            }

            // 오른쪽 선택
            long long right_choice = nums[j];
            if (i <= j - 1) {
                right_choice += get_sum(i, j - 1) - dp[i][j - 1];
            }

            dp[i][j] = max(left_choice, right_choice);
        }
    }

    // 홍익이의 최대 점수
    long long hongik_score = dp[0][N - 1];

    // 상대방 점수
    long long opponent_score = total_sum - hongik_score;

    // 차이 출력
    cout << hongik_score - opponent_score << '\n';

    return 0;
}
