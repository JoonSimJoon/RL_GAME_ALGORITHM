/*
백준 16882: 카드 게임 (Gold III)
https://www.acmicpc.net/problem/16882

문제:
- N장의 카드 (각 카드에 숫자)
- 두 명이 번갈아가며 카드를 가져감
- 규칙:
  1. 카드를 1장 또는 2장 가져갈 수 있음
  2. 2장 가져가는 경우, 두 카드의 숫자가 같아야 함
- 가져간 카드의 숫자 합이 점수
- 마지막에 점수가 높은 사람이 승리

풀이:
- Nim 게임의 변형
- 각 카드 그룹(같은 숫자)을 독립적인 게임으로 간주
- Sprague-Grundy 정리 사용

전략:
1. 카드를 숫자별로 그룹화
2. 각 그룹의 그런디 수 계산
   - 1장 그룹: grundy = 1
   - 2장 그룹: grundy = 2 (2장을 한 번에 가져갈 수 있음)
   - 3장 이상: 복잡한 계산 필요
3. 모든 그룹의 그런디 수를 XOR
4. XOR 결과가 0이 아니면 선공 승리

그런디 수 계산 (카드 n장):
- n = 0: grundy = 0
- n = 1: grundy = 1
- n = 2: grundy = 2
- n >= 3: grundy = mex(가능한 후속 상태들의 grundy)

단, 이 문제는 점수 합을 비교하는 게임이므로
실제로는 더 복잡한 분석이 필요합니다.

간단한 전략:
- 선공이 최선을 다할 때 얻을 수 있는 최대 점수를 계산
- 후공이 최선을 다할 때 얻을 수 있는 최대 점수를 계산
- DP로 해결
*/

#include <bits/stdc++.h>
using namespace std;

map<vector<int>, int> memo;
vector<int> sorted_nums;

int max_score(vector<int> state) {
    if (memo.count(state)) {
        return memo[state];
    }

    int sum = 0;
    for (int x : state) sum += x;

    if (sum == 0) {
        return 0;
    }

    int best = 0;

    for (size_t i = 0; i < state.size(); i++) {
        if (state[i] > 0) {
            // 1장 가져가기
            vector<int> new_state = state;
            new_state[i]--;

            int remaining_sum = 0;
            for (size_t j = 0; j < new_state.size(); j++) {
                remaining_sum += sorted_nums[j] * new_state[j];
            }

            int score = sorted_nums[i] + (remaining_sum - max_score(new_state));
            best = max(best, score);

            // 2장 가져가기
            if (new_state[i] > 0) {
                vector<int> new_state2 = new_state;
                new_state2[i]--;

                int remaining_sum2 = 0;
                for (size_t j = 0; j < new_state2.size(); j++) {
                    remaining_sum2 += sorted_nums[j] * new_state2[j];
                }

                int score2 = 2 * sorted_nums[i] + (remaining_sum2 - max_score(new_state2));
                best = max(best, score2);
            }
        }
    }

    memo[state] = best;
    return best;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<int> cards(n);
    map<int, int> card_count;
    int total_score = 0;

    for (int i = 0; i < n; i++) {
        cin >> cards[i];
        card_count[cards[i]]++;
        total_score += cards[i];
    }

    // 정렬된 숫자 목록
    for (auto& p : card_count) {
        sorted_nums.push_back(p.first);
    }

    // 초기 상태
    vector<int> initial_state;
    for (int num : sorted_nums) {
        initial_state.push_back(card_count[num]);
    }

    // 선공의 최대 점수
    int first_player_score = max_score(initial_state);

    // 후공의 점수
    int second_player_score = total_score - first_player_score;

    // 승자 판정
    if (first_player_score > second_player_score) {
        cout << "koosaga" << endl;
    } else if (first_player_score < second_player_score) {
        cout << "cubelover" << endl;
    } else {
        cout << "draw" << endl;
    }

    return 0;
}
