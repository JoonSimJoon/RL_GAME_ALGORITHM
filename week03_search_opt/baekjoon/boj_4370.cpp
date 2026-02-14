/*
백준 4370: 곱셈 게임 (Gold IV)
https://www.acmicpc.net/problem/4370

문제:
- 초기값 p (1 이하의 실수)
- 각 턴마다 현재 수에 2~9 중 하나를 곱함
- n 이상이 되면 게임 종료
- n 이상으로 만든 사람이 승리

풀이:
- 게임 DP
- 상태: 현재 수 (p에 몇을 곱했는지로 표현)
- dp[x] = (현재 x일 때 현재 플레이어의 승패)
- x >= n이면 이전 플레이어가 승리 (현재 플레이어는 턴을 못 가짐)

전략:
- 역으로 생각: n에서 시작하여 p까지 거슬러 올라감
- dp[x] = True (승리) if 한 수라도 상대를 패배 상태로 보낼 수 있음

주의:
- p와 n이 실수이므로 직접 계산하지 않고 비율로 처리
- p * k >= n인지 확인 → k >= n/p
*/

#include <bits/stdc++.h>
using namespace std;

map<double, bool> dp;

bool can_win(double current, double n) {
    // 이미 n 이상이면 이전 플레이어가 승리 (현재는 패배)
    if (current >= n) {
        return false;
    }

    // 메모이제이션
    if (dp.count(current)) {
        return dp[current];
    }

    // 2~9를 곱한 후 상대가 패배하는 경우가 있는가?
    for (int mult = 2; mult <= 9; mult++) {
        double next_val = current * mult;
        if (!can_win(next_val, n)) {
            dp[current] = true;
            return true;
        }
    }

    dp[current] = false;
    return false;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    double p, n;
    while (cin >> p >> n) {
        dp.clear();

        if (can_win(p, n)) {
            cout << "Stan wins." << endl;
        } else {
            cout << "Ollie wins." << endl;
        }
    }

    return 0;
}
