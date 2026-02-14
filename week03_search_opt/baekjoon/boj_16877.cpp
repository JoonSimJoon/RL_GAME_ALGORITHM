/*
백준 16877: 핌버 (Gold II)
https://www.acmicpc.net/problem/16877

문제:
- 여러 개의 돌 더미
- 각 턴마다 한 더미에서 피보나치 수만큼 돌을 제거
- 피보나치 수: 1, 2, 3, 5, 8, 13, 21, 34, ...
- 돌을 가져갈 수 없는 사람이 패배

풀이:
- Nim 게임의 변형 (Fibonacci Nim)
- Sprague-Grundy 정리 사용
- 각 돌 더미의 그런디 수(Grundy Number) 계산
- 모든 더미의 그런디 수를 XOR한 값이 0이 아니면 선공 승리

핵심 개념:
1. 그런디 수: 게임 상태의 "가치"를 나타내는 음이 아닌 정수
2. mex (minimum excludant): 집합에 없는 가장 작은 음이 아닌 정수
   - mex({0, 1, 3, 4}) = 2
   - mex({1, 2, 3}) = 0
3. 그런디 수 계산: grundy(n) = mex({grundy(n-f) for f in Fibonacci if f <= n})
4. 여러 게임의 합: 각 게임의 그런디 수를 XOR
   - XOR 결과가 0이면 후공 승리, 아니면 선공 승리
*/

#include <bits/stdc++.h>
using namespace std;

vector<int> calculate_fibonacci(int max_n) {
    vector<int> fib = {1, 2};
    while (fib.back() < max_n) {
        fib.push_back(fib[fib.size() - 1] + fib[fib.size() - 2]);
    }
    return fib;
}

vector<int> calculate_grundy(int max_n) {
    vector<int> fib = calculate_fibonacci(max_n);
    vector<int> grundy(max_n + 1, 0);

    for (int n = 1; n <= max_n; n++) {
        set<int> reachable;
        for (int f : fib) {
            if (f > n) break;
            reachable.insert(grundy[n - f]);
        }

        // mex 계산
        int mex = 0;
        while (reachable.count(mex)) {
            mex++;
        }

        grundy[n] = mex;
    }

    return grundy;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    int n;
    cin >> n;

    vector<int> piles(n);
    int max_pile = 0;
    for (int i = 0; i < n; i++) {
        cin >> piles[i];
        max_pile = max(max_pile, piles[i]);
    }

    // 그런디 수 계산
    vector<int> grundy = calculate_grundy(max_pile);

    // 모든 더미의 그런디 수를 XOR
    int xor_sum = 0;
    for (int pile : piles) {
        xor_sum ^= grundy[pile];
    }

    // XOR 결과가 0이 아니면 선공 승리
    if (xor_sum != 0) {
        cout << "koosaga" << endl;  // 선공
    } else {
        cout << "cubelover" << endl;  // 후공
    }

    return 0;
}
