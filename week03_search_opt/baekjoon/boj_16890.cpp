/*
백준 16890: 창업 (Gold I)
https://www.acmicpc.net/problem/16890

문제:
- 두 문자열 A, B가 주어짐 (길이 같음)
- 구데기(A)와 큐브러버(B)가 번갈아가며 자신의 문자열에서 문자 선택
- 선택한 문자들을 순서대로 이어붙여 최종 문자열 생성
- 구데기는 사전순으로 작은 문자열을 원함
- 큐브러버는 사전순으로 큰 문자열을 원함
- 최선을 다할 때 최종 문자열은?

풀이:
- 그리디 + 게임 이론
- 양쪽이 최선을 다하므로 다음 전략:
  1. 구데기: 가능한 한 작은 문자를 앞쪽에 배치
  2. 큐브러버: 가능한 한 큰 문자를 앞쪽에 배치

전략:
- 각자의 문자를 정렬
- 구데기: 오름차순 정렬 후 앞/뒤에서 선택
- 큐브러버: 내림차순 정렬 후 앞/뒤에서 선택
- 현재 턴에서 최선의 선택:
  - 구데기 턴: A의 가장 작은 문자 vs 큐브러버가 다음에 놓을 가장 큰 문자 비교
  - 큐브러버 턴: B의 가장 큰 문자 vs 구데기가 다음에 놓을 가장 작은 문자 비교

구현:
- 덱(deque) 사용하여 양쪽 끝에서 선택
- 위치 추적: 앞에서부터 채울지 뒤에서부터 채울지 결정
*/

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string A, B;
    cin >> A >> B;

    int n = A.length();

    // 문자 정렬
    sort(A.begin(), A.end());  // 구데기: 오름차순
    sort(B.begin(), B.end(), greater<char>());  // 큐브러버: 내림차순

    deque<char> A_sorted(A.begin(), A.end());
    deque<char> B_sorted(B.begin(), B.end());

    // 결과 문자열
    vector<char> result(n);
    int left = 0;
    int right = n - 1;

    // 턴 (true: 구데기, false: 큐브러버)
    bool turn = true;  // 구데기 선공

    for (int i = 0; i < n; i++) {
        if (turn) {
            // 구데기 턴
            if (left <= right) {
                if (!B_sorted.empty() && A_sorted.front() < B_sorted.front()) {
                    // 앞에 배치 (유리)
                    result[left] = A_sorted.front();
                    A_sorted.pop_front();
                    left++;
                } else {
                    // 뒤에 배치 (방어)
                    result[right] = A_sorted.back();
                    A_sorted.pop_back();
                    right--;
                }
            }
        } else {
            // 큐브러버 턴
            if (left <= right) {
                if (!A_sorted.empty() && B_sorted.front() > A_sorted.front()) {
                    // 앞에 배치 (유리)
                    result[left] = B_sorted.front();
                    B_sorted.pop_front();
                    left++;
                } else {
                    // 뒤에 배치 (방어)
                    result[right] = B_sorted.back();
                    B_sorted.pop_back();
                    right--;
                }
            }
        }

        // 턴 교체
        turn = !turn;
    }

    // 결과 출력
    for (char c : result) {
        cout << c;
    }
    cout << endl;

    return 0;
}
