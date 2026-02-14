// 백준 14429번: 배스킨라빈스 31
// 난이도: Bronze I
// 분류: 게임 이론 / 수학
//
// 풀이 핵심:
// - 1부터 시작하여 한 턴에 1~3개의 숫자를 말함
// - 31을 말하는 사람이 패배
// - 첫 번째 턴을 시작하는 사람 번호와 총 플레이어 수가 주어짐
//
// 전략:
// - 거꾸로 생각: 31을 말하면 패배
// - 30, 29, 28, 27 중 하나를 말한 사람은 상대방이 31을 말하게 만들 수 있음 → 승리
// - 26을 말한 사람은 무엇을 해도 상대방이 27~30 중 하나를 선택 → 패배
// - 패턴: 31, 27, 23, 19, 15, 11, 7, 3이 패배 위치 (4의 배수 - 1)
//
// 최적 플레이:
// - 31 = 4*7 + 3, 즉 3부터 시작하는 4의 배수 + 3 패턴
// - 3, 7, 11, 15, 19, 23, 27, 31이 패배 포지션
// - 현재 턴에서 몇 번째 숫자를 말하는지 계산하여 누가 패배 포지션에 도달하는지 판단

#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    int A, B;
    cin >> A >> B;

    // 현재 말해야 할 숫자
    int current_number = 1;
    // 현재 턴 플레이어 (0-indexed)
    int current_player = B - 1;

    while (current_number <= 31) {
        // 최적 플레이: 다음 패배 포지션까지 도달하도록 조정
        // 패배 포지션: (4k - 1) 형태

        // 다음 패배 포지션 찾기
        int next_lose = ((current_number / 4) + 1) * 4 - 1;

        if (next_lose > 31) {
            next_lose = 31;
        }

        // 현재 플레이어가 말할 개수
        int count = next_lose - current_number + 1;

        // 1~3 범위로 제한
        if (count > 3) {
            count = 3;
        } else if (count < 1) {
            count = 1;
        }

        // 숫자 업데이트
        current_number += count;

        // 31에 도달하면 현재 플레이어 패배
        if (current_number >= 31) {
            cout << current_player + 1 << '\n';
            return 0;
        }

        // 다음 플레이어로 넘어감
        current_player = (current_player + 1) % A;
    }

    return 0;
}
