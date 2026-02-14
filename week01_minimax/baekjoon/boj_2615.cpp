// 백준 2615번: 오목
// 난이도: Silver I
// 분류: 구현 / 브루트포스
//
// 풀이 핵심:
// - 19x19 바둑판에서 5개가 연속으로 놓인 경우 찾기
// - 6개 이상 연속은 승리가 아님 (육목 규칙)
// - 4방향 체크: 가로, 세로, 대각선(↘), 대각선(↗)
// - 가장 왼쪽 위의 돌 좌표를 출력해야 하므로 방향 고려

#include <bits/stdc++.h>
using namespace std;

int board[19][19];

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    // 19x19 바둑판 입력
    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            cin >> board[i][j];
        }
    }

    // 4방향: 가로, 세로, 대각선(↘), 대각선(↗)
    int directions[4][2] = {
        {0, 1},   // 가로 →
        {1, 0},   // 세로 ↓
        {1, 1},   // 대각선 ↘
        {1, -1}   // 대각선 ↗
    };

    // 모든 위치에서 시작하여 체크
    for (int i = 0; i < 19; i++) {
        for (int j = 0; j < 19; j++) {
            int stone = board[i][j];
            if (stone == 0) continue;  // 빈 칸은 건너뛰기

            // 4방향 체크
            for (int d = 0; d < 4; d++) {
                int dx = directions[d][0];
                int dy = directions[d][1];

                // 이전 칸이 같은 색이면 건너뛰기 (중복 체크 방지)
                int prev_x = i - dx;
                int prev_y = j - dy;
                if (prev_x >= 0 && prev_x < 19 && prev_y >= 0 && prev_y < 19) {
                    if (board[prev_x][prev_y] == stone) {
                        continue;
                    }
                }

                // 현재 방향으로 연속된 돌 개수 세기
                int count = 1;
                int nx = i + dx;
                int ny = j + dy;

                while (nx >= 0 && nx < 19 && ny >= 0 && ny < 19 && board[nx][ny] == stone) {
                    count++;
                    nx += dx;
                    ny += dy;
                }

                // 정확히 5개인 경우 승리
                if (count == 5) {
                    // 승자와 가장 왼쪽 위의 돌 좌표 출력 (1-indexed)
                    cout << stone << '\n';
                    cout << i + 1 << ' ' << j + 1 << '\n';
                    return 0;
                }
            }
        }
    }

    // 승부가 나지 않은 경우
    cout << 0 << '\n';

    return 0;
}
