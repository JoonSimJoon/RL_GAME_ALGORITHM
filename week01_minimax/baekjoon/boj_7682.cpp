// 백준 7682번: 틱택토
// 난이도: Silver II
// 분류: 게임 이론 / 구현 / 시뮬레이션
//
// 풀이 핵심:
// - X가 먼저 시작하므로 X의 개수 >= O의 개수이어야 함
// - X가 이긴 경우: X 개수 = O 개수 + 1, O는 이기지 않아야 함
// - O가 이긴 경우: X 개수 = O 개수, X는 이기지 않아야 함
// - 무승부 (보드가 꽉 찬 경우): 9개 모두 채워지고 승자가 없어야 함

#include <bits/stdc++.h>
using namespace std;

bool check_winner(const string& board, char player) {
    // 가로줄 3개
    for (int i = 0; i < 3; i++) {
        if (board[i*3] == player && board[i*3+1] == player && board[i*3+2] == player) {
            return true;
        }
    }

    // 세로줄 3개
    for (int i = 0; i < 3; i++) {
        if (board[i] == player && board[i+3] == player && board[i+6] == player) {
            return true;
        }
    }

    // 대각선 2개
    if (board[0] == player && board[4] == player && board[8] == player) {
        return true;
    }
    if (board[2] == player && board[4] == player && board[6] == player) {
        return true;
    }

    return false;
}

bool is_valid(const string& board) {
    int x_count = 0, o_count = 0;

    for (char c : board) {
        if (c == 'X') x_count++;
        else if (c == 'O') o_count++;
    }

    // 기본 규칙: X가 먼저 시작하므로 X 개수 >= O 개수
    // X와 O의 개수 차이는 0 또는 1이어야 함
    if (x_count < o_count || x_count > o_count + 1) {
        return false;
    }

    bool x_win = check_winner(board, 'X');
    bool o_win = check_winner(board, 'O');

    // 두 명 다 이긴 경우는 불가능
    if (x_win && o_win) {
        return false;
    }

    // X가 이긴 경우
    if (x_win) {
        // X가 이기면 게임이 끝나므로 X 개수 = O 개수 + 1
        return x_count == o_count + 1;
    }

    // O가 이긴 경우
    if (o_win) {
        // O가 이기면 게임이 끝나므로 X 개수 = O 개수
        return x_count == o_count;
    }

    // 승자가 없는 경우
    // 보드가 꽉 차야 함 (9개 모두 채워짐)
    return x_count + o_count == 9;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    string board;
    while (cin >> board) {
        if (board == "end") break;

        if (is_valid(board)) {
            cout << "valid" << '\n';
        } else {
            cout << "invalid" << '\n';
        }
    }

    return 0;
}
