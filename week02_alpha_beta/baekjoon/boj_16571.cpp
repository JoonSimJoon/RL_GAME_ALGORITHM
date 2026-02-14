/*
백준 16571: 알파 틱택토 (Gold III)

문제:
- 3×3 틱택토 게임
- 보드 상태가 주어짐 (0=빈칸, 1=선공돌, 2=후공돌)
- 선공이 이기면 W, 후공이 이기면 L, 무승부면 D 출력

풀이:
- Minimax 알고리즘으로 최적 플레이 시뮬레이션
- 메모이제이션으로 중복 계산 방지

상태 인코딩:
- 3×3 보드를 3진수로 인코딩 (0, 1, 2)
- state = Σ board[i][j] * 3^(i*3 + j)

Minimax:
- 현재 플레이어가 이길 수 있으면 1
- 무승부면 0
- 질 수밖에 없으면 -1

반환값:
- 1: 현재 플레이어 승리
- 0: 무승부
- -1: 현재 플레이어 패배
*/

#include <bits/stdc++.h>
using namespace std;

map<int, int> memo;

int encode_state(vector<vector<int>>& board) {
    int state = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            state = state * 3 + board[i][j];
        }
    }
    return state;
}

int check_winner(vector<vector<int>>& board) {
    // 가로
    for (int i = 0; i < 3; i++) {
        if (board[i][0] == board[i][1] && board[i][1] == board[i][2] && board[i][0] != 0) {
            return board[i][0];
        }
    }

    // 세로
    for (int j = 0; j < 3; j++) {
        if (board[0][j] == board[1][j] && board[1][j] == board[2][j] && board[0][j] != 0) {
            return board[0][j];
        }
    }

    // 대각선
    if (board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] != 0) {
        return board[0][0];
    }
    if (board[0][2] == board[1][1] && board[1][1] == board[2][0] && board[0][2] != 0) {
        return board[0][2];
    }

    // 빈 칸이 있으면 진행중
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == 0) {
                return 0;
            }
        }
    }

    // 무승부
    return 3;
}

int minimax(vector<vector<int>>& board, int player) {
    int state = encode_state(board);
    if (memo.find(state) != memo.end()) {
        return memo[state];
    }

    int winner = check_winner(board);
    if (winner != 0) {
        if (winner == 3) {  // 무승부
            return 0;
        } else if (winner == player) {  // 현재 플레이어 승리
            return 1;
        } else {  // 현재 플레이어 패배
            return -1;
        }
    }

    // 가능한 수 탐색
    int opponent = 3 - player;  // 1 <-> 2
    int best_score = -2;  // -1보다 작은 값

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == 0) {
                // 수를 둠
                board[i][j] = player;

                // 상대방 입장에서 minimax
                int score = -minimax(board, opponent);

                // 원복
                board[i][j] = 0;

                best_score = max(best_score, score);

                // Pruning: 이미 승리를 찾았으면 더 볼 필요 없음
                if (best_score == 1) {
                    break;
                }
            }
        }
        if (best_score == 1) {
            break;
        }
    }

    memo[state] = best_score;
    return best_score;
}

int main() {
    vector<vector<int>> board(3, vector<int>(3));
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            cin >> board[i][j];
        }
    }

    // 현재 누구 차례인지 판단
    int count1 = 0, count2 = 0;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            if (board[i][j] == 1) count1++;
            if (board[i][j] == 2) count2++;
        }
    }

    // 선공(1)이 먼저 두므로 count1 == count2이면 선공 차례
    int current_player = (count1 == count2) ? 1 : 2;

    int result = minimax(board, current_player);

    if (result == 1) {
        cout << "W" << endl;  // 선공 승 (현재 플레이어가 선공)
    } else if (result == -1) {
        cout << "L" << endl;  // 후공 승
    } else {
        cout << "D" << endl;  // 무승부
    }

    return 0;
}
