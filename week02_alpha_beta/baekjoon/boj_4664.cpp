/*
백준 4664: Find the Winning Move (Gold I)

문제:
- 4×4 틱택토 게임
- x의 차례이고, x가 이번 수로 강제 승리할 수 있는지 확인
- 가능하면 그 위치 출력, 불가능하면 "#####" 출력
- 여러 테스트 케이스

규칙:
- 가로, 세로, 대각선 4개를 만들면 승리
- x가 선공, o가 후공

입력:
- 각 테스트 케이스는 4줄 (4×4 보드)
- x, o, . (빈칸)
- 빈 줄로 구분
- 입력의 끝은 EOF

출력:
- 강제 승리 가능한 위치 (행, 열) 1-indexed
- 불가능하면 "#####"

풀이:
- Alpha-Beta Pruning 필수 (완전 탐색은 너무 느림)
- 각 빈 칸에 x를 놓고, 그 후 o의 모든 최선의 수에도 x가 이기는지 확인

전략:
1. 모든 빈 칸을 순회
2. 각 빈 칸에 x를 놓음
3. Alpha-Beta로 이후 게임 결과 평가
4. x가 무조건 이기는 수가 있으면 그 위치 반환

Minimax 반환값:
- 1: x 승리
- 0: 무승부
- -1: o 승리

강제 승리 조건:
- x가 수를 놓은 직후 이미 승리했거나
- 이후 모든 o의 수에 대해 x가 승리할 수 있음
*/

#include <bits/stdc++.h>
using namespace std;

char check_winner(vector<vector<char>>& board) {
    // 가로
    for (int i = 0; i < 4; i++) {
        if (board[i][0] == board[i][1] && board[i][1] == board[i][2] &&
            board[i][2] == board[i][3] && board[i][0] != '.') {
            return board[i][0];
        }
    }

    // 세로
    for (int j = 0; j < 4; j++) {
        if (board[0][j] == board[1][j] && board[1][j] == board[2][j] &&
            board[2][j] == board[3][j] && board[0][j] != '.') {
            return board[0][j];
        }
    }

    // 대각선
    if (board[0][0] == board[1][1] && board[1][1] == board[2][2] &&
        board[2][2] == board[3][3] && board[0][0] != '.') {
        return board[0][0];
    }
    if (board[0][3] == board[1][2] && board[1][2] == board[2][1] &&
        board[2][1] == board[3][0] && board[0][3] != '.') {
        return board[0][3];
    }

    return '\0';
}

bool is_full(vector<vector<char>>& board) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (board[i][j] == '.') return false;
        }
    }
    return true;
}

int minimax(vector<vector<char>>& board, char player, int alpha, int beta) {
    char winner = check_winner(board);
    if (winner == 'x') return 1;
    if (winner == 'o') return -1;
    if (is_full(board)) return 0;

    char opponent = (player == 'x') ? 'o' : 'x';

    if (player == 'x') {
        // MAX 플레이어
        int max_score = -2;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (board[i][j] == '.') {
                    board[i][j] = 'x';
                    int score = minimax(board, 'o', alpha, beta);
                    board[i][j] = '.';

                    max_score = max(max_score, score);
                    alpha = max(alpha, score);

                    if (beta <= alpha) {
                        return max_score;  // Beta cutoff
                    }
                }
            }
        }
        return max_score;
    } else {
        // MIN 플레이어
        int min_score = 2;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (board[i][j] == '.') {
                    board[i][j] = 'o';
                    int score = minimax(board, 'x', alpha, beta);
                    board[i][j] = '.';

                    min_score = min(min_score, score);
                    beta = min(beta, score);

                    if (beta <= alpha) {
                        return min_score;  // Alpha cutoff
                    }
                }
            }
        }
        return min_score;
    }
}

pair<int, int> find_winning_move(vector<vector<char>>& board) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (board[i][j] == '.') {
                // x를 놓음
                board[i][j] = 'x';

                // 즉시 승리 확인
                if (check_winner(board) == 'x') {
                    board[i][j] = '.';
                    return {i + 1, j + 1};
                }

                // Alpha-Beta로 평가
                int result = minimax(board, 'o', -2, 2);

                board[i][j] = '.';

                // x가 무조건 이기면
                if (result == 1) {
                    return {i + 1, j + 1};
                }
            }
        }
    }

    return {-1, -1};
}

int main() {
    string line;
    while (getline(cin, line)) {
        vector<vector<char>> board(4, vector<char>(4));

        // 첫 번째 줄은 이미 읽음
        for (int j = 0; j < 4; j++) {
            board[0][j] = line[j];
        }

        // 나머지 3줄 읽기
        for (int i = 1; i < 4; i++) {
            getline(cin, line);
            for (int j = 0; j < 4; j++) {
                board[i][j] = line[j];
            }
        }

        auto result = find_winning_move(board);

        if (result.first != -1) {
            cout << "(" << result.first << "," << result.second << ")" << endl;
        } else {
            cout << "#####" << endl;
        }

        // 빈 줄 소비
        getline(cin, line);
    }

    return 0;
}
