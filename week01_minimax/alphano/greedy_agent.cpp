#include <bits/stdc++.h>
using namespace std;

// ---- Board state
int board[8][8];
int turn;

// ---- Direction arrays
const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

// ---- Undo information
struct UndoInfo {
    int x1, y1, x2, y2, dist;
    int infected[8][2];
    int infected_cnt;
};
UndoInfo undo_stack[128];
int undo_top = 0;

// ---- Helper functions
int dist(int x1, int y1, int x2, int y2) {
    return max(abs(x1 - x2), abs(y1 - y2));
}

bool in_bounds(int x, int y) {
    return x >= 1 && x <= 7 && y >= 1 && y <= 7;
}

int count_pieces(int player) {
    int cnt = 0;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] == player) cnt++;
        }
    }
    return cnt;
}

// ---- Make/Undo move
void make_move(int x1, int y1, int x2, int y2, int player) {
    UndoInfo& undo = undo_stack[undo_top++];
    undo.x1 = x1;
    undo.y1 = y1;
    undo.x2 = x2;
    undo.y2 = y2;
    undo.dist = dist(x1, y1, x2, y2);
    undo.infected_cnt = 0;

    // Apply move
    if (undo.dist == 2) {
        board[x1][y1] = 0;
    }
    board[x2][y2] = player;

    // Infect neighbors
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (in_bounds(nx, ny) && board[nx][ny] == (player ^ 3)) {
            undo.infected[undo.infected_cnt][0] = nx;
            undo.infected[undo.infected_cnt][1] = ny;
            undo.infected_cnt++;
            board[nx][ny] = player;
        }
    }
}

void undo_move(int player) {
    assert(undo_top > 0);
    UndoInfo& undo = undo_stack[--undo_top];

    // Restore infected pieces
    for (int i = 0; i < undo.infected_cnt; i++) {
        int nx = undo.infected[i][0];
        int ny = undo.infected[i][1];
        board[nx][ny] = player ^ 3;
    }

    // Restore board
    board[undo.x2][undo.y2] = 0;
    if (undo.dist == 2) {
        board[undo.x1][undo.y1] = player;
    }
}

// ---- Move generation and selection
tuple<int, int, int, int> find_move() {
    int best_score = -1e9;
    int best_x1 = -1, best_y1 = -1, best_x2 = -1, best_y2 = -1;

    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (board[x1][y1] != turn) continue;

            for (int x2 = 1; x2 <= 7; x2++) {
                for (int y2 = 1; y2 <= 7; y2++) {
                    if (board[x2][y2] != 0) continue;
                    int d = dist(x1, y1, x2, y2);
                    if (d == 0 || d > 2) continue;

                    // Make move and evaluate
                    make_move(x1, y1, x2, y2, turn);
                    int score = count_pieces(turn) - count_pieces(turn ^ 3);
                    undo_move(turn);

                    // Update best
                    if (score > best_score) {
                        best_score = score;
                        best_x1 = x1;
                        best_y1 = y1;
                        best_x2 = x2;
                        best_y2 = y2;
                    }
                }
            }
        }
    }

    return make_tuple(best_x1, best_y1, best_x2, best_y2);
}

// ---- Apply moves
void apply_my_move(int x1, int y1, int x2, int y2) {
    int d = dist(x1, y1, x2, y2);
    assert(d == 1 || d == 2);
    assert(board[x1][y1] == turn);
    assert(board[x2][y2] == 0);

    if (d == 2) {
        board[x1][y1] = 0;
    }
    board[x2][y2] = turn;

    for (int i = 0; i < 8; i++) {
        int nx = x2 + dx[i];
        int ny = y2 + dy[i];
        if (in_bounds(nx, ny) && board[nx][ny] == (turn ^ 3)) {
            board[nx][ny] = turn;
        }
    }
}

void apply_opp_move(int x1, int y1, int x2, int y2) {
    int opp = turn ^ 3;
    int d = dist(x1, y1, x2, y2);
    assert(d == 1 || d == 2);
    assert(board[x1][y1] == opp);
    assert(board[x2][y2] == 0);

    if (d == 2) {
        board[x1][y1] = 0;
    }
    board[x2][y2] = opp;

    for (int i = 0; i < 8; i++) {
        int nx = x2 + dx[i];
        int ny = y2 + dy[i];
        if (in_bounds(nx, ny) && board[nx][ny] == turn) {
            board[nx][ny] = opp;
        }
    }
}

// ---- Main
int main() {
    string line;

    while (getline(cin, line)) {
        istringstream iss(line);
        string cmd;
        iss >> cmd;

        if (cmd == "FIRST") {
            turn = 1;
            memset(board, 0, sizeof(board));
            board[1][1] = 2;
            board[1][7] = 1;
            board[7][1] = 1;
            board[7][7] = 2;
        } else if (cmd == "SECOND") {
            turn = 2;
            memset(board, 0, sizeof(board));
            board[1][1] = 2;
            board[1][7] = 1;
            board[7][1] = 1;
            board[7][7] = 2;
        } else if (cmd == "TURN") {
            int t1;
            iss >> t1;

            auto [x1, y1, x2, y2] = find_move();
            cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
            apply_my_move(x1, y1, x2, y2);
        } else if (cmd == "OPP") {
            int x1, y1, x2, y2, t2;
            iss >> x1 >> y1 >> x2 >> y2 >> t2;
            apply_opp_move(x1, y1, x2, y2);
        }
    }

    return 0;
}
