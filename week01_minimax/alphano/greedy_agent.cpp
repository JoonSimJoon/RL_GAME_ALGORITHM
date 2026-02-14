#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Ataxx Greedy Agent
 *  ------------------------------------------------------------
 *  - Board size: 7x7 (1-indexed)
 *  - Piece values: 0 = empty, 1 = FIRST(O), 2 = SECOND(X)
 *  - Strategy: For each legal move, simulate and evaluate
 *    (my pieces - opponent pieces). Pick the move with the
 *    highest immediate gain (1-depth greedy).
 * ============================================================
 */

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

int board[8][8]; // 1-based index board
int turn;        // 1 = FIRST, 2 = SECOND

// ------------------------------------------------------------
// Copy board state to a temporary array for simulation.
// ------------------------------------------------------------
void copy_board(int src[8][8], int dst[8][8]) {
    for (int x = 1; x <= 7; x++) {
        for (int y = 1; y <= 7; y++) {
            dst[x][y] = src[x][y];
        }
    }
}

// ------------------------------------------------------------
// Count number of pieces for a given player on the board.
// ------------------------------------------------------------
int count_pieces(int b[8][8], int player) {
    int cnt = 0;
    for (int x = 1; x <= 7; x++) {
        for (int y = 1; y <= 7; y++) {
            if (b[x][y] == player) cnt++;
        }
    }
    return cnt;
}

// ------------------------------------------------------------
// Simulate a move on a temporary board.
// Applies the move and captures adjacent opponent pieces.
// ------------------------------------------------------------
void simulate_move(int b[8][8], int x1, int y1, int x2, int y2, int player) {
    if (x1 == -1 && y1 == -1 && x2 == -1 && y2 == -1) return;
    int dist = max(abs(x1 - x2), abs(y1 - y2));
    if (dist == 2) b[x1][y1] = 0;
    b[x2][y2] = player;
    int opp = player ^ 3;
    for (int i = 0; i < 8; i++) {
        int nx = x2 + dx[i];
        int ny = y2 + dy[i];
        if (nx < 1 || nx > 7 || ny < 1 || ny > 7) continue;
        if (b[nx][ny] == opp) b[nx][ny] = player;
    }
}

// ------------------------------------------------------------
// Find the greedy move: simulate each legal move and pick
// the one that maximizes (my_pieces - opp_pieces).
// ------------------------------------------------------------
tuple<int, int, int, int> find_move() {
    int best_x1 = -1, best_y1 = -1, best_x2 = -1, best_y2 = -1;
    int best_score = INT_MIN;

    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (board[x1][y1] != turn) continue;
            for (int x2 = x1 - 2; x2 <= x1 + 2; x2++) {
                if (x2 < 1 || x2 > 7) continue;
                for (int y2 = y1 - 2; y2 <= y1 + 2; y2++) {
                    if (y2 < 1 || y2 > 7) continue;
                    if (x2 == x1 && y2 == y1) continue;
                    if (board[x2][y2] != 0) continue;

                    // Simulate the move on a temporary board
                    int tmp[8][8];
                    copy_board(board, tmp);
                    simulate_move(tmp, x1, y1, x2, y2, turn);

                    // Evaluate: (my pieces - opponent pieces)
                    int my_count = count_pieces(tmp, turn);
                    int opp_count = count_pieces(tmp, turn ^ 3);
                    int score = my_count - opp_count;

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

    return {best_x1, best_y1, best_x2, best_y2};
}

// ------------------------------------------------------------
// Apply the player's move to the local board state.
// ------------------------------------------------------------
void apply_my_move(int x1, int y1, int x2, int y2) {
    if (x1 == -1 && y1 == -1 && x2 == -1 && y2 == -1) return;
    int dist = max(abs(x1 - x2), abs(y1 - y2));
    assert(1 <= x1 && x1 <= 7 && 1 <= y1 && y1 <= 7);
    assert(1 <= x2 && x2 <= 7 && 1 <= y2 && y2 <= 7);
    assert(1 <= dist && dist <= 2);
    if (dist == 2) board[x1][y1] = 0;
    board[x2][y2] = turn;
    for (int i = 0; i < 8; i++) {
        int nx = x2 + dx[i];
        int ny = y2 + dy[i];
        if (nx < 1 || nx > 7 || ny < 1 || ny > 7) continue;
        if (board[nx][ny] == (turn ^ 3)) board[nx][ny] = turn;
    }
}

// ------------------------------------------------------------
// Apply the opponent's move to the local board state.
// ------------------------------------------------------------
void apply_opp_move(int x1, int y1, int x2, int y2) {
    if (x1 == -1 && y1 == -1 && x2 == -1 && y2 == -1) return;
    int opp = turn ^ 3;
    int dist = max(abs(x1 - x2), abs(y1 - y2));
    assert(1 <= x1 && x1 <= 7 && 1 <= y1 && y1 <= 7);
    assert(1 <= x2 && x2 <= 7 && 1 <= y2 && y2 <= 7);
    assert(1 <= dist && dist <= 2);
    if (dist == 2) board[x1][y1] = 0;
    board[x2][y2] = opp;
    for (int i = 0; i < 8; i++) {
        int nx = x2 + dx[i];
        int ny = y2 + dy[i];
        if (nx < 1 || nx > 7 || ny < 1 || ny > 7) continue;
        if (board[nx][ny] == turn) board[nx][ny] = opp;
    }
}

// ------------------------------------------------------------
// Main event loop: handles protocol commands and moves.
// ------------------------------------------------------------
int main() {
    board[1][1] = board[7][7] = 1; // FIRST  player (O)
    board[1][7] = board[7][1] = 2; // SECOND player (X)
    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        string cmd;
        in >> cmd;
        if (cmd == "READY") {
            string role;
            in >> role;
            turn = (role == "FIRST" ? 1 : 2);
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2;
            in >> t1 >> t2;
            auto [x1, y1, x2, y2] = find_move();
            apply_my_move(x1, y1, x2, y2);
            cout << "MOVE " << x1 << ' ' << y1 << ' ' << x2 << ' ' << y2 << endl;
        }
        else if (cmd == "OPP") {
            int x1, y1, x2, y2, t;
            in >> x1 >> y1 >> x2 >> y2 >> t;
            apply_opp_move(x1, y1, x2, y2);
        }
        else if (cmd == "FINISH") {
            break;
        }
    }
    return 0;
}
