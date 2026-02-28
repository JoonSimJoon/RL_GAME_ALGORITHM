#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Catch The Mouse Sample Agent
 *  ------------------------------------------------------------
 *  - Board size: 7x11 (1-indexed)
 *  - Piece values: 0 = empty, 1 = MOUSE(M), 2 = CAT(C), 3 = NADORI(N)
 *  - This agent plays the lexicographically smallest legal move found on each turn.
 * ============================================================
 */

constexpr int inf = 1 << 30;

int board[8][12]; // 1-based index board
int turn;         // 1 = MOUSE, 2 = NADORI

// ------------------------------------------------------------
// Find the first legal move available for the current player.
// Returns (x1, y1, x2, y2)
// ------------------------------------------------------------
tuple<int, int, int, int> find_move(int turn) {
    if (turn == 1) {
        tuple ret(inf, inf, inf, inf);
        for (int i = 1; i <= 6; i++) {
            for (int j = 1; j <= 11; j++) {
                if (board[i][j] != 1) continue;
                if (board[i + 1][j] != 0) continue;
                ret = min(ret, tuple(i, j, i + 1, j));
            }
        }
        assert(ret != tuple(inf, inf, inf, inf));
        return ret;
    }
    else {
        tuple ret(inf, inf, inf, inf);
        for (int i = 2; i <= 6; i++) {
            for (int j = 1; j <= 11; j++) {
                if (board[i][j] == 2) {
                    for (int d = 0; d < 8; d++) {
                        int x = i, y = j;
                        while (1) {
                            x += "10001222"[d] - '1';
                            y += "22100012"[d] - '1';
                            if (x < 2 || x > 6) break;
                            if (y < 1 || y > 11) break;
                            if (board[x][y] != 0) break;
                            ret = min(ret, tuple(i, j, x, y));
                        }
                    }
                }
                if (board[i][j] == 3) {
                    for (int d = 0; d < 8; d++) {
                        int x = i + "10001222"[d] - '1';
                        int y = j + "22100012"[d] - '1';
                        if (x < 2 || x > 6) continue;
                        if (y < 1 || y > 11) continue;
                        if (board[x][y] == 2) continue;
                        ret = min(ret, tuple(i, j, x, y));
                    }
                }
            }
        }
        assert(ret != tuple(inf, inf, inf, inf));
        return ret;
    }
}

// ------------------------------------------------------------
// Apply the player's move to the local board state.
// ------------------------------------------------------------
void apply_move(int x1, int y1, int x2, int y2, int turn) {
    if (turn == 1) {
        assert(1 <= x1 && x1 <= 6);
        assert(1 <= y1 && y1 <= 11);
        assert(x2 == x1 + 1);
        assert(y2 == y1);
        assert(board[x1][y1] == 1);
        assert(board[x2][y2] == 0);
        board[x1][y1] = 0;
        board[x2][y2] = 1;
    }
    else {
        assert(2 <= x1 && x1 <= 6);
        assert(1 <= y1 && y1 <= 11);
        assert(2 <= x2 && x2 <= 6);
        assert(1 <= y2 && y2 <= 11);
        if (board[x1][y1] == 2) {
            int dx = x1 == x2 ? 0 : x1 < x2 ? 1 : -1;
            int dy = y1 == y2 ? 0 : y1 < y2 ? 1 : -1;
            int x = x1, y = y1;
            int flag = 0;
            while (1) {
                x += dx;
                y += dy;
                if (x < 2 || x > 6) break;
                if (y < 1 || y > 11) break;
                if (board[x][y] != 0) break;
                if (x == x2 && y == y2) { flag = 1; break; }
            }
            assert(flag == 1);
            board[x1][y1] = 0;
            board[x2][y2] = 2;
        }
        else if (board[x1][y1] == 3) {
            int d = max(abs(x1 - x2), abs(y1 - y2));
            assert(d == 1);
            assert(board[x2][y2] != 2);
            board[x1][y1] = 0;
            board[x2][y2] = 3;
        }
        else {
            assert(0);
        }
    }
}

// ------------------------------------------------------------
// Main event loop: handles protocol commands and moves.
// ------------------------------------------------------------
int main() {
    for (int i = 1; i <= 11; i++) board[1][i] = 1; // MOUSE(M)
    board[6][4] = 2; // CAT(C)
    board[6][5] = 2; // CAT(C)
    board[6][7] = 2; // CAT(C)
    board[6][8] = 2; // CAT(C)
    board[6][6] = 3; // NADORI(N)
    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        string cmd;
        in >> cmd;
        if (cmd == "READY") {
            string role; in >> role;
            turn = (role == "FIRST" ? 1 : 2);
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2;
            in >> t1 >> t2;
            auto [x1, y1, x2, y2] = find_move(turn);
            apply_move(x1, y1, x2, y2, turn);
            cout << "MOVE " << x1 << ' ' << y1 << ' ' << x2 << ' ' << y2 << endl;
        }
        else if (cmd == "OPP") {
            int x1, y1, x2, y2, t2;
            in >> x1 >> y1 >> x2 >> y2 >> t2;
            apply_move(x1, y1, x2, y2, turn ^ 3);
        }
        else if (cmd == "FINISH") {
            break;
        }
    }
    return 0;
}