#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Catch The Mouse - Random Agent
 *  ------------------------------------------------------------
 *  - Board size: 7x11 (1-indexed)
 *  - Piece values: 0 = empty, 1 = MOUSE(M), 2 = CAT(C), 3 = NADORI(N)
 *  - FIRST = Mouse player, SECOND = Cat+Nadori player
 *  - Strategy: pick a uniformly random legal move each turn.
 * ============================================================
 */

constexpr int inf = 1 << 30;

int board[8][12]; // 1-based: rows 1-7, cols 1-11
int turn;         // 1 = MOUSE, 2 = CAT+NADORI

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

const int DX[8] = {0, -1, -1, -1, 0, 1, 1, 1};
const int DY[8] = {1,  1,  0, -1,-1, -1, 0, 1};

// ------------------------------------------------------------
// Generate all legal moves for the current player.
// Returns vector of (x1, y1, x2, y2).
// ------------------------------------------------------------
vector<tuple<int,int,int,int>> gen_moves(int side) {
    vector<tuple<int,int,int,int>> moves;

    if (side == 1) {
        // MOUSE: each mouse at (i,j) can move down to (i+1,j) if empty
        for (int i = 1; i <= 6; i++)
            for (int j = 1; j <= 11; j++)
                if (board[i][j] == 1 && board[i+1][j] == 0)
                    moves.emplace_back(i, j, i+1, j);
    }
    else {
        for (int i = 2; i <= 6; i++) {
            for (int j = 1; j <= 11; j++) {
                if (board[i][j] == 2) {
                    // CAT: slides like a queen through empty cells (rows 2-6)
                    for (int d = 0; d < 8; d++) {
                        int x = i, y = j;
                        while (true) {
                            x += DX[d]; y += DY[d];
                            if (x < 2 || x > 6 || y < 1 || y > 11) break;
                            if (board[x][y] != 0) break;
                            moves.emplace_back(i, j, x, y);
                        }
                    }
                }
                else if (board[i][j] == 3) {
                    // NADORI: moves 1 step like a king (rows 2-6)
                    // Cannot land on a CAT cell; CAN land on MOUSE (capture)
                    for (int d = 0; d < 8; d++) {
                        int x = i + DX[d], y = j + DY[d];
                        if (x < 2 || x > 6 || y < 1 || y > 11) continue;
                        if (board[x][y] == 2) continue; // blocked by cat
                        moves.emplace_back(i, j, x, y);
                    }
                }
            }
        }
    }
    return moves;
}

// ------------------------------------------------------------
// Pick a random legal move.
// ------------------------------------------------------------
tuple<int,int,int,int> find_move(int side) {
    auto moves = gen_moves(side);
    assert(!moves.empty());
    return moves[rng() % moves.size()];
}

// ------------------------------------------------------------
// Apply a move to the board (used for both players).
// ------------------------------------------------------------
void apply_move(int x1, int y1, int x2, int y2, int side) {
    if (side == 1) {
        assert(board[x1][y1] == 1 && board[x2][y2] == 0);
        assert(x2 == x1 + 1 && y2 == y1);
        board[x1][y1] = 0;
        board[x2][y2] = 1;
    }
    else {
        if (board[x1][y1] == 2) {
            // CAT queen slide: validate path is clear
            int ddx = (x2 > x1) ? 1 : (x2 < x1) ? -1 : 0;
            int ddy = (y2 > y1) ? 1 : (y2 < y1) ? -1 : 0;
            int x = x1, y = y1;
            bool ok = false;
            while (true) {
                x += ddx; y += ddy;
                if (x < 2 || x > 6 || y < 1 || y > 11) break;
                if (board[x][y] != 0) break;
                if (x == x2 && y == y2) { ok = true; break; }
            }
            assert(ok);
            board[x1][y1] = 0;
            board[x2][y2] = 2;
        }
        else if (board[x1][y1] == 3) {
            assert(max(abs(x1-x2), abs(y1-y2)) == 1);
            assert(board[x2][y2] != 2); // not a cat cell
            board[x1][y1] = 0;
            board[x2][y2] = 3; // overwrites mouse if captured
        }
        else { assert(0); }
    }
}

// ------------------------------------------------------------
// Main event loop
// ------------------------------------------------------------
int main() {
    // Initial board setup
    for (int i = 1; i <= 11; i++) board[1][i] = 1; // 11 mice at row 1
    board[6][4] = 2; board[6][5] = 2;               // 4 cats
    board[6][7] = 2; board[6][8] = 2;
    board[6][6] = 3;                                 // 1 nadori

    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        string cmd;
        in >> cmd;

        if (cmd == "READY") {
            string role; in >> role;
            turn = (role == "FIRST") ? 1 : 2;
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2; in >> t1 >> t2;
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
