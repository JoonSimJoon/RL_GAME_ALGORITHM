#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Catch The Mouse - Greedy Heuristic Agent (1-ply)
 *  ------------------------------------------------------------
 *  - Board size: 7x11 (1-indexed)
 *  - Piece values: 0 = empty, 1 = MOUSE(M), 2 = CAT(C), 3 = NADORI(N)
 *  - FIRST = Mouse player, SECOND = Cat+Nadori player
 *  - Strategy: no tree search; scores each legal move with a
 *    hand-crafted heuristic and picks the best immediately.
 *
 *  MOUSE scoring (higher = better for mice):
 *    +row*100        advance bonus (prefer most-forward mouse)
 *    +2000           if destination is row 7 (immediate win)
 *    +500            if column is clear all the way to row 7
 *    -300            if NADORI is adjacent to destination
 *    -200            if CAT or NADORI is directly below destination
 *
 *  CAT/NADORI scoring (higher = better for cats):
 *    NADORI captures mouse         → +5000 + row*100
 *    NADORI approaches mouse       → +200 - dist_to_nearest*20
 *    CAT blocks advanced mouse col → +mouse_row*150
 *    CAT general forward motion    → +row_dest*10
 * ============================================================
 */

int board[8][12]; // 1-based: rows 1-7, cols 1-11
int turn;         // 1 = MOUSE, 2 = CAT+NADORI

const int DX[8] = {0, -1, -1, -1, 0, 1, 1, 1};
const int DY[8] = {1,  1,  0, -1,-1, -1, 0, 1};

// ------------------------------------------------------------
// Score a move for the MOUSE player.
// Positive values favor the mouse side.
// ------------------------------------------------------------
int score_mouse_move(int x1, int y1, int x2, int y2) {
    int s = x2 * 100; // prefer advancing further mice

    if (x2 == 7) return 100000; // immediate win

    // Clear-path bonus: no obstacles from x2+1 to row 7 in same column
    bool clear = true;
    for (int r = x2 + 1; r <= 7; r++) {
        if (board[r][y2] != 0) { clear = false; break; }
    }
    if (clear) s += 500 + (7 - x2) * 50; // closer to goal = bigger bonus

    // Penalty: NADORI adjacent to destination
    for (int d = 0; d < 8; d++) {
        int nx = x2 + DX[d], ny = y2 + DY[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 11) {
            if (board[nx][ny] == 3) { s -= 300; break; }
        }
    }

    // Penalty: blocker directly below destination
    if (x2 < 7 && board[x2 + 1][y2] != 0 && board[x2 + 1][y2] != 1)
        s -= 200; // cat or nadori blocks next step

    return s;
}

// ------------------------------------------------------------
// Score a move for the CAT+NADORI player.
// Positive values favor the cat side.
// ------------------------------------------------------------
int score_cat_move(int x1, int y1, int x2, int y2) {
    int s = 0;

    if (board[x1][y1] == 3) {
        // NADORI move
        if (board[x2][y2] == 1) {
            // Capture a mouse: huge bonus, prefer advanced mice
            return 5000 + x2 * 100;
        }

        // Move toward nearest mouse (Manhattan distance)
        int best_dist_before = INT_MAX, best_dist_after = INT_MAX;
        for (int i = 1; i <= 6; i++) {
            for (int j = 1; j <= 11; j++) {
                if (board[i][j] != 1) continue;
                int db = abs(x1 - i) + abs(y1 - j);
                int da = abs(x2 - i) + abs(y2 - j);
                best_dist_before = min(best_dist_before, db);
                best_dist_after  = min(best_dist_after,  da);
            }
        }
        // Reward closing distance; larger gain = better
        s = 200 + (best_dist_before - best_dist_after) * 30;
        // Slight preference for being near advanced mice (high row)
        for (int j = 1; j <= 11; j++) {
            for (int i = 6; i >= 1; i--) {
                if (board[i][j] == 1) {
                    s += i * 5; // bonus proportional to mouse row
                    break;
                }
            }
        }
    }
    else {
        // CAT move: position to block most advanced mouse
        // Find highest row (closest to row 7) mouse in each column
        int mouse_max_row[12] = {};
        for (int i = 1; i <= 6; i++)
            for (int j = 1; j <= 11; j++)
                if (board[i][j] == 1 && i > mouse_max_row[j])
                    mouse_max_row[j] = i;

        // Blocking bonus: cat lands directly below a mouse's column
        if (mouse_max_row[y2] > 0 && x2 > mouse_max_row[y2]) {
            // Cat is ahead (lower row = higher index in our notation... wait)
            // x2 > mouse_max_row[y2]: cat is BELOW mouse in col y2 → blocks it
            s += mouse_max_row[y2] * 150;
            if (mouse_max_row[y2] >= 5) s += 2000; // very advanced mouse → urgent
        }

        // Also reward cats that are adjacent to multiple mice columns
        for (int dy = -1; dy <= 1; dy++) {
            int col = y2 + dy;
            if (col >= 1 && col <= 11 && mouse_max_row[col] > 0)
                s += mouse_max_row[col] * 20;
        }

        // Small bonus for moving forward (toward mice)
        s += x2 * 10;
    }

    return s;
}

// ------------------------------------------------------------
// Generate all legal moves and pick the highest-scored one.
// ------------------------------------------------------------
tuple<int,int,int,int> find_move(int side) {
    int best_score = INT_MIN;
    int bx1 = -1, by1 = -1, bx2 = -1, by2 = -1;

    if (side == 1) {
        // MOUSE moves: (i,j) → (i+1,j) if empty
        for (int i = 1; i <= 6; i++) {
            for (int j = 1; j <= 11; j++) {
                if (board[i][j] != 1) continue;
                if (board[i+1][j] != 0) continue;

                int s = score_mouse_move(i, j, i+1, j);
                if (s > best_score) {
                    best_score = s;
                    bx1 = i; by1 = j; bx2 = i+1; by2 = j;
                }
            }
        }
    }
    else {
        // CAT/NADORI moves
        for (int i = 2; i <= 6; i++) {
            for (int j = 1; j <= 11; j++) {
                if (board[i][j] == 2) {
                    // CAT: queen slide through empty cells
                    for (int d = 0; d < 8; d++) {
                        int x = i, y = j;
                        while (true) {
                            x += DX[d]; y += DY[d];
                            if (x < 2 || x > 6 || y < 1 || y > 11) break;
                            if (board[x][y] != 0) break;

                            int s = score_cat_move(i, j, x, y);
                            if (s > best_score) {
                                best_score = s;
                                bx1 = i; by1 = j; bx2 = x; by2 = y;
                            }
                        }
                    }
                }
                else if (board[i][j] == 3) {
                    // NADORI: king 1-step, cannot land on cat
                    for (int d = 0; d < 8; d++) {
                        int x = i + DX[d], y = j + DY[d];
                        if (x < 2 || x > 6 || y < 1 || y > 11) continue;
                        if (board[x][y] == 2) continue;

                        int s = score_cat_move(i, j, x, y);
                        if (s > best_score) {
                            best_score = s;
                            bx1 = i; by1 = j; bx2 = x; by2 = y;
                        }
                    }
                }
            }
        }
    }

    assert(bx1 != -1);
    return {bx1, by1, bx2, by2};
}

// ------------------------------------------------------------
// Apply a move to the board.
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
            // CAT queen slide
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
            assert(board[x2][y2] != 2);
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
