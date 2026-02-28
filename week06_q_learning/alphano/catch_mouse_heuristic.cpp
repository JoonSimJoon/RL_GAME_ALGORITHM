#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Catch The Mouse - Minimax Agent with Alpha-Beta Pruning
 *  ------------------------------------------------------------
 *  - Board size: 7x11 (1-indexed)
 *  - Piece values: 0 = empty, 1 = MOUSE(M), 2 = CAT(C), 3 = NADORI(N)
 *  - FIRST = Mouse player, SECOND = Cat+Nadori player
 *  - Mice move down 1 row, Cats slide like queens (rows 2-6),
 *    Nadori moves like king (rows 2-6, can capture mice)
 *  - Mice win by reaching row 7
 *  - Negamax + Alpha-Beta + Iterative Deepening
 *  - Make/Undo move (no board copies)
 *  - Move ordering by strategic priority
 * ============================================================
 */

const int DX[8] = {0, -1, -1, -1, 0, 1, 1, 1};
const int DY[8] = {1, 1, 0, -1, -1, -1, 0, 1};

int board[8][12]; // 1-indexed, rows 1-7, cols 1-11
int turn;         // 1 = MOUSE(FIRST), 2 = CAT+NADORI(SECOND)
int node_count;

const int INF = 1e9;

// ------------------------------------------------------------
// Time management
// ------------------------------------------------------------
chrono::steady_clock::time_point search_start;
int time_limit_ms;
bool time_up;

// ------------------------------------------------------------
// Move struct with ordering score
// ------------------------------------------------------------
struct Move {
    int x1, y1, x2, y2;
    int score;
};

// ------------------------------------------------------------
// Undo info for make/undo pattern
// ------------------------------------------------------------
struct UndoInfo {
    int x1, y1, x2, y2;
    int piece;    // what piece moved (1=mouse, 2=cat, 3=nadori)
    int captured; // what was at destination (0=empty, 1=mouse)
};

UndoInfo undo_stack[256];
int undo_top = 0;

// ------------------------------------------------------------
// Make move on global board, push undo info
// ------------------------------------------------------------
void make_move(int x1, int y1, int x2, int y2) {
    UndoInfo& u = undo_stack[undo_top++];
    u.x1 = x1; u.y1 = y1; u.x2 = x2; u.y2 = y2;
    u.piece = board[x1][y1];
    u.captured = board[x2][y2];
    board[x1][y1] = 0;
    board[x2][y2] = u.piece;
}

// ------------------------------------------------------------
// Undo last move
// ------------------------------------------------------------
void undo_move() {
    UndoInfo& u = undo_stack[--undo_top];
    board[u.x1][u.y1] = u.piece;
    board[u.x2][u.y2] = u.captured;
}

// ------------------------------------------------------------
// Generate moves for mice (side=1)
// Mice move down 1 row to an empty cell.
// Sorted by strategic priority (most advanced + clear path first).
// ------------------------------------------------------------
int gen_moves_mice(Move* moves) {
    int cnt = 0;
    for (int i = 6; i >= 1; i--) {
        for (int j = 1; j <= 11; j++) {
            if (board[i][j] != 1) continue;
            if (board[i + 1][j] != 0) continue;

            int s = i * 100;
            if (i == 6) {
                s = 10000; // winning move
            } else {
                bool clear = true;
                for (int r = i + 2; r <= 7; r++) {
                    if (board[r][j] != 0) { clear = false; break; }
                }
                if (clear) s += 500 + i * 50;
            }
            moves[cnt++] = {i, j, i + 1, j, s};
        }
    }
    sort(moves, moves + cnt, [](const Move& a, const Move& b) {
        return a.score > b.score;
    });
    return cnt;
}

// ------------------------------------------------------------
// Generate moves for cats+nadori (side=2)
// Cats slide like queens (rows 2-6, empty cells only).
// Nadori moves 1 step like king (rows 2-6, can't land on cat,
// CAN land on mouse to capture it).
// Sorted by: nadori captures > blocking advanced mice > other.
// ------------------------------------------------------------
int gen_moves_cats(Move* moves) {
    int cnt = 0;

    // Pre-scan: highest row of mouse per column
    int mouse_max_row[12] = {};
    for (int i = 1; i <= 7; i++)
        for (int j = 1; j <= 11; j++)
            if (board[i][j] == 1 && i > mouse_max_row[j])
                mouse_max_row[j] = i;

    for (int i = 2; i <= 6; i++) {
        for (int j = 1; j <= 11; j++) {
            if (board[i][j] == 2) {
                // Cat: queen-like sliding through empty cells
                for (int d = 0; d < 8; d++) {
                    int x = i, y = j;
                    while (true) {
                        x += DX[d]; y += DY[d];
                        if (x < 2 || x > 6 || y < 1 || y > 11) break;
                        if (board[x][y] != 0) break;

                        int s = 0;
                        // Bonus for blocking an advanced mouse's column
                        if (mouse_max_row[y] > 0 && x > mouse_max_row[y]) {
                            s = mouse_max_row[y] * 100;
                            if (mouse_max_row[y] >= 5) s += 1500;
                        }
                        moves[cnt++] = {i, j, x, y, s};
                    }
                }
            }
            else if (board[i][j] == 3) {
                // Nadori: king-like, 1 step in any direction
                for (int d = 0; d < 8; d++) {
                    int x = i + DX[d], y = j + DY[d];
                    if (x < 2 || x > 6 || y < 1 || y > 11) continue;
                    if (board[x][y] == 2) continue; // can't land on cat

                    int s = 0;
                    if (board[x][y] == 1) {
                        // Capturing a mouse! Very high priority.
                        s = 5000 + x * 100;
                    } else {
                        // Move toward nearest advanced mouse
                        for (int mj = 1; mj <= 11; mj++) {
                            if (mouse_max_row[mj] >= 3) {
                                int dist = abs(x - mouse_max_row[mj]) + abs(y - mj);
                                s = max(s, 200 - dist * 20);
                            }
                        }
                    }
                    moves[cnt++] = {i, j, x, y, s};
                }
            }
        }
    }

    sort(moves, moves + cnt, [](const Move& a, const Move& b) {
        return a.score > b.score;
    });
    return cnt;
}

// ------------------------------------------------------------
// Evaluate from mice perspective (positive = good for mice)
// Key factors: mouse advancement, clear paths, multiple threats.
// ------------------------------------------------------------
int evaluate() {
    int score = 0;
    int mice_count = 0;
    int can_move = 0;
    int threats_1 = 0, threats_2 = 0;

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 11; j++) {
            if (board[i][j] != 1) continue;
            mice_count++;

            if (i == 7) return 10000; // mouse reached goal

            score += i * 40; // advancement bonus

            if (i <= 6 && board[i + 1][j] == 0) {
                can_move++;
                score += 30; // mobility bonus

                // Check clear path to row 7
                bool clear = true;
                for (int r = i + 1; r <= 7; r++) {
                    if (board[r][j] != 0) { clear = false; break; }
                }
                if (clear) {
                    int steps = 7 - i;
                    if (steps == 1) { score += 3000; threats_1++; }
                    else if (steps == 2) { score += 1000; threats_2++; }
                    else if (steps == 3) { score += 400; }
                    else { score += 200; }
                }
            }
        }
    }

    if (mice_count == 0) return -10000; // all mice captured
    if (can_move == 0) return -9000;    // all mice blocked

    score += mice_count * 100; // having more mice is better

    // Multiple simultaneous threats are devastating
    if (threats_1 >= 2) score += 5000;
    else if (threats_1 >= 1 && threats_2 >= 1) score += 2000;
    else if (threats_2 >= 2) score += 1000;

    return score;
}

// ------------------------------------------------------------
// Negamax with Alpha-Beta pruning
// Returns score from perspective of `side`.
// ------------------------------------------------------------
int negamax(int depth, int alpha, int beta, int side) {
    node_count++;

    // Time check every 4096 nodes
    if ((node_count & 4095) == 0) {
        int elapsed = chrono::duration_cast<chrono::milliseconds>(
            chrono::steady_clock::now() - search_start).count();
        if (elapsed >= time_limit_ms) { time_up = true; return 0; }
    }

    // Terminal: mouse on row 7
    for (int j = 1; j <= 11; j++) {
        if (board[7][j] == 1) {
            return side == 1 ? (10000 + depth) : -(10000 + depth);
        }
    }

    if (depth == 0) {
        int e = evaluate();
        return side == 1 ? e : -e;
    }

    Move moves[400];
    int n = (side == 1) ? gen_moves_mice(moves) : gen_moves_cats(moves);

    if (n == 0) {
        if (side == 1) return -(9000 + depth); // mice can't move
        return 0; // cats can't move (very unlikely)
    }

    int best = -INF;
    for (int i = 0; i < n; i++) {
        auto& m = moves[i];
        make_move(m.x1, m.y1, m.x2, m.y2);

        int val;
        if (side == 1 && m.x2 == 7) {
            val = 10000 + depth; // immediate win
        } else {
            val = -negamax(depth - 1, -beta, -alpha, side ^ 3);
        }

        undo_move();
        if (time_up) return 0;

        if (val > best) best = val;
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }

    return best;
}

// ------------------------------------------------------------
// Find best move with iterative deepening
// ------------------------------------------------------------
tuple<int, int, int, int> find_move(int my_time) {
    // Time management
    if (my_time > 30000) time_limit_ms = 800;
    else if (my_time > 10000) time_limit_ms = 400;
    else if (my_time > 3000) time_limit_ms = 150;
    else time_limit_ms = 50;

    search_start = chrono::steady_clock::now();
    time_up = false;

    Move moves[400];
    int n = (turn == 1) ? gen_moves_mice(moves) : gen_moves_cats(moves);

    if (n == 0) return {-1, -1, -1, -1};
    if (n == 1) return {moves[0].x1, moves[0].y1, moves[0].x2, moves[0].y2};

    int best_x1 = moves[0].x1, best_y1 = moves[0].y1;
    int best_x2 = moves[0].x2, best_y2 = moves[0].y2;

    // Iterative deepening
    for (int depth = 1; depth <= 30; depth++) {
        int best_score = -INF;
        int cx1 = -1, cy1 = -1, cx2 = -1, cy2 = -1;
        node_count = 0;

        for (int i = 0; i < n; i++) {
            auto& m = moves[i];
            make_move(m.x1, m.y1, m.x2, m.y2);

            int score;
            if (turn == 1 && m.x2 == 7) {
                score = 10000 + depth;
            } else {
                score = -negamax(depth - 1, -INF, -best_score, turn ^ 3);
            }

            undo_move();
            if (time_up) break;

            if (score > best_score) {
                best_score = score;
                cx1 = m.x1; cy1 = m.y1;
                cx2 = m.x2; cy2 = m.y2;

                // Re-sort: put best move first for next iteration
                if (i > 0) {
                    Move tmp = moves[i];
                    for (int j = i; j > 0; j--) moves[j] = moves[j - 1];
                    moves[0] = tmp;
                }
            }
        }

        if (time_up) break;

        if (cx1 != -1) {
            best_x1 = cx1; best_y1 = cy1;
            best_x2 = cx2; best_y2 = cy2;
        }

        if (best_score >= 9000) break; // winning
    }

    return {best_x1, best_y1, best_x2, best_y2};
}

// ------------------------------------------------------------
// Apply move to board (protocol handler, with validation)
// ------------------------------------------------------------
void apply_move(int x1, int y1, int x2, int y2, int side) {
    if (side == 1) {
        // Mouse: move down 1 row
        assert(1 <= x1 && x1 <= 6);
        assert(1 <= y1 && y1 <= 11);
        assert(x2 == x1 + 1 && y2 == y1);
        assert(board[x1][y1] == 1);
        assert(board[x2][y2] == 0);
        board[x1][y1] = 0;
        board[x2][y2] = 1;
    } else {
        assert(2 <= x1 && x1 <= 6 && 1 <= y1 && y1 <= 11);
        assert(2 <= x2 && x2 <= 6 && 1 <= y2 && y2 <= 11);
        if (board[x1][y1] == 2) {
            // Cat: queen slide validation
            int ddx = (x2 > x1) ? 1 : (x2 < x1) ? -1 : 0;
            int ddy = (y2 > y1) ? 1 : (y2 < y1) ? -1 : 0;
            int x = x1, y = y1;
            bool valid = false;
            while (true) {
                x += ddx; y += ddy;
                if (x < 2 || x > 6 || y < 1 || y > 11) break;
                if (board[x][y] != 0) break;
                if (x == x2 && y == y2) { valid = true; break; }
            }
            assert(valid);
            board[x1][y1] = 0;
            board[x2][y2] = 2;
        } else if (board[x1][y1] == 3) {
            // Nadori: king move validation
            assert(max(abs(x1 - x2), abs(y1 - y2)) == 1);
            assert(board[x2][y2] != 2);
            board[x1][y1] = 0;
            board[x2][y2] = 3; // may capture mouse
        } else {
            assert(0);
        }
    }
}

// ------------------------------------------------------------
// Main event loop: handles protocol commands and moves.
// ------------------------------------------------------------
int main() {
    // Initial board setup
    for (int i = 1; i <= 11; i++) board[1][i] = 1; // 11 mice at row 1
    board[6][4] = 2; board[6][5] = 2;               // 4 cats at row 6
    board[6][7] = 2; board[6][8] = 2;
    board[6][6] = 3;                                  // nadori at center

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
            auto [x1, y1, x2, y2] = find_move(t1);
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
