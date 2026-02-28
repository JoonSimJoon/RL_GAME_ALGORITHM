#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Ataxx Alpha-Beta Agent (Enhanced)
 *  ------------------------------------------------------------
 *  - Board size: 7x7 (1-indexed)
 *  - Piece values: 0 = empty, 1 = FIRST(O), 2 = SECOND(X)
 *  - Negamax + Alpha-Beta + Iterative Deepening
 *  - Make/Undo move (no board copies)
 *  - Move ordering by capture count
 * ============================================================
 */

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

int board[8][8];
int turn;
int node_count;

const int INF = 1e9;

// ------------------------------------------------------------
// Undo stack: stores info to reverse a move
// ------------------------------------------------------------
struct UndoInfo {
    int x1, y1, x2, y2;
    int dist;
    int infected[8][2];
    int infected_cnt;
};

UndoInfo undo_stack[128];
int undo_top = 0;

// ------------------------------------------------------------
// Count pieces for a player
// ------------------------------------------------------------
int count_pieces(int player) {
    int cnt = 0;
    for (int i = 1; i <= 7; i++)
        for (int j = 1; j <= 7; j++)
            if (board[i][j] == player) cnt++;
    return cnt;
}

// ------------------------------------------------------------
// Count how many opponent pieces are adjacent to (x,y)
// ------------------------------------------------------------
int count_adj_opp(int x, int y, int opp) {
    int cnt = 0;
    for (int d = 0; d < 8; d++) {
        int nx = x + dx[d], ny = y + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && board[nx][ny] == opp)
            cnt++;
    }
    return cnt;
}

// ------------------------------------------------------------
// Make move on global board, push undo info
// ------------------------------------------------------------
void make_move(int x1, int y1, int x2, int y2, int player) {
    UndoInfo& u = undo_stack[undo_top++];
    u.x1 = x1; u.y1 = y1; u.x2 = x2; u.y2 = y2;
    u.infected_cnt = 0;

    u.dist = max(abs(x2 - x1), abs(y2 - y1));
    if (u.dist == 2) board[x1][y1] = 0;
    board[x2][y2] = player;

    int opp = player ^ 3;
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d], ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && board[nx][ny] == opp) {
            u.infected[u.infected_cnt][0] = nx;
            u.infected[u.infected_cnt][1] = ny;
            u.infected_cnt++;
            board[nx][ny] = player;
        }
    }
}

// ------------------------------------------------------------
// Undo the last move
// ------------------------------------------------------------
void undo_move(int player) {
    UndoInfo& u = undo_stack[--undo_top];
    int opp = player ^ 3;

    // Restore infected pieces
    for (int i = 0; i < u.infected_cnt; i++)
        board[u.infected[i][0]][u.infected[i][1]] = opp;

    // Remove piece from destination
    board[u.x2][u.y2] = 0;

    // Restore source if jump
    if (u.dist == 2) board[u.x1][u.y1] = player;
}

// ------------------------------------------------------------
// Generate all legal moves with their capture scores.
// Returns moves sorted by capture count (descending).
// Splits before jumps at equal capture count.
// ------------------------------------------------------------
struct ScoredMove {
    int x1, y1, x2, y2;
    int score; // higher = better ordering
};

int gen_moves(ScoredMove* moves, int player) {
    int cnt = 0;
    int opp = player ^ 3;

    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (board[x1][y1] != player) continue;
            for (int x2 = x1 - 2; x2 <= x1 + 2; x2++) {
                if (x2 < 1 || x2 > 7) continue;
                for (int y2 = y1 - 2; y2 <= y1 + 2; y2++) {
                    if (y2 < 1 || y2 > 7) continue;
                    if (x2 == x1 && y2 == y1) continue;
                    if (board[x2][y2] != 0) continue;

                    int dist = max(abs(x2 - x1), abs(y2 - y1));
                    int captures = count_adj_opp(x2, y2, opp);
                    // Split gains 1 piece + captures*2, Jump gains captures*2
                    // Score for ordering: captures * 10 + split_bonus
                    int s = captures * 10 + (dist == 1 ? 5 : 0);

                    moves[cnt++] = {x1, y1, x2, y2, s};
                }
            }
        }
    }

    // Sort by score descending (best moves first for better pruning)
    sort(moves, moves + cnt, [](const ScoredMove& a, const ScoredMove& b) {
        return a.score > b.score;
    });

    return cnt;
}

// ------------------------------------------------------------
// Evaluation: piece difference + capture potential
// ------------------------------------------------------------
int evaluate(int player) {
    int opp = player ^ 3;
    int my_cnt = 0, opp_cnt = 0;
    int my_adj = 0, opp_adj = 0;

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] == player) {
                my_cnt++;
                // Count empty neighbors (expansion potential)
                for (int d = 0; d < 8; d++) {
                    int ni = i + dx[d], nj = j + dy[d];
                    if (ni >= 1 && ni <= 7 && nj >= 1 && nj <= 7) {
                        if (board[ni][nj] == opp) my_adj++;
                    }
                }
            } else if (board[i][j] == opp) {
                opp_cnt++;
                for (int d = 0; d < 8; d++) {
                    int ni = i + dx[d], nj = j + dy[d];
                    if (ni >= 1 && ni <= 7 && nj >= 1 && nj <= 7) {
                        if (board[ni][nj] == player) opp_adj++;
                    }
                }
            }
        }
    }

    // Terminal check
    if (my_cnt == 0) return -10000;
    if (opp_cnt == 0) return 10000;

    // Piece count is king in ATAXX
    int score = (my_cnt - opp_cnt) * 100;

    // Capture potential: my pieces adjacent to opponent = offensive power
    score += (my_adj - opp_adj) * 5;

    return score;
}

// ------------------------------------------------------------
// Negamax with Alpha-Beta pruning
// ------------------------------------------------------------
chrono::steady_clock::time_point search_start;
int time_limit_ms;
bool time_up;

int negamax(int depth, int alpha, int beta, int player) {
    node_count++;

    // Time check every 4096 nodes
    if ((node_count & 4095) == 0) {
        auto now = chrono::steady_clock::now();
        int elapsed = chrono::duration_cast<chrono::milliseconds>(now - search_start).count();
        if (elapsed >= time_limit_ms) {
            time_up = true;
            return 0;
        }
    }

    if (depth == 0) return evaluate(player);

    ScoredMove moves[200];
    int n = gen_moves(moves, player);

    if (n == 0) {
        // No moves: check if opponent also has no moves
        ScoredMove opp_moves[200];
        int opp_n = gen_moves(opp_moves, player ^ 3);
        if (opp_n == 0) return evaluate(player); // game over
        // Pass: opponent plays
        return -negamax(depth - 1, -beta, -alpha, player ^ 3);
    }

    for (int i = 0; i < n; i++) {
        auto& m = moves[i];
        make_move(m.x1, m.y1, m.x2, m.y2, player);
        int score = -negamax(depth - 1, -beta, -alpha, player ^ 3);
        undo_move(player);

        if (time_up) return 0;

        if (score > alpha) alpha = score;
        if (alpha >= beta) break;
    }

    return alpha;
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

    ScoredMove moves[200];
    int n = gen_moves(moves, turn);
    if (n == 0) return {-1, -1, -1, -1};
    if (n == 1) return {moves[0].x1, moves[0].y1, moves[0].x2, moves[0].y2};

    int best_x1 = moves[0].x1, best_y1 = moves[0].y1;
    int best_x2 = moves[0].x2, best_y2 = moves[0].y2;

    // Iterative deepening
    for (int depth = 1; depth <= 30; depth++) {
        int best_score = -INF;
        int cur_x1 = -1, cur_y1 = -1, cur_x2 = -1, cur_y2 = -1;

        node_count = 0;

        for (int i = 0; i < n; i++) {
            auto& m = moves[i];
            make_move(m.x1, m.y1, m.x2, m.y2, turn);
            int score = -negamax(depth - 1, -INF, -best_score, turn ^ 3);
            undo_move(turn);

            if (time_up) break;

            if (score > best_score) {
                best_score = score;
                cur_x1 = m.x1; cur_y1 = m.y1;
                cur_x2 = m.x2; cur_y2 = m.y2;

                // Re-sort: put best move first for next iteration
                if (i > 0) {
                    ScoredMove tmp = moves[i];
                    for (int j = i; j > 0; j--) moves[j] = moves[j-1];
                    moves[0] = tmp;
                }
            }
        }

        if (time_up) break;

        // Completed this depth: update best
        if (cur_x1 != -1) {
            best_x1 = cur_x1; best_y1 = cur_y1;
            best_x2 = cur_x2; best_y2 = cur_y2;
        }

        // If found a winning move, stop early
        if (best_score >= 9000) break;
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
    board[1][1] = board[7][7] = 1;
    board[1][7] = board[7][1] = 2;
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
