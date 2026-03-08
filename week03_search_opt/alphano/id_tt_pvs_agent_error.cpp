#include <bits/stdc++.h>
using namespace std;

// ---- Global Board State ----
int board[8][8];
int turn;
const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
const int INF = 1000000;

// ---- Zobrist Hashing ----
uint64_t zobrist[8][8][3]; // [row][col][piece: 0=empty, 1=FIRST, 2=SECOND]
uint64_t board_hash;

// ---- Transposition Table ----
const int PV_NODE = 0, CUT_NODE = 1, ALL_NODE = 2;
struct TTEntry {
    int best_x1, best_y1, best_x2, best_y2;
    int flag, depth, value;
};
unordered_map<uint64_t, TTEntry> tt;

// ---- Make/Undo Stack ----
struct UndoInfo {
    int x1, y1, x2, y2, dist;
    int infected[8][2];
    int infected_cnt;
};
UndoInfo undo_stack[128];
int undo_top = 0;
int current_player;

// ---- Move Ordering ----
struct ScoredMove {
    int x1, y1, x2, y2, score;
    bool operator<(const ScoredMove& other) const {
        return score > other.score; // descending
    }
};

// ---- Time Management ----
chrono::steady_clock::time_point search_start;
int time_limit_ms;
bool time_up;
int node_count;

// ---- Zobrist Initialization ----
void init_zobrist() {
    uint64_t seed = 987654321ULL;
    auto xorshift64 = [&]() {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        return seed;
    };

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            for (int p = 0; p <= 2; p++) {
                zobrist[i][j][p] = xorshift64();
            }
        }
    }
}

uint64_t compute_hash() {
    uint64_t h = 0;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] != 0) {
                h ^= zobrist[i][j][board[i][j]];
            }
        }
    }
    return h;
}

// ---- Move Application ----
void apply_my_move(int x1, int y1, int x2, int y2) {
    assert(board[x2][y2] == 0);
    board[x2][y2] = turn;
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    if (dist == 2) {
        board[x1][y1] = 0;
    }
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d], ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (board[nx][ny] == (turn ^ 3)) {
                board[nx][ny] = turn;
            }
        }
    }
}

void apply_opp_move(int x1, int y1, int x2, int y2) {
    int opp = turn ^ 3;
    assert(board[x2][y2] == 0);
    board[x2][y2] = opp;
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    if (dist == 2) {
        board[x1][y1] = 0;
    }
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d], ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (board[nx][ny] == turn) {
                board[nx][ny] = opp;
            }
        }
    }
}

// ---- Make/Undo with Hash Update ----
void make_move(int x1, int y1, int x2, int y2, int player) {
    UndoInfo& u = undo_stack[undo_top++];
    u.x1 = x1; u.y1 = y1; u.x2 = x2; u.y2 = y2;
    u.dist = max(abs(x2 - x1), abs(y2 - y1));
    u.infected_cnt = 0;

    int opp = player ^ 3;

    // Place piece at destination
    board_hash ^= zobrist[x2][y2][0];
    board_hash ^= zobrist[x2][y2][player];
    board[x2][y2] = player;

    // Remove from source if jump
    if (u.dist == 2) {
        board_hash ^= zobrist[x1][y1][player];
        board_hash ^= zobrist[x1][y1][0];
        board[x1][y1] = 0;
    }

    // Infect adjacent opponents
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d], ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (board[nx][ny] == opp) {
                board_hash ^= zobrist[nx][ny][opp];
                board_hash ^= zobrist[nx][ny][player];
                board[nx][ny] = player;
                u.infected[u.infected_cnt][0] = nx;
                u.infected[u.infected_cnt][1] = ny;
                u.infected_cnt++;
            }
        }
    }

    current_player = opp;
}

void undo_move() {
    assert(undo_top > 0);
    UndoInfo& u = undo_stack[--undo_top];

    int opp = current_player;
    int player = opp ^ 3;

    // Restore infected pieces
    for (int i = 0; i < u.infected_cnt; i++) {
        int nx = u.infected[i][0];
        int ny = u.infected[i][1];
        board_hash ^= zobrist[nx][ny][player];
        board_hash ^= zobrist[nx][ny][opp];
        board[nx][ny] = opp;
    }

    // Restore source if jump
    if (u.dist == 2) {
        board_hash ^= zobrist[u.x1][u.y1][0];
        board_hash ^= zobrist[u.x1][u.y1][player];
        board[u.x1][u.y1] = player;
    }

    // Remove from destination
    board_hash ^= zobrist[u.x2][u.y2][player];
    board_hash ^= zobrist[u.x2][u.y2][0];
    board[u.x2][u.y2] = 0;

    current_player = player;
}

// ---- Move Generation with Ordering ----
int gen_moves(ScoredMove* moves, int player, TTEntry* tt_hint = nullptr) {
    int cnt = 0;
    int opp = player ^ 3;

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] == player) {
                for (int di = -2; di <= 2; di++) {
                    for (int dj = -2; dj <= 2; dj++) {
                        if (di == 0 && dj == 0) continue;
                        int ni = i + di, nj = j + dj;
                        if (ni >= 1 && ni <= 7 && nj >= 1 && nj <= 7) {
                            if (board[ni][nj] == 0) {
                                int dist = max(abs(di), abs(dj));
                                int captures = 0;
                                for (int d = 0; d < 8; d++) {
                                    int nx = ni + dx[d], ny = nj + dy[d];
                                    if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
                                        if (board[nx][ny] == opp) {
                                            captures++;
                                        }
                                    }
                                }

                                int score = captures * 10 + (dist == 1 ? 5 : 0);

                                // TT move bonus
                                if (tt_hint && tt_hint->best_x1 == i && tt_hint->best_y1 == j &&
                                    tt_hint->best_x2 == ni && tt_hint->best_y2 == nj) {
                                    score += 1000;
                                }

                                moves[cnt++] = {i, j, ni, nj, score};
                            }
                        }
                    }
                }
            }
        }
    }

    sort(moves, moves + cnt);
    return cnt;
}

// ---- Evaluation ----
int evaluate(int player) {
    int my_cnt = 0, opp_cnt = 0;
    int my_adj = 0, opp_adj = 0;
    int opp = player ^ 3;

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] == player) {
                my_cnt++;
                // Count adjacent to opponent
                for (int d = 0; d < 8; d++) {
                    int nx = i + dx[d], ny = j + dy[d];
                    if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
                        if (board[nx][ny] == opp) {
                            my_adj++;
                        }
                    }
                }
            } else if (board[i][j] == opp) {
                opp_cnt++;
                // Count adjacent to player
                for (int d = 0; d < 8; d++) {
                    int nx = i + dx[d], ny = j + dy[d];
                    if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
                        if (board[nx][ny] == player) {
                            opp_adj++;
                        }
                    }
                }
            }
        }
    }

    if (my_cnt == 0) return -100000;
    if (opp_cnt == 0) return 100000;

    return (my_cnt - opp_cnt) * 100 + (my_adj - opp_adj) * 5;
}

// ---- PVS Search ----
int pvs(int depth, int alpha, int beta) {
    node_count++;

    // Time check every 4096 nodes
    if ((node_count & 4095) == 0) {
        auto now = chrono::steady_clock::now();
        int elapsed = chrono::duration_cast<chrono::milliseconds>(now - search_start).count();
        if (elapsed >= time_limit_ms) {
            time_up = true;
        }
    }

    if (time_up) return 0;

    // TT lookup
    TTEntry* tt_entry = nullptr;
    auto it = tt.find(board_hash);
    if (it != tt.end()) {
        tt_entry = &it->second;
        if (tt_entry->depth >= depth) {
            if (tt_entry->flag == PV_NODE) {
                return tt_entry->value;
            } else if (tt_entry->flag == CUT_NODE) {
                alpha = max(alpha, tt_entry->value);
            } else if (tt_entry->flag == ALL_NODE) {
                beta = min(beta, tt_entry->value);
            }
            if (alpha >= beta) {
                return tt_entry->value;
            }
        }
    }

    // Terminal or depth 0
    ScoredMove moves[256];
    int n = gen_moves(moves, current_player, tt_entry);

    if (n == 0 || depth == 0) {
        return evaluate(current_player);
    }

    int best_value = -INF;
    int alpha_original = alpha;
    ScoredMove best_move = moves[0];

    for (int i = 0; i < n; i++) {
        make_move(moves[i].x1, moves[i].y1, moves[i].x2, moves[i].y2, current_player);

        int value;
        if (i == 0) {
            // Full window search
            value = -pvs(depth - 1, -beta, -alpha);
        } else {
            // Null window search
            value = -pvs(depth - 1, -alpha - 1, -alpha);
            // Re-search if in window
            if (!time_up && alpha < value && value < beta) {
                value = -pvs(depth - 1, -beta, -value);
            }
        }

        undo_move();

        if (time_up) return 0;

        if (value > best_value) {
            best_value = value;
            best_move = moves[i];
        }

        alpha = max(alpha, value);
        if (alpha >= beta) {
            break; // Beta cutoff
        }
    }

    // Store in TT
    TTEntry new_entry;
    new_entry.best_x1 = best_move.x1;
    new_entry.best_y1 = best_move.y1;
    new_entry.best_x2 = best_move.x2;
    new_entry.best_y2 = best_move.y2;
    new_entry.depth = depth;
    new_entry.value = best_value;

    if (best_value <= alpha_original) {
        new_entry.flag = ALL_NODE;
    } else if (best_value >= beta) {
        new_entry.flag = CUT_NODE;
    } else {
        new_entry.flag = PV_NODE;
    }

    tt[board_hash] = new_entry;

    return best_value;
}

// ---- Iterative Deepening ----
ScoredMove find_move(int my_time) {
    // Time management
    if (my_time > 30000) {
        time_limit_ms = 800;
    } else if (my_time > 10000) {
        time_limit_ms = 400;
    } else if (my_time > 3000) {
        time_limit_ms = 150;
    } else {
        time_limit_ms = 50;
    }

    search_start = chrono::steady_clock::now();
    time_up = false;
    node_count = 0;
    current_player = turn;
    board_hash = compute_hash();
    undo_top = 0;

    ScoredMove root_moves[256];
    int n = gen_moves(root_moves, turn);

    if (n == 0) {
        return {0, 0, 0, 0, 0};
    }

    ScoredMove best_move = root_moves[0];

    for (int depth = 1; depth <= 50; depth++) {
        auto now = chrono::steady_clock::now();
        int elapsed = chrono::duration_cast<chrono::milliseconds>(now - search_start).count();
        if (elapsed > time_limit_ms * 85 / 100) {
            break;
        }

        int best_score = -INF;
        int best_idx = 0;

        for (int i = 0; i < n; i++) {
            make_move(root_moves[i].x1, root_moves[i].y1, root_moves[i].x2, root_moves[i].y2, turn);

            int score = -pvs(depth - 1, -INF, -best_score);

            undo_move();

            if (time_up) break;

            root_moves[i].score = score;

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        if (time_up) break;

        // Move best to front
        if (best_idx > 0) {
            swap(root_moves[0], root_moves[best_idx]);
        }

        best_move = root_moves[0];

        cerr << "Depth " << depth << " completed, score=" << best_score
             << " nodes=" << node_count << endl;
    }

    return best_move;
}

// ---- Main ----
int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    init_zobrist();

    memset(board, 0, sizeof(board));
    board[1][1] = board[7][7] = 1;
    board[1][7] = board[7][1] = 2;

    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        string cmd;
        in >> cmd;

        if (cmd == "FIRST") {
            turn = 1;
            ScoredMove m = find_move(100000);
            cout << m.x1 << " " << m.y1 << " " << m.x2 << " " << m.y2 << endl;
            apply_my_move(m.x1, m.y1, m.x2, m.y2);
        }
        else if (cmd == "SECOND") {
            turn = 2;
        }
        else if (cmd == "THIRD") {
            turn = 2;
        }
        else if (cmd == "OPP") {
            int x1, y1, x2, y2, t;
            in >> x1 >> y1 >> x2 >> y2 >> t;
            apply_opp_move(x1, y1, x2, y2);

            ScoredMove m = find_move(t);
            cout << m.x1 << " " << m.y1 << " " << m.x2 << " " << m.y2 << endl;
            apply_my_move(m.x1, m.y1, m.x2, m.y2);
        }
        else if (cmd == "END") {
            break;
        }
    }

    return 0;
}
