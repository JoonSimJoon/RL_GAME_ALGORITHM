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

// ---- Move ordering
struct ScoredMove {
    int x1, y1, x2, y2, score;
    bool operator<(const ScoredMove& other) const {
        return score > other.score;
    }
};

// ---- Time management
chrono::time_point<chrono::high_resolution_clock> search_start;
int time_limit_ms;
bool time_up;
int nodes_checked;

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

int count_adjacent_to(int x, int y, int player) {
    int cnt = 0;
    for (int d = 0; d < 8; d++) {
        int nx = x + dx[d];
        int ny = y + dy[d];
        if (in_bounds(nx, ny) && board[nx][ny] == player) {
            cnt++;
        }
    }
    return cnt;
}

void check_time() {
    nodes_checked++;
    if (nodes_checked % 4096 == 0) {
        auto now = chrono::high_resolution_clock::now();
        int elapsed = chrono::duration_cast<chrono::milliseconds>(now - search_start).count();
        if (elapsed >= time_limit_ms) {
            time_up = true;
        }
    }
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

    if (undo.dist == 2) {
        board[x1][y1] = 0;
    }
    board[x2][y2] = player;

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

    for (int i = 0; i < undo.infected_cnt; i++) {
        int nx = undo.infected[i][0];
        int ny = undo.infected[i][1];
        board[nx][ny] = player ^ 3;
    }

    board[undo.x2][undo.y2] = 0;
    if (undo.dist == 2) {
        board[undo.x1][undo.y1] = player;
    }
}

// ---- Evaluation
int evaluate(int player) {
    int my_pieces = count_pieces(player);
    int opp_pieces = count_pieces(player ^ 3);

    int my_adj_to_opp = 0;
    int opp_adj_to_me = 0;

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] == player) {
                my_adj_to_opp += count_adjacent_to(i, j, player ^ 3);
            } else if (board[i][j] == (player ^ 3)) {
                opp_adj_to_me += count_adjacent_to(i, j, player);
            }
        }
    }

    return (my_pieces - opp_pieces) * 100 + (my_adj_to_opp - opp_adj_to_me) * 5;
}

// ---- Move generation
int gen_moves(ScoredMove* moves, int player) {
    int cnt = 0;

    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (board[x1][y1] != player) continue;

            for (int x2 = 1; x2 <= 7; x2++) {
                for (int y2 = 1; y2 <= 7; y2++) {
                    if (board[x2][y2] != 0) continue;
                    int d = dist(x1, y1, x2, y2);
                    if (d == 0 || d > 2) continue;

                    int score = count_adjacent_to(x2, y2, player ^ 3);
                    if (d == 1) score += 5;

                    moves[cnt++] = {x1, y1, x2, y2, score};
                }
            }
        }
    }

    sort(moves, moves + cnt);
    return cnt;
}

// ---- Negamax search
int negamax(int depth, int alpha, int beta, int player) {
    check_time();
    if (time_up) return 0;

    if (depth == 0) {
        return evaluate(player);
    }

    ScoredMove moves[256];
    int move_cnt = gen_moves(moves, player);

    if (move_cnt == 0) {
        return evaluate(player);
    }

    for (int i = 0; i < move_cnt; i++) {
        make_move(moves[i].x1, moves[i].y1, moves[i].x2, moves[i].y2, player);
        int score = -negamax(depth - 1, -beta, -alpha, player ^ 3);
        undo_move(player);

        if (score > alpha) {
            alpha = score;
        }
        if (alpha >= beta) {
            break;
        }
    }

    return alpha;
}

// ---- Root search with iterative deepening
tuple<int, int, int, int> find_move(int my_time) {
    search_start = chrono::high_resolution_clock::now();
    time_up = false;
    nodes_checked = 0;

    if (my_time > 30000) {
        time_limit_ms = 800;
    } else if (my_time > 10000) {
        time_limit_ms = 400;
    } else if (my_time > 3000) {
        time_limit_ms = 150;
    } else {
        time_limit_ms = 50;
    }

    ScoredMove moves[256];
    int move_cnt = gen_moves(moves, turn);

    int best_x1 = moves[0].x1;
    int best_y1 = moves[0].y1;
    int best_x2 = moves[0].x2;
    int best_y2 = moves[0].y2;

    for (int depth = 1; depth <= 30; depth++) {
        int best_score = -1e9;
        int best_idx = 0;

        for (int i = 0; i < move_cnt; i++) {
            make_move(moves[i].x1, moves[i].y1, moves[i].x2, moves[i].y2, turn);
            int score = -negamax(depth - 1, -1e9, 1e9, turn ^ 3);
            undo_move(turn);

            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }

            if (time_up) break;
        }

        if (time_up) break;

        best_x1 = moves[best_idx].x1;
        best_y1 = moves[best_idx].y1;
        best_x2 = moves[best_idx].x2;
        best_y2 = moves[best_idx].y2;

        if (best_idx != 0) {
            swap(moves[0], moves[best_idx]);
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

            auto [x1, y1, x2, y2] = find_move(t1);
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
