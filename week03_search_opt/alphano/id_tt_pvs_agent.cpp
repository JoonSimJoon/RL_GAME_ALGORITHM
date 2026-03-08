/*
 * ==== Ataxx ID-TT-PVS Agent ====
 *
 * Iterative Deepening + Transposition Table + Principal Variation Search
 *
 * Strategy:
 * 1. Zobrist Hashing for board positions
 * 2. Transposition Table (TT) with PV/CUT/ALL flags
 * 3. Principal Variation Search (PVS) with null window
 * 4. Iterative Deepening with time management
 * 5. Move ordering using TT best move
 * 6. Evaluation: piece_diff + mobility * 0.1
 */

#include <iostream>
#include <vector>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <chrono>
#include <cassert>
#include <cstdint>
using namespace std;

// ----------------------------------------------------------------------------
// Global State
// ----------------------------------------------------------------------------

int board[8][8];
int turn;

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

const int FIRST = 1;
const int SECOND = 2;
const double INF = 1e9;

// TT Flags
const int PV_NODE = 0;   // Exact value
const int CUT_NODE = 1;  // Lower bound (beta cutoff)
const int ALL_NODE = 2;  // Upper bound (alpha cutoff)

// ----------------------------------------------------------------------------
// Zobrist Hashing
// ----------------------------------------------------------------------------

uint64_t zobrist[8][8][3];  // [row][col][piece_type] (0=empty, 1=FIRST, 2=SECOND)
uint64_t board_hash;

uint64_t xorshift64(uint64_t x) {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    return x;
}

void init_zobrist() {
    uint64_t seed = 987654321;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            for (int piece = 0; piece < 3; piece++) {
                seed = xorshift64(seed);
                zobrist[i][j][piece] = seed;
            }
        }
    }
}

uint64_t compute_hash() {
    uint64_t h = 0;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            h ^= zobrist[i][j][board[i][j]];
        }
    }
    return h;
}

// ----------------------------------------------------------------------------
// Transposition Table
// ----------------------------------------------------------------------------

struct TTEntry {
    tuple<int,int,int,int> best_move;
    int flag;
    int depth;
    double value;

    TTEntry() : best_move(-1,-1,-1,-1), flag(0), depth(0), value(0.0) {}
    TTEntry(tuple<int,int,int,int> bm, int f, int d, double v)
        : best_move(bm), flag(f), depth(d), value(v) {}
};

unordered_map<uint64_t, TTEntry> tt;

// ----------------------------------------------------------------------------
// Move History (for undo)
// ----------------------------------------------------------------------------

// Each entry: (move, distance, infected_cells)
vector<tuple<tuple<int,int,int,int>, int, vector<pair<int,int>>>> move_history;

// ----------------------------------------------------------------------------
// Search State
// ----------------------------------------------------------------------------

int current_player;  // Player whose turn it is during search

// ----------------------------------------------------------------------------
// Move Generation
// ----------------------------------------------------------------------------

bool is_valid(int x, int y) {
    if (x < 1 || x > 7 || y < 1 || y > 7) return false;
    if (x == 4 && y == 4) return false;  // Wall at center
    return true;
}

vector<tuple<int,int,int,int>> find_all_moves(int player) {
    vector<tuple<int,int,int,int>> moves;

    for (int x = 1; x <= 7; x++) {
        for (int y = 1; y <= 7; y++) {
            if (board[x][y] != player) continue;

            // Try all destinations within distance 2
            for (int d_x = -2; d_x <= 2; d_x++) {
                for (int d_y = -2; d_y <= 2; d_y++) {
                    if (d_x == 0 && d_y == 0) continue;
                    int dist = max(abs(d_x), abs(d_y));
                    if (dist > 2) continue;

                    int nx = x + d_x;
                    int ny = y + d_y;

                    if (is_valid(nx, ny) && board[nx][ny] == 0) {
                        moves.push_back({x, y, nx, ny});
                    }
                }
            }
        }
    }

    return moves;
}

// ----------------------------------------------------------------------------
// Make/Undo Move (for search)
// ----------------------------------------------------------------------------

void make_move(tuple<int,int,int,int> move) {
    auto [x1, y1, x2, y2] = move;
    int player = current_player;
    int opp = (player == FIRST) ? SECOND : FIRST;

    int dist = max(abs(x2 - x1), abs(y2 - y1));

    // Update hash: place piece at destination
    board_hash ^= zobrist[x2][y2][0];
    board_hash ^= zobrist[x2][y2][player];
    board[x2][y2] = player;

    // If jump (distance 2), remove source
    if (dist == 2) {
        board_hash ^= zobrist[x1][y1][player];
        board_hash ^= zobrist[x1][y1][0];
        board[x1][y1] = 0;
    }

    // Infect adjacent opponent pieces
    vector<pair<int,int>> infected;
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (is_valid(nx, ny) && board[nx][ny] == opp) {
            infected.push_back({nx, ny});
            board_hash ^= zobrist[nx][ny][opp];
            board_hash ^= zobrist[nx][ny][player];
            board[nx][ny] = player;
        }
    }

    move_history.push_back({move, dist, infected});
    current_player = opp;
}

void undo_move() {
    auto [move, dist, infected] = move_history.back();
    move_history.pop_back();
    auto [x1, y1, x2, y2] = move;

    // Switch back to previous player
    current_player = (current_player == FIRST) ? SECOND : FIRST;
    int player = current_player;
    int opp = (player == FIRST) ? SECOND : FIRST;

    // Remove piece from destination
    board_hash ^= zobrist[x2][y2][player];
    board_hash ^= zobrist[x2][y2][0];
    board[x2][y2] = 0;

    // If jump, restore source
    if (dist == 2) {
        board_hash ^= zobrist[x1][y1][0];
        board_hash ^= zobrist[x1][y1][player];
        board[x1][y1] = player;
    }

    // Restore infected pieces
    for (auto [nx, ny] : infected) {
        board_hash ^= zobrist[nx][ny][player];
        board_hash ^= zobrist[nx][ny][opp];
        board[nx][ny] = opp;
    }
}

// ----------------------------------------------------------------------------
// Evaluation
// ----------------------------------------------------------------------------

pair<int,int> count_pieces() {
    int first_count = 0, second_count = 0;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (board[i][j] == FIRST) first_count++;
            else if (board[i][j] == SECOND) second_count++;
        }
    }
    return {first_count, second_count};
}

double evaluate() {
    auto [first_count, second_count] = count_pieces();
    int mobility = find_all_moves(current_player).size();

    int piece_diff;
    if (current_player == FIRST) {
        piece_diff = first_count - second_count;
    } else {
        piece_diff = second_count - first_count;
    }

    return piece_diff + mobility * 0.1;
}

bool is_terminal() {
    auto [first_count, second_count] = count_pieces();
    if (first_count == 0 || second_count == 0) return true;
    if (first_count + second_count == 48) return true;
    return false;
}

// ----------------------------------------------------------------------------
// Principal Variation Search
// ----------------------------------------------------------------------------

pair<double, tuple<int,int,int,int>> pvs(
    int depth, double alpha, double beta,
    chrono::steady_clock::time_point start_time, int time_limit) {

    // Time check
    auto now = chrono::steady_clock::now();
    auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - start_time).count();
    if (elapsed > time_limit * 0.95) {
        return {evaluate(), {-1,-1,-1,-1}};
    }

    double alpha_original = alpha;

    // TT lookup
    tuple<int,int,int,int> tt_move = {-1,-1,-1,-1};
    if (tt.count(board_hash)) {
        tt_move = tt[board_hash].best_move;
    }

    // Terminal or depth 0
    if (depth == 0 || is_terminal()) {
        return {evaluate(), {-1,-1,-1,-1}};
    }

    auto moves = find_all_moves(current_player);

    // No moves (pass)
    if (moves.empty()) {
        return {evaluate(), {-1,-1,-1,-1}};
    }

    // Move ordering: TT move first
    auto it = find(moves.begin(), moves.end(), tt_move);
    if (it != moves.end()) {
        moves.erase(it);
        moves.insert(moves.begin(), tt_move);
    }

    double best_value = -INF;
    tuple<int,int,int,int> best_move = {-1,-1,-1,-1};

    for (size_t i = 0; i < moves.size(); i++) {
        make_move(moves[i]);

        double value;
        if (i == 0) {
            // Full window search
            auto [v, _] = pvs(depth - 1, -beta, -alpha, start_time, time_limit);
            value = -v;
        } else {
            // Null window search
            auto [v, _] = pvs(depth - 1, -alpha - 1, -alpha, start_time, time_limit);
            value = -v;

            // Re-search if necessary
            if (alpha < value && value < beta) {
                auto [v2, _] = pvs(depth - 1, -beta, -value, start_time, time_limit);
                value = -v2;
            }
        }

        undo_move();

        if (value > best_value) {
            best_value = value;
            best_move = moves[i];
        }

        alpha = max(alpha, value);
        if (alpha >= beta) break;  // Beta cutoff
    }

    // Store in TT
    int flag;
    if (best_value <= alpha_original) flag = ALL_NODE;
    else if (best_value >= beta) flag = CUT_NODE;
    else flag = PV_NODE;

    if (!tt.count(board_hash) || tt[board_hash].depth <= depth) {
        tt[board_hash] = TTEntry(best_move, flag, depth, best_value);
    }

    return {best_value, best_move};
}

// ----------------------------------------------------------------------------
// Iterative Deepening
// ----------------------------------------------------------------------------

tuple<int,int,int,int> iterative_deepening(int time_limit) {
    auto start_time = chrono::steady_clock::now();
    tuple<int,int,int,int> best_move = {-1,-1,-1,-1};

    for (int depth = 1; depth < 50; depth++) {
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - start_time).count();
        if (elapsed > time_limit * 0.85) break;

        auto [value, move] = pvs(depth, -INF, INF, start_time, time_limit);

        if (get<0>(move) != -1) {
            best_move = move;
        } else {
            break;
        }
    }

    return best_move;
}

// ----------------------------------------------------------------------------
// Time Management
// ----------------------------------------------------------------------------

int calculate_time_limit(int my_time) {
    if (my_time > 60000) return 50;
    else if (my_time > 20000) return 150;
    else return 10;
}

// ----------------------------------------------------------------------------
// Protocol Functions
// ----------------------------------------------------------------------------

void apply_my_move(int x1, int y1, int x2, int y2) {
    assert(x1 >= 1 && x1 <= 7 && y1 >= 1 && y1 <= 7);
    assert(x2 >= 1 && x2 <= 7 && y2 >= 1 && y2 <= 7);
    assert(board[x1][y1] == turn);
    assert(board[x2][y2] == 0);

    int dist = max(abs(x2 - x1), abs(y2 - y1));
    board[x2][y2] = turn;

    if (dist == 2) {
        board[x1][y1] = 0;
    }

    int opp = turn ^ 3;
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (is_valid(nx, ny) && board[nx][ny] == opp) {
            board[nx][ny] = turn;
        }
    }
}

void apply_opp_move(int x1, int y1, int x2, int y2) {
    assert(x1 >= 1 && x1 <= 7 && y1 >= 1 && y1 <= 7);
    assert(x2 >= 1 && x2 <= 7 && y2 >= 1 && y2 <= 7);

    int opp = turn ^ 3;
    assert(board[x1][y1] == opp);
    assert(board[x2][y2] == 0);

    int dist = max(abs(x2 - x1), abs(y2 - y1));
    board[x2][y2] = opp;

    if (dist == 2) {
        board[x1][y1] = 0;
    }

    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (is_valid(nx, ny) && board[nx][ny] == turn) {
            board[nx][ny] = opp;
        }
    }
}

tuple<int,int,int,int> find_move() {
    // Synchronize current_player with global turn for search
    current_player = turn;
    board_hash = compute_hash();
    move_history.clear();

    // Time is read from context (global variable set by main)
    int time_limit = calculate_time_limit(100000);  // Default
    return iterative_deepening(time_limit);
}

// ----------------------------------------------------------------------------
// Main
// ----------------------------------------------------------------------------

int main() {
    init_zobrist();

    // Initialize board
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            board[i][j] = 0;
        }
    }

    string cmd;
    while (cin >> cmd) {
        if (cmd == "READY") {
            string position;
            cin >> position;

            // Initial setup
            board[1][1] = FIRST;
            board[7][7] = FIRST;
            board[1][7] = SECOND;
            board[7][1] = SECOND;

            if (position == "FIRST") {
                turn = FIRST;
            } else {
                turn = SECOND;
            }

            cout << "OK" << endl;

        } else if (cmd == "TURN") {
            int my_time, opp_time;
            cin >> my_time >> opp_time;

            // Update search state
            current_player = turn;
            board_hash = compute_hash();
            move_history.clear();

            int time_limit = calculate_time_limit(my_time);
            auto best_move = iterative_deepening(time_limit);

            if (get<0>(best_move) != -1) {
                auto [x1, y1, x2, y2] = best_move;
                cout << "MOVE " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
                apply_my_move(x1, y1, x2, y2);
            } else {
                cout << "PASS" << endl;
            }

        } else if (cmd == "OPP") {
            int x1, y1, x2, y2, t;
            cin >> x1 >> y1 >> x2 >> y2 >> t;
            apply_opp_move(x1, y1, x2, y2);

        } else if (cmd == "FINISH") {
            break;
        }
    }

    return 0;
}
