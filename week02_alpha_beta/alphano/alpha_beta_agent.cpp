/*
 * ============================================================
 * Ataxx Alpha-Beta Agent
 * ============================================================
 * Strategy: Negamax with Alpha-Beta Pruning (depth=4)
 * Evaluation: Piece count + Position weights + Mobility
 * Game phase adaptive weighting
 */

#include <bits/stdc++.h>
using namespace std;

// ------------------------------------------------------------
// Global state
// ------------------------------------------------------------

int board[8][8];  // 1-indexed, 0=empty, 1=FIRST, 2=SECOND
int turn;         // 1 or 2
string role;      // "FIRST" or "SECOND"

// ------------------------------------------------------------
// Constants
// ------------------------------------------------------------

const int dx[8] = {-1, -1, -1,  0,  0,  1,  1,  1};
const int dy[8] = {-1,  0,  1, -1,  1, -1,  0,  1};

const int MAX_DEPTH = 4;
const int INF = 1e9;

// Position weights (1-indexed, index 0 unused)
const int POS_WEIGHT[8][8] = {
    {0,   0,   0,   0,   0,   0,   0,   0},  // row 0 unused
    {0, 100, -20,  10,   5,  10, -20, 100},  // row 1
    {0, -20, -40,  -5,  -5,  -5, -40, -20},  // row 2
    {0,  10,  -5,  10,   5,  10,  -5,  10},  // row 3
    {0,   5,  -5,   5,   0,   5,  -5,   5},  // row 4
    {0,  10,  -5,  10,   5,  10,  -5,  10},  // row 5
    {0, -20, -40,  -5,  -5,  -5, -40, -20},  // row 6
    {0, 100, -20,  10,   5,  10, -20, 100},  // row 7
};

// ------------------------------------------------------------
// Board utilities
// ------------------------------------------------------------

void copy_board(int src[8][8], int dst[8][8]) {
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            dst[i][j] = src[i][j];
        }
    }
}

int count_pieces(int b[8][8], int player) {
    int cnt = 0;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (b[i][j] == player) cnt++;
        }
    }
    return cnt;
}

void simulate_move(int b[8][8], int x1, int y1, int x2, int y2, int player) {
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(b[x1][y1] == player);
    assert(b[x2][y2] == 0);

    if (dist == 1) {
        // Split: copy
        b[x2][y2] = player;
    } else {
        // Jump: move
        b[x1][y1] = 0;
        b[x2][y2] = player;
    }

    // Infect adjacent opponent pieces
    int opp = player ^ 3;
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && b[nx][ny] == opp) {
            b[nx][ny] = player;
        }
    }
}

vector<tuple<int,int,int,int>> find_all_moves(int b[8][8], int player) {
    vector<tuple<int,int,int,int>> moves;

    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (b[x1][y1] != player) continue;

            // Check all positions within distance 2
            for (int dx = -2; dx <= 2; dx++) {
                for (int dy = -2; dy <= 2; dy++) {
                    if (dx == 0 && dy == 0) continue;

                    int x2 = x1 + dx;
                    int y2 = y1 + dy;

                    if (x2 < 1 || x2 > 7 || y2 < 1 || y2 > 7) continue;
                    if (b[x2][y2] != 0) continue;

                    int dist = max(abs(dx), abs(dy));
                    if (dist == 1 || dist == 2) {
                        moves.push_back(make_tuple(x1, y1, x2, y2));
                    }
                }
            }
        }
    }

    return moves;
}

// ------------------------------------------------------------
// Evaluation function
// ------------------------------------------------------------

int evaluate(int b[8][8], int player) {
    int opp = player ^ 3;

    // Check if game over
    bool player_has_moves = !find_all_moves(b, player).empty();
    bool opp_has_moves = !find_all_moves(b, opp).empty();

    if (!player_has_moves && !opp_has_moves) {
        int my_cnt = count_pieces(b, player);
        int opp_cnt = count_pieces(b, opp);
        if (my_cnt > opp_cnt) return 10000;
        if (my_cnt < opp_cnt) return -10000;
        return 0;
    }

    // 1. Piece count
    int my_pieces = count_pieces(b, player);
    int opp_pieces = count_pieces(b, opp);
    int piece_score = my_pieces - opp_pieces;

    // 2. Position value
    int my_position = 0;
    int opp_position = 0;
    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (b[i][j] == player) {
                my_position += POS_WEIGHT[i][j];
            } else if (b[i][j] == opp) {
                opp_position += POS_WEIGHT[i][j];
            }
        }
    }
    int position_score = my_position - opp_position;

    // 3. Mobility (number of legal moves)
    int my_mobility = find_all_moves(b, player).size();
    int opp_mobility = find_all_moves(b, opp).size();
    int mobility_score = my_mobility - opp_mobility;

    // 4. Game phase detection: adjust weights based on total pieces
    int total_pieces = my_pieces + opp_pieces;
    int weight_piece, weight_position, weight_mobility;

    if (total_pieces < 20) {
        // Early game: mobility and position important
        weight_piece = 1;
        weight_position = 3;
        weight_mobility = 5;
    } else if (total_pieces < 35) {
        // Mid game: balanced
        weight_piece = 2;
        weight_position = 2;
        weight_mobility = 3;
    } else {
        // Late game: piece count most important
        weight_piece = 5;
        weight_position = 1;
        weight_mobility = 1;
    }

    return weight_piece * piece_score +
           weight_position * position_score +
           weight_mobility * mobility_score;
}

// ------------------------------------------------------------
// Negamax with Alpha-Beta pruning
// ------------------------------------------------------------

int negamax(int b[8][8], int depth, int alpha, int beta, int player) {
    // Base case: depth 0 or game over
    if (depth == 0) {
        return evaluate(b, player);
    }

    vector<tuple<int,int,int,int>> moves = find_all_moves(b, player);

    // No moves available: pass to opponent
    if (moves.empty()) {
        int opp = player ^ 3;
        vector<tuple<int,int,int,int>> opp_moves = find_all_moves(b, opp);
        if (opp_moves.empty()) {
            // Game over
            return evaluate(b, player);
        }
        // Pass turn
        return -negamax(b, depth - 1, -beta, -alpha, opp);
    }

    // Move ordering: sort by destination position weight (descending)
    sort(moves.begin(), moves.end(), [](const auto& a, const auto& b) {
        int x2a = get<2>(a), y2a = get<3>(a);
        int x2b = get<2>(b), y2b = get<3>(b);
        return POS_WEIGHT[x2a][y2a] > POS_WEIGHT[x2b][y2b];
    });

    // Search all moves
    for (const auto& mv : moves) {
        int x1 = get<0>(mv), y1 = get<1>(mv);
        int x2 = get<2>(mv), y2 = get<3>(mv);

        // Simulate move on copy
        int tmp[8][8];
        copy_board(b, tmp);
        simulate_move(tmp, x1, y1, x2, y2, player);

        // Recursive call with negamax
        int opp = player ^ 3;
        int score = -negamax(tmp, depth - 1, -beta, -alpha, opp);

        // Update alpha
        if (score > alpha) {
            alpha = score;
        }

        // Beta cutoff: pruning!
        if (alpha >= beta) {
            break;
        }
    }

    return alpha;
}

// ------------------------------------------------------------
// Find best move (root search)
// ------------------------------------------------------------

tuple<int,int,int,int> find_move() {
    vector<tuple<int,int,int,int>> moves = find_all_moves(board, turn);

    if (moves.empty()) {
        // No moves: should not happen in normal game
        return make_tuple(0, 0, 0, 0);
    }

    // Move ordering: sort by destination position weight
    sort(moves.begin(), moves.end(), [](const auto& a, const auto& b) {
        int x2a = get<2>(a), y2a = get<3>(a);
        int x2b = get<2>(b), y2b = get<3>(b);
        return POS_WEIGHT[x2a][y2a] > POS_WEIGHT[x2b][y2b];
    });

    int best_score = -INF;
    tuple<int,int,int,int> best_move = moves[0];

    for (const auto& mv : moves) {
        int x1 = get<0>(mv), y1 = get<1>(mv);
        int x2 = get<2>(mv), y2 = get<3>(mv);

        // Simulate move
        int tmp[8][8];
        copy_board(board, tmp);
        simulate_move(tmp, x1, y1, x2, y2, turn);

        // Search with negamax
        int opp = turn ^ 3;
        int score = -negamax(tmp, MAX_DEPTH - 1, -INF, INF, opp);

        if (score > best_score) {
            best_score = score;
            best_move = mv;
        }
    }

    return best_move;
}

// ------------------------------------------------------------
// Move application
// ------------------------------------------------------------

void apply_my_move(int x1, int y1, int x2, int y2) {
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(board[x1][y1] == turn);
    assert(board[x2][y2] == 0);

    if (dist == 1) {
        // Split
        board[x2][y2] = turn;
    } else {
        // Jump
        board[x1][y1] = 0;
        board[x2][y2] = turn;
    }

    // Infect adjacent opponent pieces
    int opp = turn ^ 3;
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && board[nx][ny] == opp) {
            board[nx][ny] = turn;
        }
    }
}

void apply_opp_move(int x1, int y1, int x2, int y2) {
    int opp = turn ^ 3;
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(board[x1][y1] == opp);
    assert(board[x2][y2] == 0);

    if (dist == 1) {
        // Split
        board[x2][y2] = opp;
    } else {
        // Jump
        board[x1][y1] = 0;
        board[x2][y2] = opp;
    }

    // Infect adjacent my pieces
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && board[nx][ny] == turn) {
            board[nx][ny] = opp;
        }
    }
}

// ------------------------------------------------------------
// Main
// ------------------------------------------------------------

int main() {
    // Initialize board: corners
    board[1][1] = board[7][7] = 1;  // FIRST
    board[1][7] = board[7][1] = 2;  // SECOND

    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        string cmd;
        in >> cmd;

        if (cmd == "READY") {
            in >> role;
            turn = (role == "FIRST" ? 1 : 2);
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2;
            in >> t1 >> t2;

            auto [x1, y1, x2, y2] = find_move();
            apply_my_move(x1, y1, x2, y2);

            cout << "MOVE " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
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
