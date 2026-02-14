#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Ataxx Random Agent
 *  ------------------------------------------------------------
 *  - Board size: 7x7 (1-indexed)
 *  - Piece values: 0 = empty, 1 = FIRST(O), 2 = SECOND(X)
 *  - This agent plays a uniformly random legal move each turn.
 * ============================================================
 */

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

int board[8][8]; // 1-based index board
int turn;        // 1 = FIRST, 2 = SECOND

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// ------------------------------------------------------------
// Collect all legal moves for the current player.
// Each move is (x1, y1, x2, y2).
// ------------------------------------------------------------
vector<tuple<int,int,int,int>> find_all_moves() {
    vector<tuple<int,int,int,int>> moves;
    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (board[x1][y1] != turn) continue;
            for (int x2 = x1 - 2; x2 <= x1 + 2; x2++) {
                if (x2 < 1 || x2 > 7) continue;
                for (int y2 = y1 - 2; y2 <= y1 + 2; y2++) {
                    if (y2 < 1 || y2 > 7) continue;
                    if (x2 == x1 && y2 == y1) continue;
                    if (board[x2][y2] != 0) continue;
                    moves.emplace_back(x1, y1, x2, y2);
                }
            }
        }
    }
    return moves;
}

// ------------------------------------------------------------
// Pick a uniformly random legal move.
// Returns (-1, -1, -1, -1) if no move exists.
// ------------------------------------------------------------
tuple<int, int, int, int> find_move() {
    auto moves = find_all_moves();
    if (moves.empty()) return {-1, -1, -1, -1};
    uniform_int_distribution<int> dist(0, (int)moves.size() - 1);
    return moves[dist(rng)];
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
