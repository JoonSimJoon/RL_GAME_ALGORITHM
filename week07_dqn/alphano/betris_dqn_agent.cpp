#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Betris Heuristic Agent
 *  ------------------------------------------------------------
 *  - Board size: 5x5 (1-indexed)
 *  - Bet strategy: based on board fullness
 *  - Placement: prioritize line completion
 * ============================================================
 */

int board[6][6];  // 1-indexed, 0=empty, 1=filled
int score = 0;
int coins = 100;
int turn_count = 0;

mt19937 rng(42);

// ----
// Count empty cells
// ----
int empty_cells() {
    int cnt = 0;
    for (int i = 1; i <= 5; i++)
        for (int j = 1; j <= 5; j++)
            if (board[i][j] == 0) cnt++;
    return cnt;
}

// ----
// Check if placing at (r,c) would complete a line
// ----
bool completes_line(int r, int c) {
    // Check row
    bool row_full = true;
    for (int j = 1; j <= 5; j++)
        if (j != c && board[r][j] == 0) { row_full = false; break; }
    // Check column
    bool col_full = true;
    for (int i = 1; i <= 5; i++)
        if (i != r && board[i][c] == 0) { col_full = false; break; }
    return row_full || col_full;
}

// ----
// Clear completed lines, return count
// ----
int clear_lines() {
    int cleared = 0;
    for (int i = 1; i <= 5; i++) {
        bool full = true;
        for (int j = 1; j <= 5; j++) if (board[i][j] == 0) { full = false; break; }
        if (full) { for (int j = 1; j <= 5; j++) board[i][j] = 0; cleared++; }
    }
    for (int j = 1; j <= 5; j++) {
        bool full = true;
        for (int i = 1; i <= 5; i++) if (board[i][j] == 0) { full = false; break; }
        if (full) { for (int i = 1; i <= 5; i++) board[i][j] = 0; cleared++; }
    }
    return cleared;
}

// ----
// Select bet amount based on board state
// ----
int select_bet() {
    int e = empty_cells();
    if (e > 15) return min(1, coins);
    if (e > 10) return min(2, coins);
    if (e > 5)  return min(5, coins);
    return min(10, coins);
}

// ----
// Select placement position: prioritize line completion
// ----
tuple<int, int> select_placement() {
    // Priority: complete a line
    for (int i = 1; i <= 5; i++)
        for (int j = 1; j <= 5; j++)
            if (board[i][j] == 0 && completes_line(i, j))
                return {i, j};
    // Fallback: random empty cell
    vector<pair<int,int>> empties;
    for (int i = 1; i <= 5; i++)
        for (int j = 1; j <= 5; j++)
            if (board[i][j] == 0) empties.push_back({i, j});
    if (!empties.empty()) {
        uniform_int_distribution<int> dist(0, empties.size() - 1);
        return empties[dist(rng)];
    }
    return {1, 1};
}

// ----
// Main event loop
// ----
int main() {
    string line;
    while (getline(cin, line)) {
        istringstream in(line);
        string cmd;
        in >> cmd;
        if (cmd == "READY") {
            string role;
            in >> role;
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2;
            in >> t1 >> t2;
            int bet = select_bet();
            auto [r, c] = select_placement();
            board[r][c] = 1;
            coins -= bet;
            int cleared = clear_lines();
            if (cleared > 0) score += cleared * 10 * bet;
            turn_count++;
            cout << "MOVE " << bet << ' ' << r << ' ' << c << ' ' << 0 << endl;
        }
        else if (cmd == "OPP") {
            // Read and ignore opponent action
            string rest;
            getline(in, rest);
        }
        else if (cmd == "FINISH") {
            break;
        }
    }
    return 0;
}
