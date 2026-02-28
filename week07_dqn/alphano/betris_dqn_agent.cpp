#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Betris Improved Heuristic Agent
 *  ------------------------------------------------------------
 *  - Board size: 5x5 (1-indexed)
 *  - Better board evaluation with density and line-completion metrics
 *  - Smarter placement: tries all positions, picks best
 *  - Adaptive betting: based on predicted line clears
 * ============================================================
 */

int board[6][6];  // 1-indexed, 0=empty, 1=filled
int score = 0;
int coins = 100;

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// ----
// Count filled cells in a row
// ----
int row_fill_count(int r) {
    int count = 0;
    for (int c = 1; c <= 5; c++) {
        if (board[r][c] == 1) count++;
    }
    return count;
}

// ----
// Count filled cells in a column
// ----
int col_fill_count(int c) {
    int count = 0;
    for (int r = 1; r <= 5; r++) {
        if (board[r][c] == 1) count++;
    }
    return count;
}

// ----
// Count how many lines (rows + cols) can be cleared
// ----
int count_clearable_lines() {
    int lines = 0;
    // Check rows
    for (int r = 1; r <= 5; r++) {
        if (row_fill_count(r) == 5) lines++;
    }
    // Check columns
    for (int c = 1; c <= 5; c++) {
        if (col_fill_count(c) == 5) lines++;
    }
    return lines;
}

// ----
// Clear completed lines, return count
// ----
int clear_lines() {
    int cleared = 0;

    // Clear complete rows
    for (int r = 1; r <= 5; r++) {
        if (row_fill_count(r) == 5) {
            for (int c = 1; c <= 5; c++) {
                board[r][c] = 0;
            }
            cleared++;
        }
    }

    // Clear complete columns
    for (int c = 1; c <= 5; c++) {
        if (col_fill_count(c) == 5) {
            for (int r = 1; r <= 5; r++) {
                board[r][c] = 0;
            }
            cleared++;
        }
    }

    return cleared;
}

// ----
// Count isolated cells (cells with no adjacent filled cells)
// ----
int count_isolated_cells() {
    int isolated = 0;
    for (int r = 1; r <= 5; r++) {
        for (int c = 1; c <= 5; c++) {
            if (board[r][c] == 1) {
                bool has_neighbor = false;
                // Check 4 adjacent cells
                if (r > 1 && board[r-1][c] == 1) has_neighbor = true;
                if (r < 5 && board[r+1][c] == 1) has_neighbor = true;
                if (c > 1 && board[r][c-1] == 1) has_neighbor = true;
                if (c < 5 && board[r][c+1] == 1) has_neighbor = true;
                if (!has_neighbor) isolated++;
            }
        }
    }
    return isolated;
}

// ----
// Evaluate board quality: higher = better
// ----
int eval_board() {
    int val = 0;

    // Bonus for nearly complete rows
    for (int r = 1; r <= 5; r++) {
        int filled = row_fill_count(r);
        if (filled >= 4) val += filled * 10;
        if (filled == 5) val += 100;
    }

    // Bonus for nearly complete columns
    for (int c = 1; c <= 5; c++) {
        int filled = col_fill_count(c);
        if (filled >= 4) val += filled * 10;
        if (filled == 5) val += 100;
    }

    // Penalty for scattered pieces
    val -= count_isolated_cells() * 5;

    return val;
}

// ----
// Select best placement position
// ----
pair<int, int> select_placement() {
    int best_val = -1000000;
    int best_r = 1, best_c = 1;

    // Try all empty cells
    for (int r = 1; r <= 5; r++) {
        for (int c = 1; c <= 5; c++) {
            if (board[r][c] == 1) continue; // Skip filled cells

            // Simulate placing piece here
            board[r][c] = 1;

            int val = 0;

            // Check for immediate line clears
            int lines = count_clearable_lines();
            if (lines > 0) {
                // Huge bonus for clearing lines
                val = lines * 100;
            } else {
                // No clear: prefer positions that contribute to nearly-full lines
                int row_fill = row_fill_count(r);
                int col_fill = col_fill_count(c);
                val = row_fill * 10 + col_fill * 10;

                // Extra bonus for completing 4th cell in a line
                if (row_fill == 4) val += 20;
                if (col_fill == 4) val += 20;
            }

            // Undo placement
            board[r][c] = 0;

            if (val > best_val) {
                best_val = val;
                best_r = r;
                best_c = c;
            }
        }
    }

    return {best_r, best_c};
}

// ----
// Select bet amount based on expected outcome
// ----
int select_bet() {
    if (coins <= 0) return 0;

    // Simulate placement to predict line clears
    auto [r, c] = select_placement();
    board[r][c] = 1;
    int lines = count_clearable_lines();
    board[r][c] = 0;

    // Bet based on expected clears
    if (lines >= 2) {
        // Multi-line clear: bet big
        return min(coins, 20);
    } else if (lines == 1) {
        // Single line clear: moderate bet
        return min(coins, 10);
    } else {
        // No clear expected: minimal bet
        return min(coins, 1);
    }
}

// ----
// Main event loop
// ----
int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    string line;
    while (getline(cin, line)) {
        istringstream iss(line);
        string cmd;
        iss >> cmd;

        if (cmd == "READY") {
            string role;
            iss >> role;
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2;
            iss >> t1 >> t2;

            // Select bet and placement
            int bet = select_bet();
            auto [r, c] = select_placement();

            // Place piece and update state
            board[r][c] = 1;
            coins -= bet;

            // Clear lines and update score
            int cleared = clear_lines();
            score += cleared * 10 * bet;

            // Output move
            cout << "MOVE " << bet << " " << r << " " << c << " 0" << endl;
        }
        else if (cmd == "OPP") {
            // Ignore opponent moves for now
            // Could track opponent board state in future version
        }
        else if (cmd == "FINISH") {
            break;
        }
    }

    return 0;
}
