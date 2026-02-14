#include <bits/stdc++.h>
using namespace std;

/*
 * ============================================================
 *  Catch the Mouse - Heuristic Agent
 *  ------------------------------------------------------------
 *  - Board size: 7x11 (1-indexed)
 *  - FIRST = Cat (minimize distance to mouse)
 *  - SECOND = Mouse (maximize distance from cat)
 *  - 4-directional movement
 * ============================================================
 */

const int DX[4] = {0, 0, -1, 1};
const int DY[4] = {-1, 1, 0, 0};

int W = 7, H = 11;      // board dimensions (width x height)
int turn;                // 1 = FIRST (cat), 2 = SECOND (mouse)
int my_x, my_y;         // my position (1-indexed)
int opp_x, opp_y;       // opponent position (1-indexed)

// ----
// Check if position is valid
// ----
bool valid(int x, int y) {
    return x >= 1 && x <= W && y >= 1 && y <= H;
}

// ----
// Manhattan distance
// ----
int dist(int x1, int y1, int x2, int y2) {
    return abs(x1 - x2) + abs(y1 - y2);
}

// ----
// Find best move: cat minimizes distance, mouse maximizes distance
// ----
tuple<int, int> find_move() {
    int best_x = my_x, best_y = my_y;
    int best_d = (turn == 1) ? INT_MAX : INT_MIN;

    for (int i = 0; i < 4; i++) {
        int nx = my_x + DX[i];
        int ny = my_y + DY[i];
        if (!valid(nx, ny)) continue;
        int d = dist(nx, ny, opp_x, opp_y);
        if (turn == 1) {  // cat: minimize
            if (d < best_d) { best_d = d; best_x = nx; best_y = ny; }
        } else {  // mouse: maximize
            if (d > best_d) { best_d = d; best_x = nx; best_y = ny; }
        }
    }
    return {best_x, best_y};
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
            turn = (role == "FIRST" ? 1 : 2);
            if (turn == 1) {  // cat starts left-center
                my_x = 1; my_y = H / 2 + 1;
                opp_x = W; opp_y = H / 2 + 1;
            } else {  // mouse starts right-center
                my_x = W; my_y = H / 2 + 1;
                opp_x = 1; opp_y = H / 2 + 1;
            }
            cout << "OK" << endl;
        }
        else if (cmd == "TURN") {
            int t1, t2;
            in >> t1 >> t2;
            auto [x, y] = find_move();
            my_x = x; my_y = y;
            cout << "MOVE " << x << ' ' << y << endl;
        }
        else if (cmd == "OPP") {
            int x, y, t;
            in >> x >> y >> t;
            opp_x = x; opp_y = y;
        }
        else if (cmd == "FINISH") {
            break;
        }
    }
    return 0;
}
