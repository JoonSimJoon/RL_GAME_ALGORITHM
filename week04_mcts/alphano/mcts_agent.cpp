/*
 * ==== Ataxx MCTS Agent ====
 *
 * Uses Monte Carlo Tree Search with UCB1 for move selection.
 * - Selection: UCB1 (C=1.414)
 * - Expansion: Add one child node
 * - Simulation: Random rollout to terminal state
 * - Backpropagation: Update wins/visits up the tree
 * - Time management: 150ms if my_time > 1000ms, else 10ms
 */

#include <bits/stdc++.h>
using namespace std;

// ---- Global State ----
int board[8][8];
int turn;

const int dx[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
const int dy[8] = {-1, 0, 1, -1, 1, -1, 0, 1};

mt19937 rng(chrono::steady_clock::now().time_since_epoch().count());

// ---- Board Utilities ----

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

void sim_move(int b[8][8], int x1, int y1, int x2, int y2, int player) {
    if (x1 == -1) return; // pass

    int dist = max(abs(x2 - x1), abs(y2 - y1));
    if (dist == 2) {
        b[x1][y1] = 0; // jump
    }
    b[x2][y2] = player;

    // Convert adjacent opponent pieces
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (b[nx][ny] == (3 - player)) {
                b[nx][ny] = player;
            }
        }
    }
}

vector<tuple<int,int,int,int>> get_moves(int b[8][8], int player) {
    vector<tuple<int,int,int,int>> moves;

    for (int i = 1; i <= 7; i++) {
        for (int j = 1; j <= 7; j++) {
            if (b[i][j] != player) continue;

            // Try all 8 directions, distance 1 or 2
            for (int d = 0; d < 8; d++) {
                for (int dist = 1; dist <= 2; dist++) {
                    int ni = i + dx[d] * dist;
                    int nj = j + dy[d] * dist;

                    if (ni >= 1 && ni <= 7 && nj >= 1 && nj <= 7 && b[ni][nj] == 0) {
                        moves.push_back({i, j, ni, nj});
                    }
                }
            }
        }
    }

    if (moves.empty()) {
        moves.push_back({-1, -1, -1, -1}); // pass
    }

    return moves;
}

bool is_terminal(int b[8][8]) {
    int first = count_pieces(b, 1);
    int second = count_pieces(b, 2);

    if (first == 0 || second == 0) return true;
    if (first + second == 49) return true; // board full

    // Check if both players have no moves
    auto m1 = get_moves(b, 1);
    auto m2 = get_moves(b, 2);

    if (m1.size() == 1 && get<0>(m1[0]) == -1 &&
        m2.size() == 1 && get<0>(m2[0]) == -1) {
        return true;
    }

    return false;
}

// ---- MCTS Node ----

struct Node {
    int b[8][8];
    int player;
    Node* parent;
    tuple<int,int,int,int> move;
    vector<Node*> children;
    double wins;
    int visits;
    vector<tuple<int,int,int,int>> untried;

    Node(int board[8][8], int p, Node* par = nullptr, tuple<int,int,int,int> m = {0,0,0,0})
        : player(p), parent(par), move(m), wins(0), visits(0) {
        copy_board(board, b);
        untried = get_moves(b, player);
    }

    ~Node() {
        for (auto child : children) {
            delete child;
        }
    }
};

// ---- MCTS Functions ----

double ucb1(Node* node, double c = 1.414) {
    if (node->visits == 0) return 1e9;
    return (node->wins / node->visits) + c * sqrt(log(node->parent->visits) / node->visits);
}

Node* select(Node* node) {
    while (!node->children.empty()) {
        Node* best = nullptr;
        double best_ucb = -1e9;

        for (auto child : node->children) {
            double val = ucb1(child);
            if (val > best_ucb) {
                best_ucb = val;
                best = child;
            }
        }

        node = best;
    }
    return node;
}

Node* expand(Node* node) {
    if (node->untried.empty()) return node;

    // Pick random untried move
    int idx = rng() % node->untried.size();
    auto move = node->untried[idx];
    node->untried.erase(node->untried.begin() + idx);

    // Create child node
    int new_board[8][8];
    copy_board(node->b, new_board);
    sim_move(new_board, get<0>(move), get<1>(move), get<2>(move), get<3>(move), node->player);

    Node* child = new Node(new_board, 3 - node->player, node, move);
    node->children.push_back(child);

    return child;
}

double rollout(Node* node) {
    int sim_board[8][8];
    copy_board(node->b, sim_board);
    int sim_player = node->player;

    int max_moves = 200;
    for (int iter = 0; iter < max_moves && !is_terminal(sim_board); iter++) {
        auto moves = get_moves(sim_board, sim_player);
        auto move = moves[rng() % moves.size()];
        sim_move(sim_board, get<0>(move), get<1>(move), get<2>(move), get<3>(move), sim_player);
        sim_player = 3 - sim_player;
    }

    int first = count_pieces(sim_board, 1);
    int second = count_pieces(sim_board, 2);

    // Return result from perspective of player 1
    if (first > second) return 1.0;
    if (first < second) return 0.0;
    return 0.5;
}

void backprop(Node* node, double result) {
    while (node) {
        node->visits++;
        // result is from player 1's perspective
        // node->player is the player who just MOVED to create this state
        // so we need to flip perspective
        if (node->player == 2) {
            node->wins += result;
        } else {
            node->wins += (1.0 - result);
        }
        node = node->parent;
    }
}

tuple<int,int,int,int> mcts_search(int time_limit_ms) {
    Node* root = new Node(board, turn);

    auto start = chrono::steady_clock::now();
    int iterations = 0;

    while (true) {
        auto now = chrono::steady_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - start).count();
        if (elapsed >= time_limit_ms) break;

        // Selection
        Node* node = select(root);

        // Expansion
        if (!node->untried.empty() && node->visits > 0) {
            node = expand(node);
        }

        // Simulation
        double result = rollout(node);

        // Backpropagation
        backprop(node, result);

        iterations++;
    }

    // Select best move (most visited child)
    Node* best = nullptr;
    int best_visits = -1;

    for (auto child : root->children) {
        if (child->visits > best_visits) {
            best_visits = child->visits;
            best = child;
        }
    }

    tuple<int,int,int,int> best_move = {-1, -1, -1, -1};
    if (best) {
        best_move = best->move;
        cerr << "MCTS iterations: " << iterations
             << " | Best move visits: " << best->visits
             << " | Win rate: " << (best->wins / best->visits) << endl;
    }

    delete root;
    return best_move;
}

// ---- Move Selection ----

tuple<int,int,int,int> find_move(int my_time) {
    int time_limit = (my_time > 1000) ? 150 : 10;
    return mcts_search(time_limit);
}

// ---- Protocol Functions ----

void apply_my_move(int x1, int y1, int x2, int y2) {
    assert(turn == 1);
    if (x1 == -1) {
        turn = 2;
        return;
    }

    assert(board[x1][y1] == 1);
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(board[x2][y2] == 0);

    if (dist == 2) board[x1][y1] = 0;
    board[x2][y2] = 1;

    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && board[nx][ny] == 2) {
            board[nx][ny] = 1;
        }
    }

    turn = 2;
}

void apply_opp_move(int x1, int y1, int x2, int y2) {
    assert(turn == 2);
    if (x1 == -1) {
        turn = 1;
        return;
    }

    assert(board[x1][y1] == 2);
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(board[x2][y2] == 0);

    if (dist == 2) board[x1][y1] = 0;
    board[x2][y2] = 2;

    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7 && board[nx][ny] == 1) {
            board[nx][ny] = 2;
        }
    }

    turn = 1;
}

// ---- Main ----

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    memset(board, 0, sizeof(board));
    board[1][1] = board[7][7] = 1;
    board[1][7] = board[7][1] = 2;
    turn = 1;

    string cmd;
    while (cin >> cmd) {
        if (cmd == "TURN") {
            int my_time;
            cin >> my_time;

            auto [x1, y1, x2, y2] = find_move(my_time);
            cout << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
            apply_my_move(x1, y1, x2, y2);

        } else if (cmd == "OPP") {
            int x1, y1, x2, y2, t;
            cin >> x1 >> y1 >> x2 >> y2 >> t;
            apply_opp_move(x1, y1, x2, y2);

        } else if (cmd == "END") {
            int result;
            cin >> result;
            break;
        }
    }

    return 0;
}
