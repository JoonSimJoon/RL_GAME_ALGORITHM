#include <bits/stdc++.h>
using namespace std;

int board[8][8];
int turn;

const int dx[8] = {-1,-1,-1,0,0,1,1,1};
const int dy[8] = {-1,0,1,-1,1,-1,0,1};

int root_player;

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
    int dist = max(abs(x2 - x1), abs(y2 - y1));

    if (dist == 1) {
        // Split move
        b[x2][y2] = player;
    } else {
        // Jump move
        b[x1][y1] = 0;
        b[x2][y2] = player;
    }

    // Infect adjacent opponent pieces
    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (b[nx][ny] == (player ^ 3)) {
                b[nx][ny] = player;
            }
        }
    }
}

int gen_moves(int b[8][8], int player, int moves[][4]) {
    int cnt = 0;
    for (int x1 = 1; x1 <= 7; x1++) {
        for (int y1 = 1; y1 <= 7; y1++) {
            if (b[x1][y1] != player) continue;

            for (int x2 = x1 - 2; x2 <= x1 + 2; x2++) {
                if (x2 < 1 || x2 > 7) continue;

                for (int y2 = y1 - 2; y2 <= y1 + 2; y2++) {
                    if (y2 < 1 || y2 > 7) continue;
                    if (x2 == x1 && y2 == y1) continue;
                    if (b[x2][y2] != 0) continue;

                    moves[cnt][0] = x1;
                    moves[cnt][1] = y1;
                    moves[cnt][2] = x2;
                    moves[cnt][3] = y2;
                    cnt++;
                }
            }
        }
    }
    return cnt;
}

struct Node {
    int b[8][8];
    int player;
    Node* parent;
    int move[4];
    vector<Node*> children;
    double wins;
    int visits;
    int untried[200][4];
    int untried_cnt;

    Node(int board[8][8], int p, Node* par, int* m) {
        copy_board(board, b);
        player = p;
        parent = par;
        if (m) {
            move[0] = m[0];
            move[1] = m[1];
            move[2] = m[2];
            move[3] = m[3];
        }
        wins = 0.0;
        visits = 0;
        untried_cnt = gen_moves(b, player, untried);
    }

    ~Node() {
        for (Node* child : children) {
            delete child;
        }
    }

    bool is_terminal() {
        return untried_cnt == 0 && children.empty();
    }

    bool has_untried_moves() {
        return untried_cnt > 0;
    }
};

double ucb1(Node* n) {
    if (n->visits == 0) return 1e9;
    return (n->wins / n->visits) + 1.414 * sqrt(log(n->parent->visits) / n->visits);
}

Node* best_child_ucb(Node* node) {
    Node* best = nullptr;
    double best_score = -1e9;

    for (Node* child : node->children) {
        double score = ucb1(child);
        if (score > best_score) {
            best_score = score;
            best = child;
        }
    }

    return best;
}

Node* expand(Node* node) {
    int idx = rand() % node->untried_cnt;
    int* m = node->untried[idx];

    int new_board[8][8];
    copy_board(node->b, new_board);
    sim_move(new_board, m[0], m[1], m[2], m[3], node->player);

    Node* child = new Node(new_board, node->player ^ 3, node, m);
    node->children.push_back(child);

    // Remove from untried
    node->untried[idx][0] = node->untried[node->untried_cnt - 1][0];
    node->untried[idx][1] = node->untried[node->untried_cnt - 1][1];
    node->untried[idx][2] = node->untried[node->untried_cnt - 1][2];
    node->untried[idx][3] = node->untried[node->untried_cnt - 1][3];
    node->untried_cnt--;

    return child;
}

Node* tree_policy(Node* root) {
    Node* node = root;

    while (!node->is_terminal()) {
        if (node->has_untried_moves()) {
            return expand(node);
        } else {
            node = best_child_ucb(node);
        }
    }

    return node;
}

double rollout(Node* node) {
    int sim_board[8][8];
    copy_board(node->b, sim_board);
    int current_player = node->player;

    for (int step = 0; step < 200; step++) {
        int moves[200][4];
        int cnt = gen_moves(sim_board, current_player, moves);

        if (cnt == 0) {
            // Pass
            current_player ^= 3;
            int opp_cnt = gen_moves(sim_board, current_player, moves);
            if (opp_cnt == 0) {
                // Both pass - game over
                break;
            }
            continue;
        }

        // Random move
        int idx = rand() % cnt;
        sim_move(sim_board, moves[idx][0], moves[idx][1], moves[idx][2], moves[idx][3], current_player);
        current_player ^= 3;
    }

    // Evaluate from root player's perspective
    int my_count = count_pieces(sim_board, root_player);
    int opp_count = count_pieces(sim_board, root_player ^ 3);

    if (my_count > opp_count) return 1.0;
    if (my_count < opp_count) return 0.0;
    return 0.5;
}

void backprop(Node* node, double result) {
    while (node) {
        node->visits++;
        // Result is from root player's perspective
        // node->player is who plays NEXT from this state
        // If it's opponent's turn at this node, this is good for root player
        if (node->player != root_player) {
            node->wins += result;
        } else {
            node->wins += (1.0 - result);
        }
        node = node->parent;
    }
}

tuple<int,int,int,int> mcts_search(int time_limit_ms) {
    Node* root = new Node(board, turn, nullptr, nullptr);
    root_player = turn;

    auto start = chrono::high_resolution_clock::now();
    int iterations = 0;

    while (true) {
        auto now = chrono::high_resolution_clock::now();
        auto elapsed = chrono::duration_cast<chrono::milliseconds>(now - start).count();
        if (elapsed >= time_limit_ms) break;

        Node* node = tree_policy(root);
        double result = rollout(node);
        backprop(node, result);
        iterations++;
    }

    // Best child = most visited
    Node* best = nullptr;
    int max_visits = -1;
    for (Node* child : root->children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best = child;
        }
    }

    cerr << "MCTS: " << iterations << " iterations, best visits: " << max_visits << endl;

    tuple<int,int,int,int> result;
    if (best) {
        result = make_tuple(best->move[0], best->move[1], best->move[2], best->move[3]);
    } else {
        // No moves found - should not happen if called correctly
        result = make_tuple(-1, -1, -1, -1);
    }

    delete root;
    return result;
}

tuple<int,int,int,int> find_move(int my_time) {
    int time_limit;
    if (my_time > 30000) time_limit = 800;
    else if (my_time > 10000) time_limit = 400;
    else if (my_time > 3000) time_limit = 150;
    else time_limit = 50;

    return mcts_search(time_limit);
}

void apply_my_move(int x1, int y1, int x2, int y2) {
    if (x1 == -1) return; // pass

    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(board[x1][y1] == turn);
    assert(board[x2][y2] == 0);

    if (dist == 1) {
        board[x2][y2] = turn;
    } else {
        board[x1][y1] = 0;
        board[x2][y2] = turn;
    }

    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (board[nx][ny] == (turn ^ 3)) {
                board[nx][ny] = turn;
            }
        }
    }
}

void apply_opp_move(int x1, int y1, int x2, int y2) {
    if (x1 == -1) return; // pass

    int opp = turn ^ 3;
    int dist = max(abs(x2 - x1), abs(y2 - y1));
    assert(dist == 1 || dist == 2);
    assert(board[x1][y1] == opp);
    assert(board[x2][y2] == 0);

    if (dist == 1) {
        board[x2][y2] = opp;
    } else {
        board[x1][y1] = 0;
        board[x2][y2] = opp;
    }

    for (int d = 0; d < 8; d++) {
        int nx = x2 + dx[d];
        int ny = y2 + dy[d];
        if (nx >= 1 && nx <= 7 && ny >= 1 && ny <= 7) {
            if (board[nx][ny] == turn) {
                board[nx][ny] = opp;
            }
        }
    }
}

int main() {
    srand(time(0));

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
            turn = (role == "FIRST") ? 1 : 2;
            cout << "OK" << endl;
        } else if (cmd == "TURN") {
            int t1, t2;
            in >> t1 >> t2;

            auto [x1, y1, x2, y2] = find_move(t1);
            apply_my_move(x1, y1, x2, y2);
            cout << "MOVE " << x1 << " " << y1 << " " << x2 << " " << y2 << endl;
        } else if (cmd == "OPP") {
            int x1, y1, x2, y2, t;
            in >> x1 >> y1 >> x2 >> y2 >> t;
            apply_opp_move(x1, y1, x2, y2);
        } else if (cmd == "FINISH") {
            break;
        }
    }

    return 0;
}
