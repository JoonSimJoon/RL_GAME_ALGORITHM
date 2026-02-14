/*
틱택토 MCTS 예제

간단한 게임으로 MCTS를 이해하기 위한 예제입니다.
ATAXX MCTS를 구현하기 전에 이 예제를 먼저 공부하세요.
*/

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <memory>
#include <limits>

using namespace std;

// Random number generator
random_device rd;
mt19937 gen(rd());

// Move type: (r, c)
struct Move {
    int r, c;
    Move(int r_ = 0, int c_ = 0) : r(r_), c(c_) {}
};

class TicTacToe {
public:
    vector<vector<int>> board;
    int current_player;  // 1 = X, 2 = O

    TicTacToe() : board(3, vector<int>(3, 0)), current_player(1) {}

    TicTacToe copy() const {
        TicTacToe new_game;
        new_game.board = board;
        new_game.current_player = current_player;
        return new_game;
    }

    vector<Move> get_legal_moves() const {
        vector<Move> moves;
        for (int r = 0; r < 3; r++) {
            for (int c = 0; c < 3; c++) {
                if (board[r][c] == 0) {
                    moves.push_back(Move(r, c));
                }
            }
        }
        return moves;
    }

    TicTacToe apply_move(const Move& move) const {
        TicTacToe new_game = copy();
        new_game.board[move.r][move.c] = new_game.current_player;
        new_game.current_player = 3 - new_game.current_player;
        return new_game;
    }

    bool is_terminal() const {
        // 승리 조건 확인
        if (check_winner() != 0) return true;

        // 빈 칸이 없으면 무승부
        if (get_legal_moves().empty()) return true;

        return false;
    }

    double get_result() const {
        int winner = check_winner();

        if (winner == 0) return 0.5;  // 무승부

        // 현재 플레이어가 승자인가?
        return (winner == current_player) ? 1.0 : 0.0;
    }

    int check_winner() const {
        // 가로 확인
        for (int r = 0; r < 3; r++) {
            if (board[r][0] == board[r][1] && board[r][1] == board[r][2] && board[r][0] != 0) {
                return board[r][0];
            }
        }

        // 세로 확인
        for (int c = 0; c < 3; c++) {
            if (board[0][c] == board[1][c] && board[1][c] == board[2][c] && board[0][c] != 0) {
                return board[0][c];
            }
        }

        // 대각선 확인
        if (board[0][0] == board[1][1] && board[1][1] == board[2][2] && board[0][0] != 0) {
            return board[0][0];
        }
        if (board[0][2] == board[1][1] && board[1][1] == board[2][0] && board[0][2] != 0) {
            return board[0][2];
        }

        return 0;  // 승자 없음
    }

    void print() const {
        vector<char> symbols = {'.', 'X', 'O'};
        for (const auto& row : board) {
            for (int cell : row) {
                cout << symbols[cell] << " ";
            }
            cout << endl;
        }
    }
};

class MCTSNode {
public:
    TicTacToe state;
    MCTSNode* parent;
    Move move;
    vector<unique_ptr<MCTSNode>> children;
    double wins;
    int visits;
    vector<Move> untried_moves;

    MCTSNode(const TicTacToe& state_, MCTSNode* parent_ = nullptr, const Move& move_ = Move())
        : state(state_), parent(parent_), move(move_), wins(0.0), visits(0) {
        untried_moves = state.get_legal_moves();
    }

    bool is_fully_expanded() const {
        return untried_moves.empty();
    }

    bool is_terminal() const {
        return state.is_terminal();
    }

    double ucb1(double c = 1.414) const {
        /*
        UCB1 값 계산

        UCB1 = exploitation + exploration
             = w/n + c * sqrt(ln(N)/n)

        Args:
            c: 탐험 상수 (exploration constant)

        Returns:
            UCB1 값
        */
        if (visits == 0) {
            return numeric_limits<double>::infinity();  // 미방문 노드는 최우선 선택
        }

        double exploitation = wins / visits;
        double exploration = c * sqrt(log(parent->visits) / visits);

        return exploitation + exploration;
    }

    MCTSNode* select_child() {
        /*
        UCB1 값이 가장 큰 자식 선택

        Returns:
            최선의 자식 노드
        */
        MCTSNode* best = nullptr;
        double best_ucb = -1.0;

        for (auto& child : children) {
            double ucb = child->ucb1();
            if (ucb > best_ucb) {
                best_ucb = ucb;
                best = child.get();
            }
        }

        return best;
    }

    MCTSNode* expand() {
        /*
        미탐색 수 중 하나를 선택해서 자식 노드 추가

        Returns:
            새로 추가된 자식 노드
        */
        // 미탐색 수 중 하나 선택
        Move move = untried_moves.back();
        untried_moves.pop_back();

        // 수를 적용한 새 상태 생성
        TicTacToe next_state = state.apply_move(move);

        // 자식 노드 생성
        children.push_back(make_unique<MCTSNode>(next_state, this, move));

        return children.back().get();
    }

    double rollout() {
        /*
        현재 상태에서 게임 끝까지 랜덤 플레이

        Returns:
            게임 결과 (현재 플레이어 관점)
        */
        TicTacToe sim_state = state.copy();

        // 게임이 끝날 때까지 랜덤으로 진행
        while (!sim_state.is_terminal()) {
            auto legal_moves = sim_state.get_legal_moves();
            uniform_int_distribution<> dis(0, legal_moves.size() - 1);
            Move move = legal_moves[dis(gen)];
            sim_state = sim_state.apply_move(move);
        }

        // 결과를 원래 플레이어 관점으로 변환
        int winner = sim_state.check_winner();

        if (winner == 0) return 0.5;  // 무승부

        // 원래 플레이어가 승자인가?
        return (winner == state.current_player) ? 1.0 : 0.0;
    }

    void backpropagate(double result) {
        /*
        결과를 루트까지 역전파

        Args:
            result: 시뮬레이션 결과
        */
        MCTSNode* node = this;
        while (node != nullptr) {
            node->visits++;
            node->wins += result;
            result = 1.0 - result;  // 관점 전환 (부모는 상대편)
            node = node->parent;
        }
    }

    MCTSNode* best_child(double c = 0.0) const {
        /*
        최선의 자식 선택

        Args:
            c: 0이면 방문 횟수 기준, 0 아니면 UCB1 기준

        Returns:
            최선의 자식 노드
        */
        if (c == 0.0) {
            // 방문 횟수가 가장 많은 자식
            MCTSNode* best = nullptr;
            int max_visits = -1;
            for (const auto& child : children) {
                if (child->visits > max_visits) {
                    max_visits = child->visits;
                    best = child.get();
                }
            }
            return best;
        } else {
            // UCB1 값이 가장 큰 자식
            MCTSNode* best = nullptr;
            double best_ucb = -1.0;
            for (const auto& child : children) {
                double ucb = child->ucb1(c);
                if (ucb > best_ucb) {
                    best_ucb = ucb;
                    best = child.get();
                }
            }
            return best;
        }
    }
};

Move mcts(const TicTacToe& root_state, int iterations = 1000) {
    /*
    Monte Carlo Tree Search

    Args:
        root_state: 초기 게임 상태
        iterations: 시뮬레이션 반복 횟수

    Returns:
        최선의 수
    */
    auto root = make_unique<MCTSNode>(root_state);

    for (int i = 0; i < iterations; i++) {
        MCTSNode* node = root.get();

        // 1. Selection: 리프 노드까지 내려가기
        while (!node->is_terminal() && node->is_fully_expanded()) {
            node = node->select_child();
        }

        // 2. Expansion: 가능하면 확장
        if (!node->is_terminal() && !node->is_fully_expanded()) {
            node = node->expand();
        }

        // 3. Simulation: 게임 끝까지 랜덤 플레이
        double result = node->rollout();

        // 4. Backpropagation: 결과 역전파
        node->backpropagate(result);
    }

    // 최종 수 선택: 가장 많이 방문된 자식
    if (!root->children.empty()) {
        MCTSNode* best = root->best_child(0.0);
        double winrate = best->wins / best->visits;
        cout << "선택된 수: (" << best->move.r << "," << best->move.c
             << "), 방문: " << best->visits
             << ", 승률: " << (winrate * 100.0) << "%" << endl;
        return best->move;
    } else {
        // 가능한 수가 없음
        return Move();
    }
}

void play_game_mcts_vs_random() {
    /*MCTS vs 랜덤 플레이어*/
    cout << string(60, '=') << endl;
    cout << "틱택토: MCTS(X) vs 랜덤(O)" << endl;
    cout << string(60, '=') << endl;

    TicTacToe game;

    while (!game.is_terminal()) {
        cout << "\n현재 보드 (차례: " << (game.current_player == 1 ? "X" : "O") << "):" << endl;
        game.print();
        cout << endl;

        Move move;
        if (game.current_player == 1) {
            // MCTS 차례
            cout << "MCTS 생각 중..." << endl;
            move = mcts(game, 1000);
        } else {
            // 랜덤 차례
            auto moves = game.get_legal_moves();
            uniform_int_distribution<> dis(0, moves.size() - 1);
            move = moves[dis(gen)];
            cout << "랜덤 선택: (" << move.r << "," << move.c << ")" << endl;
        }

        game = game.apply_move(move);
    }

    // 최종 결과
    cout << "\n최종 보드:" << endl;
    game.print();
    cout << endl;

    int winner = game.check_winner();
    if (winner == 0) {
        cout << "결과: 무승부" << endl;
    } else if (winner == 1) {
        cout << "결과: MCTS(X) 승리!" << endl;
    } else {
        cout << "결과: 랜덤(O) 승리!" << endl;
    }
}

void play_game_mcts_vs_mcts() {
    /*MCTS vs MCTS*/
    cout << string(60, '=') << endl;
    cout << "틱택토: MCTS(X) vs MCTS(O)" << endl;
    cout << string(60, '=') << endl;

    TicTacToe game;

    while (!game.is_terminal()) {
        cout << "\n현재 보드 (차례: " << (game.current_player == 1 ? "X" : "O") << "):" << endl;
        game.print();
        cout << endl;

        cout << "MCTS 생각 중..." << endl;
        Move move = mcts(game, 500);

        game = game.apply_move(move);
    }

    // 최종 결과
    cout << "\n최종 보드:" << endl;
    game.print();
    cout << endl;

    int winner = game.check_winner();
    if (winner == 0) {
        cout << "결과: 무승부" << endl;
    } else if (winner == 1) {
        cout << "결과: MCTS(X) 승리!" << endl;
    } else {
        cout << "결과: MCTS(O) 승리!" << endl;
    }
}

void test_iterations() {
    /*시뮬레이션 횟수에 따른 성능 테스트*/
    cout << string(60, '=') << endl;
    cout << "성능 테스트: 시뮬레이션 횟수 vs 승률" << endl;
    cout << string(60, '=') << endl;

    vector<int> iteration_counts = {10, 50, 100, 500, 1000};
    int games_per_setting = 20;

    for (int iterations : iteration_counts) {
        int wins = 0, draws = 0, losses = 0;

        cout << "\nMCTS(" << iterations << " iterations) vs 랜덤: " << flush;

        for (int game_num = 0; game_num < games_per_setting; game_num++) {
            TicTacToe game;

            while (!game.is_terminal()) {
                Move move;
                if (game.current_player == 1) {
                    // MCTS
                    auto root = make_unique<MCTSNode>(game);
                    for (int i = 0; i < iterations; i++) {
                        MCTSNode* node = root.get();
                        while (!node->is_terminal() && node->is_fully_expanded()) {
                            node = node->select_child();
                        }
                        if (!node->is_terminal() && !node->is_fully_expanded()) {
                            node = node->expand();
                        }
                        double result = node->rollout();
                        node->backpropagate(result);
                    }
                    if (!root->children.empty()) {
                        MCTSNode* best = root->best_child(0.0);
                        move = best->move;
                    } else {
                        break;
                    }
                } else {
                    // 랜덤
                    auto moves = game.get_legal_moves();
                    uniform_int_distribution<> dis(0, moves.size() - 1);
                    move = moves[dis(gen)];
                }

                game = game.apply_move(move);
            }

            // 결과 집계
            int winner = game.check_winner();
            if (winner == 1) {
                wins++;
                cout << "W" << flush;
            } else if (winner == 2) {
                losses++;
                cout << "L" << flush;
            } else {
                draws++;
                cout << "D" << flush;
            }
        }

        cout << endl;
        cout << "  승: " << wins << "/" << games_per_setting
             << " (" << (wins * 100.0 / games_per_setting) << "%)" << endl;
        cout << "  무: " << draws << "/" << games_per_setting
             << " (" << (draws * 100.0 / games_per_setting) << "%)" << endl;
        cout << "  패: " << losses << "/" << games_per_setting
             << " (" << (losses * 100.0 / games_per_setting) << "%)" << endl;
    }
}

int main() {
    cout << "\n틱택토 MCTS 예제\n" << endl;
    cout << "1. MCTS vs 랜덤 (1게임)" << endl;
    cout << "2. MCTS vs MCTS (1게임)" << endl;
    cout << "3. 성능 테스트 (시뮬레이션 횟수별)" << endl;
    cout << "4. 모두 실행" << endl;
    cout << endl;

    string choice;
    cout << "선택 (1-4): ";
    getline(cin, choice);

    if (choice == "1") {
        play_game_mcts_vs_random();
    } else if (choice == "2") {
        play_game_mcts_vs_mcts();
    } else if (choice == "3") {
        test_iterations();
    } else if (choice == "4") {
        play_game_mcts_vs_random();
        cout << "\n" << string(60, '=') << "\n" << endl;
        play_game_mcts_vs_mcts();
        cout << "\n" << string(60, '=') << "\n" << endl;
        test_iterations();
    } else {
        cout << "잘못된 선택입니다." << endl;
    }

    return 0;
}
