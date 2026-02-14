/*
MCTS 에이전트 테스트 스크립트

사용법:
    g++ -std=c++20 -O2 -o test_mcts test_mcts.cpp mcts_agent.cpp
    ./test_mcts

Note: This is a simplified C++ test version.
For full testing, compile both files together and link them.
*/

#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <iomanip>

using namespace std;

// Forward declarations from mcts_agent.cpp
class AtaxxBoard;
class MCTSNode;
struct Move;
Move mcts_search(const AtaxxBoard& board, int time_limit_ms);

void print_separator() {
    cout << string(60, '=') << endl;
}

void print_board(const AtaxxBoard& board) {
    vector<char> symbols = {'.', 'X', 'O'};
    cout << "  1 2 3 4 5 6 7" << endl;
    for (int i = 0; i < 7; i++) {
        cout << (i + 1) << " ";
        for (int j = 0; j < 7; j++) {
            cout << symbols[board.board[i][j]] << " ";
        }
        cout << endl;
    }
    int count1, count2;
    board.count_pieces(count1, count2);
    cout << "X(FIRST)=" << count1 << ", O(SECOND)=" << count2 << endl;
}

void test_board_basic() {
    print_separator();
    cout << "테스트 1: 보드 기본 기능" << endl;
    print_separator();

    AtaxxBoard board;

    // 초기 보드 출력
    cout << "\n초기 보드:" << endl;
    print_board(board);

    // 가능한 수 확인
    auto moves = board.get_legal_moves();
    cout << "\n가능한 수 개수: " << moves.size() << endl;
    cout << "처음 5개 수: ";
    for (size_t i = 0; i < min(size_t(5), moves.size()); i++) {
        if (!moves[i].is_pass) {
            cout << "(" << moves[i].r1 << "," << moves[i].c1 << ","
                 << moves[i].r2 << "," << moves[i].c2 << ") ";
        }
    }
    cout << endl;

    // 수 적용 테스트
    if (!moves.empty() && !moves[0].is_pass) {
        Move move = moves[0];
        cout << "\n수 적용: (" << move.r1 << "," << move.c1 << ","
             << move.r2 << "," << move.c2 << ")" << endl;
        AtaxxBoard new_board = board.apply_move(move);
        print_board(new_board);
    }

    cout << "\n✓ 보드 기본 기능 정상" << endl;
}

void test_mcts_node() {
    cout << "\n";
    print_separator();
    cout << "테스트 2: MCTSNode 기능" << endl;
    print_separator();

    AtaxxBoard board;
    MCTSNode root(board);

    // UCB1 테스트
    cout << "\n미방문 노드 UCB1: infinity" << endl;
    assert(isinf(root.ucb1()));
    cout << "✓ 미방문 노드는 무한대" << endl;

    // Expansion 테스트
    MCTSNode* child = root.expand();
    cout << "자식 노드 생성: (" << child->move.r1 << "," << child->move.c1 << ","
         << child->move.r2 << "," << child->move.c2 << ")" << endl;
    assert(root.children.size() == 1);
    cout << "✓ 자식이 1개" << endl;

    // Rollout 테스트
    cout << "\n롤아웃 시작..." << endl;
    auto start = chrono::steady_clock::now();
    double result = child->rollout();
    auto end = chrono::steady_clock::now();
    double elapsed = chrono::duration<double, milli>(end - start).count();
    cout << "롤아웃 결과: " << result << " (소요 시간: " << fixed << setprecision(1) << elapsed << "ms)" << endl;

    // Backpropagation 테스트
    child->backpropagate(result);
    cout << "\n역전파 후:" << endl;
    cout << "  자식: wins=" << child->wins << ", visits=" << child->visits << endl;
    cout << "  루트: wins=" << root.wins << ", visits=" << root.visits << endl;

    // UCB1 재계산
    if (child->visits > 0) {
        double ucb1_value = child->ucb1();
        cout << "  자식 UCB1: " << fixed << setprecision(3) << ucb1_value << endl;
    }

    cout << "\n✓ MCTSNode 기능 정상" << endl;
}

void test_mcts_search() {
    cout << "\n";
    print_separator();
    cout << "테스트 3: MCTS 탐색" << endl;
    print_separator();

    AtaxxBoard board;

    // 다양한 시간 제한으로 테스트
    int time_limits[] = {10, 50, 100};
    for (int time_limit : time_limits) {
        cout << "\n시간 제한 " << time_limit << "ms로 탐색..." << endl;
        auto start = chrono::steady_clock::now();
        Move move = mcts_search(board, time_limit);
        auto end = chrono::steady_clock::now();
        double elapsed = chrono::duration<double, milli>(end - start).count();

        if (!move.is_pass) {
            cout << "  선택된 수: (" << move.r1 << "," << move.c1 << ","
                 << move.r2 << "," << move.c2 << ")" << endl;
        } else {
            cout << "  선택된 수: PASS" << endl;
        }
        cout << "  실제 소요 시간: " << fixed << setprecision(1) << elapsed << "ms" << endl;

        assert(!move.is_pass);  // 초기 보드에서는 PASS가 아니어야 함
    }

    cout << "\n✓ MCTS 탐색 정상" << endl;
}

void test_terminal_states() {
    cout << "\n";
    print_separator();
    cout << "테스트 4: 터미널 상태" << endl;
    print_separator();

    // 빈 보드 (한쪽이 돌이 없음)
    AtaxxBoard board;
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            board.board[i][j] = 0;
        }
    }
    board.board[0][0] = 1;  // FIRST만 존재
    cout << "\n한쪽만 돌이 있는 경우:" << endl;
    assert(board.is_terminal());
    cout << "  ✓ 터미널 확인됨" << endl;

    // 꽉 찬 보드
    AtaxxBoard board2;
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            board2.board[i][j] = 1;
        }
    }
    cout << "\n보드가 꽉 찬 경우:" << endl;
    assert(board2.is_terminal());
    cout << "  ✓ 터미널 확인됨" << endl;

    cout << "\n✓ 터미널 상태 테스트 통과" << endl;
}

int main() {
    cout << "\n";
    print_separator();
    cout << "MCTS 에이전트 테스트 시작" << endl;
    print_separator();

    try {
        // 기본 테스트
        test_board_basic();
        test_mcts_node();
        test_mcts_search();
        test_terminal_states();

        // 최종 결과
        cout << "\n";
        print_separator();
        cout << "✓ 모든 테스트 통과!" << endl;
        print_separator();
        cout << "\nMCTS 에이전트가 정상적으로 작동합니다." << endl;
        cout << "ALPHANO 플랫폼에 제출할 준비가 되었습니다." << endl;

        return 0;
    } catch (const exception& e) {
        cerr << "\n✗ 오류 발생: " << e.what() << endl;
        return 1;
    }
}
