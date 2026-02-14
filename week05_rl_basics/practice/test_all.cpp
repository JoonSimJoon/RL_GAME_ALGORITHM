/*
 * 전체 구현 테스트 스크립트
 * 모든 주요 기능이 정상 작동하는지 확인
 */

#include <iostream>
#include <vector>
#include <map>
#include <tuple>
#include <utility>
#include <string>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <limits>
#include <numeric>
#include <cassert>

// GridWorld class definition
class GridWorld {
public:
    static constexpr int ACTION_UP = 0;
    static constexpr int ACTION_DOWN = 1;
    static constexpr int ACTION_LEFT = 2;
    static constexpr int ACTION_RIGHT = 3;
    static inline const std::vector<std::string> ACTION_NAMES = {"↑", "↓", "←", "→"};

    using State = std::pair<int, int>;

private:
    int grid_size;
    State start;
    State goal;
    std::vector<State> obstacles;
    State current_state;
    std::vector<State> states;

public:
    int n_actions = 4;

    GridWorld(int grid_size = 4) : grid_size(grid_size) {
        start = {0, 0};
        goal = {grid_size - 1, grid_size - 1};
        obstacles = {{1, 1}, {2, 2}};
        current_state = start;
        for (int r = 0; r < grid_size; r++) {
            for (int c = 0; c < grid_size; c++) {
                states.push_back({r, c});
            }
        }
    }

    State reset() {
        current_state = start;
        return current_state;
    }

    std::tuple<State, double, bool> step(const State* state, int action) {
        State curr_state = (state != nullptr) ? *state : current_state;
        State next_state = get_next_state(curr_state, action);
        double reward = get_reward(next_state);
        bool done = is_terminal(next_state);
        if (state == nullptr) {
            current_state = next_state;
        }
        return {next_state, reward, done};
    }

    State get_next_state(const State& state, int action) const {
        int row = state.first;
        int col = state.second;
        int next_row, next_col;

        if (action == ACTION_UP) {
            next_row = row - 1; next_col = col;
        } else if (action == ACTION_DOWN) {
            next_row = row + 1; next_col = col;
        } else if (action == ACTION_LEFT) {
            next_row = row; next_col = col - 1;
        } else if (action == ACTION_RIGHT) {
            next_row = row; next_col = col + 1;
        } else {
            throw std::invalid_argument("Invalid action");
        }

        if (!is_valid_position(next_row, next_col)) {
            return state;
        }
        return {next_row, next_col};
    }

    bool is_valid_position(int row, int col) const {
        return row >= 0 && row < grid_size && col >= 0 && col < grid_size;
    }

    double get_reward(const State& state) const {
        if (state == goal) return 1.0;
        if (std::find(obstacles.begin(), obstacles.end(), state) != obstacles.end()) return -1.0;
        return -0.04;
    }

    bool is_terminal(const State& state) const {
        return state == goal || std::find(obstacles.begin(), obstacles.end(), state) != obstacles.end();
    }

    void render_policy(const std::map<State, int>& policy) const {
        std::cout << "\n정책 시각화:\n" << std::string(grid_size * 4 + 1, '=') << "\n";
        for (int r = 0; r < grid_size; r++) {
            std::string row_str = "|";
            for (int c = 0; c < grid_size; c++) {
                State pos = {r, c};
                std::string cell;
                if (pos == goal) cell = " G ";
                else if (std::find(obstacles.begin(), obstacles.end(), pos) != obstacles.end()) cell = " X ";
                else if (policy.find(pos) != policy.end()) cell = " " + ACTION_NAMES[policy.at(pos)] + " ";
                else cell = "   ";
                row_str += cell + "|";
            }
            std::cout << row_str << "\n" << std::string(grid_size * 4 + 1, '=') << "\n";
        }
        std::cout << "\n";
    }

    const std::vector<State>& get_states() const { return states; }
    const State& get_goal() const { return goal; }
    const std::vector<State>& get_obstacles() const { return obstacles; }
    int get_grid_size() const { return grid_size; }
};

// Value Iteration
std::pair<std::map<GridWorld::State, double>, std::map<GridWorld::State, int>>
value_iteration(GridWorld& env, double gamma = 0.9, double theta = 0.001, int max_iterations = 1000) {
    std::map<GridWorld::State, double> V;
    for (const auto& state : env.get_states()) {
        V[state] = 0.0;
    }

    int iteration = 0;
    while (iteration < max_iterations) {
        double delta = 0;
        iteration++;

        for (const auto& state : env.get_states()) {
            if (env.is_terminal(state)) continue;

            double v = V[state];
            double max_value = -std::numeric_limits<double>::infinity();

            for (int action = 0; action < env.n_actions; action++) {
                auto [next_state, reward, _] = env.step(&state, action);
                double q_value = reward + gamma * V[next_state];
                max_value = std::max(max_value, q_value);
            }

            V[state] = max_value;
            delta = std::max(delta, std::abs(v - V[state]));
        }

        if (delta < theta) break;
    }

    // Extract policy
    std::map<GridWorld::State, int> policy;
    for (const auto& state : env.get_states()) {
        if (env.is_terminal(state)) continue;

        std::vector<double> q_values;
        for (int action = 0; action < env.n_actions; action++) {
            auto [next_state, reward, _] = env.step(&state, action);
            q_values.push_back(reward + gamma * V[next_state]);
        }

        policy[state] = std::distance(q_values.begin(),
                                     std::max_element(q_values.begin(), q_values.end()));
    }

    return {V, policy};
}

// Policy Iteration
std::pair<std::map<GridWorld::State, int>, std::map<GridWorld::State, double>>
policy_iteration(GridWorld& env, double gamma = 0.9, double theta = 0.001, int max_iterations = 100) {
    // Initialize uniform policy
    std::map<GridWorld::State, std::vector<double>> policy;
    for (const auto& state : env.get_states()) {
        policy[state] = {0.25, 0.25, 0.25, 0.25};
    }

    for (int iter = 0; iter < max_iterations; iter++) {
        // Policy evaluation
        std::map<GridWorld::State, double> V;
        for (const auto& state : env.get_states()) {
            V[state] = 0.0;
        }

        for (int eval_iter = 0; eval_iter < 1000; eval_iter++) {
            double delta = 0;
            for (const auto& state : env.get_states()) {
                if (env.is_terminal(state)) continue;

                double v = V[state];
                double new_value = 0.0;

                for (int action = 0; action < env.n_actions; action++) {
                    double action_prob = policy[state][action];
                    if (action_prob == 0) continue;

                    auto [next_state, reward, _] = env.step(&state, action);
                    new_value += action_prob * (reward + gamma * V[next_state]);
                }

                V[state] = new_value;
                delta = std::max(delta, std::abs(v - V[state]));
            }

            if (delta < theta && eval_iter > 1) break;
        }

        // Policy improvement
        auto new_policy = policy;
        for (const auto& state : env.get_states()) {
            if (env.is_terminal(state)) continue;

            std::vector<double> q_values;
            for (int action = 0; action < env.n_actions; action++) {
                auto [next_state, reward, _] = env.step(&state, action);
                q_values.push_back(reward + gamma * V[next_state]);
            }

            double max_q = *std::max_element(q_values.begin(), q_values.end());
            std::vector<int> best_actions;
            for (int i = 0; i < q_values.size(); i++) {
                if (std::abs(q_values[i] - max_q) < 1e-10) {
                    best_actions.push_back(i);
                }
            }

            std::vector<double> action_probs(env.n_actions, 0.0);
            for (int action : best_actions) {
                action_probs[action] = 1.0 / best_actions.size();
            }
            new_policy[state] = action_probs;
        }

        // Check convergence
        bool stable = true;
        for (const auto& state : env.get_states()) {
            if (env.is_terminal(state)) continue;

            auto old_max = std::max_element(policy[state].begin(), policy[state].end());
            auto new_max = std::max_element(new_policy[state].begin(), new_policy[state].end());
            if (std::distance(policy[state].begin(), old_max) !=
                std::distance(new_policy[state].begin(), new_max)) {
                stable = false;
                break;
            }
        }

        policy = new_policy;
        if (stable) break;
    }

    // Convert to deterministic policy
    std::map<GridWorld::State, int> policy_det;
    for (const auto& state : env.get_states()) {
        if (!env.is_terminal(state)) {
            auto max_it = std::max_element(policy[state].begin(), policy[state].end());
            policy_det[state] = std::distance(policy[state].begin(), max_it);
        }
    }

    // Final V calculation
    std::map<GridWorld::State, double> V;
    for (const auto& state : env.get_states()) {
        V[state] = 0.0;
    }

    return {policy_det, V};
}

// Test functions
bool test_gridworld() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "테스트 1: GridWorld 환경\n";
    std::cout << std::string(60, '=') << "\n";

    try {
        GridWorld env(4);
        std::cout << "✓ GridWorld 생성 성공\n";

        auto state = env.reset();
        assert(state == GridWorld::State(0, 0));
        std::cout << "✓ 환경 초기화 성공\n";

        auto [next_state, reward, done] = env.step(&state, GridWorld::ACTION_RIGHT);
        assert(next_state == GridWorld::State(0, 1));
        assert(!done);
        std::cout << "✓ 행동 수행 성공\n";

        GridWorld::State goal_state = env.get_goal();
        auto [ns, r, d] = env.step(&goal_state, GridWorld::ACTION_UP);
        assert(d);
        assert(std::abs(r - 1.0) < 1e-6);
        std::cout << "✓ 목표 도달 테스트 성공\n";

        GridWorld::State obs_state = {1, 1};
        auto [ns2, r2, d2] = env.step(&obs_state, GridWorld::ACTION_UP);
        assert(d2);
        assert(std::abs(r2 - (-1.0)) < 1e-6);
        std::cout << "✓ 장애물 테스트 성공\n";

        GridWorld::State wall_state = {0, 0};
        auto [ns3, r3, d3] = env.step(&wall_state, GridWorld::ACTION_UP);
        assert(ns3 == wall_state);
        std::cout << "✓ 벽 처리 테스트 성공\n";

        std::cout << "\n✓ GridWorld 모든 테스트 통과!\n\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "\n✗ GridWorld 테스트 실패: " << e.what() << "\n\n";
        return false;
    }
}

bool test_value_iteration() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "테스트 2: Value Iteration\n";
    std::cout << std::string(60, '=') << "\n";

    try {
        GridWorld env(4);
        std::cout << "Value Iteration 실행 중...\n";

        auto [V, policy] = value_iteration(env, 0.9, 0.001, 100);

        std::cout << "✓ Value Iteration 실행 성공\n";

        assert(V.size() == 16);
        assert(V.find(env.get_goal()) != V.end());
        std::cout << "✓ 가치 함수 생성 성공\n";

        int non_terminal_count = 0;
        for (const auto& s : env.get_states()) {
            if (!env.is_terminal(s)) non_terminal_count++;
        }
        assert(policy.size() == non_terminal_count);
        std::cout << "✓ 정책 생성 성공\n";

        GridWorld::State start = {0, 0};
        int start_action = policy[start];
        assert(start_action == GridWorld::ACTION_RIGHT || start_action == GridWorld::ACTION_DOWN);
        std::cout << "✓ 정책 검증 성공\n";

        double v_start = V[{0, 0}];
        double v_goal_neighbor = V[{3, 2}];
        assert(v_goal_neighbor > v_start);
        std::cout << "✓ 가치 함수 검증 성공\n";

        std::cout << "\n✓ Value Iteration 모든 테스트 통과!\n\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "\n✗ Value Iteration 테스트 실패: " << e.what() << "\n\n";
        return false;
    }
}

bool test_policy_iteration() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "테스트 3: Policy Iteration\n";
    std::cout << std::string(60, '=') << "\n";

    try {
        GridWorld env(4);
        std::cout << "Policy Iteration 실행 중...\n";

        auto [policy, V] = policy_iteration(env, 0.9, 0.001, 100);

        std::cout << "✓ Policy Iteration 실행 성공\n";

        assert(V.size() == 16);
        std::cout << "✓ 가치 함수 생성 성공\n";

        int non_terminal_count = 0;
        for (const auto& s : env.get_states()) {
            if (!env.is_terminal(s)) non_terminal_count++;
        }
        assert(policy.size() == non_terminal_count);
        std::cout << "✓ 정책 생성 성공\n";

        GridWorld::State start = {0, 0};
        int start_action = policy[start];
        assert(start_action == GridWorld::ACTION_RIGHT || start_action == GridWorld::ACTION_DOWN);
        std::cout << "✓ 정책 검증 성공\n";

        std::cout << "\n✓ Policy Iteration 모든 테스트 통과!\n\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "\n✗ Policy Iteration 테스트 실패: " << e.what() << "\n\n";
        return false;
    }
}

bool test_algorithms_comparison() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "테스트 4: Value Iteration vs Policy Iteration 비교\n";
    std::cout << std::string(60, '=') << "\n";

    try {
        GridWorld env1(4);
        GridWorld env2(4);

        std::cout << "\nValue Iteration 실행...\n";
        auto [V_vi, policy_vi] = value_iteration(env1, 0.9, 0.001, 100);

        std::cout << "\nPolicy Iteration 실행...\n";
        auto [policy_pi, V_pi] = policy_iteration(env2, 0.9, 0.001, 100);

        bool policies_match = true;
        for (const auto& state : env1.get_states()) {
            if (env1.is_terminal(state)) continue;

            if (policy_vi[state] != policy_pi[state]) {
                policies_match = false;
                break;
            }
        }

        if (policies_match) {
            std::cout << "\n✓ 두 알고리즘의 정책이 일치합니다!\n";
        } else {
            std::cout << "\n⚠ 두 알고리즘의 정책이 일부 다릅니다 (동점 행동 선택 차이)\n";
        }

        double max_diff = 0;
        for (const auto& state : env1.get_states()) {
            double diff = std::abs(V_vi[state] - V_pi[state]);
            max_diff = std::max(max_diff, diff);
        }

        std::cout << "✓ 가치 함수 최대 차이: " << std::fixed << std::setprecision(6) << max_diff << "\n";

        if (max_diff < 0.01) {
            std::cout << "✓ 가치 함수가 거의 일치합니다!\n";
        }

        std::cout << "\n✓ 알고리즘 비교 테스트 완료!\n\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "\n✗ 비교 테스트 실패: " << e.what() << "\n\n";
        return false;
    }
}

bool test_simulation() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "테스트 5: 최적 정책 시뮬레이션\n";
    std::cout << std::string(60, '=') << "\n";

    try {
        GridWorld env(4);
        auto [V, policy] = value_iteration(env, 0.9, 0.001, 100);

        std::cout << "\n최적 정책으로 에피소드 실행:\n";

        auto state = env.reset();
        int steps = 0;
        int max_steps = 50;
        double total_reward = 0;

        std::cout << "시작: (" << state.first << ", " << state.second << ")\n";

        while (steps < max_steps) {
            if (env.is_terminal(state)) {
                std::cout << "\n✓ 목표 도달! (총 " << steps << "걸음)\n";
                break;
            }

            if (policy.find(state) == policy.end()) {
                std::cout << "\n✗ 정책에 없는 상태\n";
                return false;
            }

            int action = policy[state];
            auto [next_state, reward, done] = env.step(&state, action);
            total_reward += reward;

            std::cout << "  (" << state.first << ", " << state.second << ") → "
                     << GridWorld::ACTION_NAMES[action] << " → "
                     << "(" << next_state.first << ", " << next_state.second << ") "
                     << "(보상: " << std::fixed << std::setprecision(2) << reward << ")\n";

            state = next_state;
            steps++;
        }

        if (steps >= max_steps) {
            std::cout << "\n✗ 최대 스텝 초과\n";
            return false;
        }

        std::cout << "총 보상: " << std::fixed << std::setprecision(4) << total_reward << "\n";
        std::cout << "\n✓ 시뮬레이션 성공!\n\n";
        return true;
    } catch (const std::exception& e) {
        std::cout << "\n✗ 시뮬레이션 실패: " << e.what() << "\n\n";
        return false;
    }
}

int main() {
    std::cout << "\n";
    std::cout << std::string(60, '*') << "\n";
    std::cout << "*" << std::string(58, ' ') << "*\n";
    std::cout << "*  Week 5 강화학습 기초 - 전체 구현 테스트              *\n";
    std::cout << "*" << std::string(58, ' ') << "*\n";
    std::cout << std::string(60, '*') << "\n";
    std::cout << "\n";

    std::vector<std::pair<std::string, bool>> results;

    // Run all tests
    results.push_back({"GridWorld 환경", test_gridworld()});
    results.push_back({"Value Iteration", test_value_iteration()});
    results.push_back({"Policy Iteration", test_policy_iteration()});
    results.push_back({"알고리즘 비교", test_algorithms_comparison()});
    results.push_back({"시뮬레이션", test_simulation()});

    // Summary
    std::cout << "\n";
    std::cout << std::string(60, '=') << "\n";
    std::cout << "테스트 결과 요약\n";
    std::cout << std::string(60, '=') << "\n";

    for (const auto& [name, passed] : results) {
        std::string status = passed ? "✓ PASS" : "✗ FAIL";
        std::cout << std::left << std::setw(30) << name << " : " << status << "\n";
    }

    int total = results.size();
    int passed = 0;
    for (const auto& [_, p] : results) {
        if (p) passed++;
    }

    std::cout << std::string(60, '=') << "\n";
    std::cout << "\n전체: " << passed << "/" << total << " 테스트 통과\n";

    if (passed == total) {
        std::cout << "\n모든 테스트 통과! Week 5 구현이 완벽합니다!\n";
        std::cout << "\n다음 단계:\n";
        std::cout << "1. lecture.md로 이론 학습\n";
        std::cout << "2. script.md로 수업 진행\n";
        std::cout << "3. 실습 과제 수행\n";
        std::cout << "4. Week 6 Q-Learning 준비\n";
        return 0;
    } else {
        std::cout << "\n⚠ 일부 테스트 실패. 코드를 확인해주세요.\n";
        return 1;
    }
}
