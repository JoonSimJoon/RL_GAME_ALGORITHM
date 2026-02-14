/*
 * Value Iteration 구현
 * 벨만 최적 방정식을 반복 적용하여 최적 가치 함수와 정책을 찾는 알고리즘
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
#include <random>

// GridWorld class definition (copy from gridworld.cpp)
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

    std::vector<int> get_possible_actions(const State& state) const {
        if (is_terminal(state)) return {};
        return {0, 1, 2, 3};
    }

    void render(bool show_agent = true) const {
        std::cout << "\n" << std::string(grid_size * 4 + 1, '=') << "\n";
        for (int r = 0; r < grid_size; r++) {
            std::string row_str = "|";
            for (int c = 0; c < grid_size; c++) {
                State pos = {r, c};
                std::string cell;
                if (show_agent && pos == current_state) cell = " A ";
                else if (pos == goal) cell = " G ";
                else if (std::find(obstacles.begin(), obstacles.end(), pos) != obstacles.end()) cell = " X ";
                else if (pos == start) cell = " S ";
                else cell = "   ";
                row_str += cell + "|";
            }
            std::cout << row_str << "\n" << std::string(grid_size * 4 + 1, '=') << "\n";
        }
        std::cout << "\n";
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

    void render_values(const std::map<State, double>& values) const {
        std::cout << "\n가치 함수 시각화:\n" << std::string(grid_size * 7 + 1, '=') << "\n";
        for (int r = 0; r < grid_size; r++) {
            std::string row_str = "|";
            for (int c = 0; c < grid_size; c++) {
                State pos = {r, c};
                if (values.find(pos) != values.end()) {
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << std::setw(6) << values.at(pos);
                    row_str += oss.str() + "|";
                } else {
                    row_str += "  N/A |";
                }
            }
            std::cout << row_str << "\n" << std::string(grid_size * 7 + 1, '=') << "\n";
        }
        std::cout << "\n";
    }

    const std::vector<State>& get_states() const { return states; }
    const State& get_goal() const { return goal; }
    const std::vector<State>& get_obstacles() const { return obstacles; }
    const State& get_start() const { return start; }
    int get_grid_size() const { return grid_size; }
};

// Extract policy from value function
std::map<GridWorld::State, int> extract_policy(const GridWorld& env,
                                                const std::map<GridWorld::State, double>& V,
                                                double gamma) {
    std::map<GridWorld::State, int> policy;

    for (const auto& state : env.get_states()) {
        if (env.is_terminal(state)) {
            continue;
        }

        // 각 행동의 Q값 계산
        std::vector<double> q_values;
        for (int action = 0; action < env.n_actions; action++) {
            auto [next_state, reward, _] = env.step(&state, action);
            double q_value = reward + gamma * V.at(next_state);
            q_values.push_back(q_value);
        }

        // 최대 Q값을 주는 행동 선택
        int best_action = std::distance(q_values.begin(),
                                       std::max_element(q_values.begin(), q_values.end()));
        policy[state] = best_action;
    }

    return policy;
}

// Value Iteration Algorithm
std::pair<std::map<GridWorld::State, double>, std::map<GridWorld::State, int>>
value_iteration(GridWorld& env, double gamma = 0.9, double theta = 0.001, int max_iterations = 1000) {
    std::cout << "=== Value Iteration 시작 ===\n\n";

    // 1. 가치 함수 초기화
    std::map<GridWorld::State, double> V;
    for (const auto& state : env.get_states()) {
        V[state] = 0.0;
    }

    // 2. 가치 반복
    int iteration = 0;

    while (iteration < max_iterations) {
        double delta = 0;
        iteration++;

        // 모든 상태에 대해 업데이트
        for (const auto& state : env.get_states()) {
            // 종료 상태는 건너뛰기
            if (env.is_terminal(state)) {
                continue;
            }

            double v = V[state];  // 이전 가치 저장

            // 벨만 최적 방정식: V(s) = max_a [R + γV(s')]
            double max_value = -std::numeric_limits<double>::infinity();

            for (int action = 0; action < env.n_actions; action++) {
                // 다음 상태와 보상
                auto [next_state, reward, _] = env.step(&state, action);

                // Q(s, a) 계산
                double q_value = reward + gamma * V[next_state];

                // 최대값 찾기
                max_value = std::max(max_value, q_value);
            }

            V[state] = max_value;

            // 변화량 계산
            delta = std::max(delta, std::abs(v - V[state]));
        }

        // 진행 상황 출력 (5번마다)
        if (iteration % 5 == 0 || iteration == 1) {
            std::cout << "반복 " << std::setw(3) << iteration
                     << ": delta = " << std::fixed << std::setprecision(6) << delta << "\n";
        }

        // 3. 수렴 확인
        if (delta < theta) {
            std::cout << "\n수렴 완료! (반복 " << iteration << "회)\n";
            break;
        }
    }

    // 4. 최적 정책 추출
    auto policy = extract_policy(env, V, gamma);

    return {V, policy};
}

// 정책 평가 (시뮬레이션)
double evaluate_policy(GridWorld& env, const std::map<GridWorld::State, int>& policy,
                      double gamma = 0.9, int n_episodes = 100) {
    std::vector<double> total_rewards;
    std::mt19937 rng(42);

    for (int ep = 0; ep < n_episodes; ep++) {
        auto state = env.reset();
        double episode_reward = 0;
        double discount = 1.0;
        int steps = 0;
        int max_steps = 100;

        while (steps < max_steps) {
            if (policy.find(state) == policy.end() || env.is_terminal(state)) {
                break;
            }

            int action = policy.at(state);
            auto [next_state, reward, done] = env.step(nullptr, action);

            episode_reward += discount * reward;
            discount *= gamma;

            state = next_state;
            steps++;

            if (done) {
                break;
            }
        }

        total_rewards.push_back(episode_reward);
    }

    double sum = std::accumulate(total_rewards.begin(), total_rewards.end(), 0.0);
    return sum / total_rewards.size();
}

// 간단한 예시
void simple_example() {
    std::cout << "\n=== 간단한 예시 ===\n\n";

    GridWorld env(4);
    auto [V, policy] = value_iteration(env, 0.9);

    std::cout << "\n최적 정책:\n";
    env.render_policy(policy);

    std::cout << "\n최적 경로 시뮬레이션:\n";
    auto state = env.reset();
    int steps = 0;
    double total_reward = 0;

    std::cout << "시작: (" << state.first << ", " << state.second << ")\n";

    while (steps < 20) {
        if (env.is_terminal(state)) {
            std::cout << "\n목표 도달! (총 " << steps << "걸음)\n";
            break;
        }

        if (policy.find(state) == policy.end()) {
            std::cout << "\n정책에 없는 상태!\n";
            break;
        }

        int action = policy[state];
        auto [next_state, reward, done] = env.step(&state, action);
        total_reward += reward;

        std::cout << "  → " << GridWorld::ACTION_NAMES[action]
                 << " → (" << next_state.first << ", " << next_state.second << ") "
                 << "(보상: " << std::fixed << std::setprecision(2) << reward << ")\n";

        state = next_state;
        steps++;
    }

    std::cout << "\n총 보상: " << std::fixed << std::setprecision(4) << total_reward << "\n";
}

// 메인 함수
int main() {
    std::cout << std::string(60, '=') << "\n";
    std::cout << "Value Iteration 알고리즘 구현\n";
    std::cout << std::string(60, '=') << "\n";

    // 환경 생성
    GridWorld env(4);

    std::cout << "\n환경 초기화:\n";
    env.render();

    // Value Iteration 실행
    std::cout << "\n" << std::string(60, '=') << "\n";
    auto [V, policy] = value_iteration(env, 0.9, 0.001);

    // 결과 출력
    std::cout << "\n\n=== 최종 결과 ===\n\n";

    std::cout << "최적 가치 함수:\n";
    env.render_values(V);

    std::cout << "최적 정책:\n";
    env.render_policy(policy);

    // 정책 평가
    double avg_reward = evaluate_policy(env, policy, 0.9, 100);
    std::cout << "정책 성능 (평균 누적 보상): " << std::fixed << std::setprecision(4) << avg_reward << "\n\n";

    // 간단한 예시
    simple_example();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Value Iteration 완료!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
