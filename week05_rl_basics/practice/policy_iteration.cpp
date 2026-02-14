/*
 * Policy Iteration 구현
 * 정책 평가와 정책 개선을 번갈아 수행하여 최적 정책을 찾는 알고리즘
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

// Policy evaluation
std::map<GridWorld::State, double> policy_evaluation(
    const GridWorld& env,
    const std::map<GridWorld::State, std::vector<double>>& policy,
    double gamma = 0.9,
    double theta = 0.001,
    int max_iterations = 1000) {

    // 가치 함수 초기화
    std::map<GridWorld::State, double> V;
    for (const auto& state : env.get_states()) {
        V[state] = 0.0;
    }

    int iteration = 0;
    while (iteration < max_iterations) {
        double delta = 0;
        iteration++;

        // 모든 상태에 대해 업데이트
        for (const auto& state : env.get_states()) {
            if (env.is_terminal(state)) {
                continue;
            }

            double v = V[state];  // 이전 가치 저장

            // 벨만 기대 방정식: V^π(s) = Σ_a π(a|s)[R + γV(s')]
            double new_value = 0.0;

            for (int action = 0; action < env.n_actions; action++) {
                // 정책에 따른 행동 확률
                double action_prob = policy.at(state)[action];

                if (action_prob == 0) {
                    continue;
                }

                // 다음 상태와 보상
                auto [next_state, reward, _] = env.step(&state, action);

                // 기대값 계산
                new_value += action_prob * (reward + gamma * V[next_state]);
            }

            V[state] = new_value;

            // 변화량 계산
            delta = std::max(delta, std::abs(v - V[state]));
        }

        // 수렴 확인
        if (delta < theta) {
            if (iteration > 1) {  // 최소 2번은 반복
                break;
            }
        }
    }

    return V;
}

// Policy improvement
std::map<GridWorld::State, std::vector<double>> policy_improvement(
    const GridWorld& env,
    const std::map<GridWorld::State, double>& V,
    double gamma = 0.9) {

    std::map<GridWorld::State, std::vector<double>> policy;

    for (const auto& state : env.get_states()) {
        if (env.is_terminal(state)) {
            // 종료 상태는 균등 정책
            policy[state] = {0.25, 0.25, 0.25, 0.25};
            continue;
        }

        // 각 행동의 Q값 계산
        std::vector<double> q_values;
        for (int action = 0; action < env.n_actions; action++) {
            auto [next_state, reward, _] = env.step(&state, action);
            double q_value = reward + gamma * V.at(next_state);
            q_values.push_back(q_value);
        }

        // 최대 Q값을 가진 행동 찾기
        double max_q = *std::max_element(q_values.begin(), q_values.end());
        std::vector<int> best_actions;
        for (int i = 0; i < q_values.size(); i++) {
            if (std::abs(q_values[i] - max_q) < 1e-10) {
                best_actions.push_back(i);
            }
        }

        // 결정적 정책 생성
        std::vector<double> action_probs(env.n_actions, 0.0);
        for (int action : best_actions) {
            action_probs[action] = 1.0 / best_actions.size();
        }

        policy[state] = action_probs;
    }

    return policy;
}

// Policy Iteration Algorithm
std::pair<std::map<GridWorld::State, int>, std::map<GridWorld::State, double>>
policy_iteration(GridWorld& env, double gamma = 0.9, double theta = 0.001, int max_iterations = 100) {
    std::cout << "=== Policy Iteration 시작 ===\n\n";

    // 1. 초기 정책 생성 (균등 정책)
    std::map<GridWorld::State, std::vector<double>> policy;
    for (const auto& state : env.get_states()) {
        policy[state] = {0.25, 0.25, 0.25, 0.25};
    }

    std::cout << "초기 정책: 균등 정책 (모든 행동 25%)\n";

    int iteration = 0;
    while (iteration < max_iterations) {
        iteration++;
        std::cout << "\n" << std::string(60, '=') << "\n";
        std::cout << "반복 " << iteration << "\n";
        std::cout << std::string(60, '=') << "\n";

        // 2. 정책 평가
        std::cout << "정책 평가 중...\n";
        auto V = policy_evaluation(env, policy, gamma, theta);

        // 3. 정책 개선
        std::cout << "정책 개선 중...\n";
        auto new_policy = policy_improvement(env, V, gamma);

        // 4. 수렴 확인
        bool policy_stable = true;
        for (const auto& state : env.get_states()) {
            if (env.is_terminal(state)) {
                continue;
            }

            // 최선의 행동이 바뀌었는지 확인
            auto old_max = std::max_element(policy[state].begin(), policy[state].end());
            auto new_max = std::max_element(new_policy[state].begin(), new_policy[state].end());
            int old_best = std::distance(policy[state].begin(), old_max);
            int new_best = std::distance(new_policy[state].begin(), new_max);

            if (old_best != new_best) {
                policy_stable = false;
                break;
            }
        }

        if (policy_stable) {
            std::cout << "\n정책이 수렴했습니다! (반복 " << iteration << "회)\n";
            policy = new_policy;
            break;
        }

        policy = new_policy;
    }

    // 최종 정책을 결정적 형태로 변환
    std::map<GridWorld::State, int> policy_deterministic;
    for (const auto& state : env.get_states()) {
        if (!env.is_terminal(state)) {
            auto max_it = std::max_element(policy[state].begin(), policy[state].end());
            policy_deterministic[state] = std::distance(policy[state].begin(), max_it);
        }
    }

    // 최종 가치 함수 계산
    auto V = policy_evaluation(env, policy, gamma, theta);

    return {policy_deterministic, V};
}

// 정책 평가 (시뮬레이션)
double evaluate_policy(GridWorld& env, const std::map<GridWorld::State, int>& policy,
                      double gamma = 0.9, int n_episodes = 100) {
    std::vector<double> total_rewards;

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
    auto [policy, V] = policy_iteration(env, 0.9);

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
    std::cout << "Policy Iteration 알고리즘 구현\n";
    std::cout << std::string(60, '=') << "\n";

    // 환경 생성
    GridWorld env(4);

    std::cout << "\n환경 초기화:\n";
    env.render();

    // Policy Iteration 실행
    std::cout << "\n" << std::string(60, '=') << "\n";
    auto [policy, V] = policy_iteration(env, 0.9, 0.001);

    // 결과 출력
    std::cout << "\n\n=== 최종 결과 ===\n\n";

    std::cout << "최적 가치 함수:\n";
    env.render_values(V);

    std::cout << "최적 정책:\n";
    env.render_policy(policy);

    // 정책 평가
    double avg_reward = evaluate_policy(env, policy, 0.9);
    std::cout << "정책 성능 (평균 누적 보상): " << std::fixed << std::setprecision(4) << avg_reward << "\n\n";

    // 간단한 예시
    simple_example();

    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "Policy Iteration 완료!\n";
    std::cout << std::string(60, '=') << "\n";

    return 0;
}
