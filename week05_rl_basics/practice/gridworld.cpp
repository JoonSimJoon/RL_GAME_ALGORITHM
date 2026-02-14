/*
 * GridWorld 환경 구현
 * 4x4 격자 세계에서 에이전트가 목표를 찾아가는 간단한 MDP 환경
 */

#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <string>
#include <algorithm>
#include <map>
#include <iomanip>
#include <cmath>

class GridWorld {
public:
    // 행동 정의
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

    // Constructor
    GridWorld(int grid_size = 4) : grid_size(grid_size) {
        start = {0, 0};
        goal = {grid_size - 1, grid_size - 1};
        obstacles = {{1, 1}, {2, 2}};
        current_state = start;

        // 모든 가능한 상태 리스트 생성
        for (int r = 0; r < grid_size; r++) {
            for (int c = 0; c < grid_size; c++) {
                states.push_back({r, c});
            }
        }
    }

    // 환경 초기화
    State reset() {
        current_state = start;
        return current_state;
    }

    // 상태 전이 수행
    std::tuple<State, double, bool> step(const State* state, int action) {
        State curr_state = (state != nullptr) ? *state : current_state;

        // 다음 상태 계산
        State next_state = get_next_state(curr_state, action);

        // 보상 계산
        double reward = get_reward(next_state);

        // 종료 확인
        bool done = is_terminal(next_state);

        // 현재 상태 업데이트
        if (state == nullptr) {
            current_state = next_state;
        }

        return {next_state, reward, done};
    }

    // 행동에 따른 다음 상태 계산
    State get_next_state(const State& state, int action) const {
        int row = state.first;
        int col = state.second;
        int next_row, next_col;

        if (action == ACTION_UP) {
            next_row = row - 1;
            next_col = col;
        } else if (action == ACTION_DOWN) {
            next_row = row + 1;
            next_col = col;
        } else if (action == ACTION_LEFT) {
            next_row = row;
            next_col = col - 1;
        } else if (action == ACTION_RIGHT) {
            next_row = row;
            next_col = col + 1;
        } else {
            throw std::invalid_argument("Invalid action: " + std::to_string(action));
        }

        // 벽 체크
        if (!is_valid_position(next_row, next_col)) {
            return state;
        }

        return {next_row, next_col};
    }

    // 위치가 격자 내부인지 확인
    bool is_valid_position(int row, int col) const {
        return row >= 0 && row < grid_size && col >= 0 && col < grid_size;
    }

    // 보상 계산
    double get_reward(const State& state) const {
        if (state == goal) {
            return 1.0;  // 목표 도달
        } else if (std::find(obstacles.begin(), obstacles.end(), state) != obstacles.end()) {
            return -1.0;  // 장애물 충돌
        } else {
            return -0.04;  // 일반 이동
        }
    }

    // 종료 상태 확인
    bool is_terminal(const State& state) const {
        return state == goal ||
               std::find(obstacles.begin(), obstacles.end(), state) != obstacles.end();
    }

    // 가능한 행동 리스트
    std::vector<int> get_possible_actions(const State& state) const {
        if (is_terminal(state)) {
            return {};
        }
        return {0, 1, 2, 3};
    }

    // 현재 상태 시각화
    void render(bool show_agent = true) const {
        std::cout << "\n" << std::string(grid_size * 4 + 1, '=') << "\n";

        for (int r = 0; r < grid_size; r++) {
            std::string row_str = "|";
            for (int c = 0; c < grid_size; c++) {
                State pos = {r, c};
                std::string cell;

                if (show_agent && pos == current_state) {
                    cell = " A ";  // Agent
                } else if (pos == goal) {
                    cell = " G ";  // Goal
                } else if (std::find(obstacles.begin(), obstacles.end(), pos) != obstacles.end()) {
                    cell = " X ";  // Obstacle
                } else if (pos == start) {
                    cell = " S ";  // Start
                } else {
                    cell = "   ";
                }

                row_str += cell + "|";
            }
            std::cout << row_str << "\n";
            std::cout << std::string(grid_size * 4 + 1, '=') << "\n";
        }
        std::cout << "\n";
    }

    // 정책 시각화
    void render_policy(const std::map<State, int>& policy) const {
        std::cout << "\n정책 시각화:\n";
        std::cout << std::string(grid_size * 4 + 1, '=') << "\n";

        for (int r = 0; r < grid_size; r++) {
            std::string row_str = "|";
            for (int c = 0; c < grid_size; c++) {
                State pos = {r, c};
                std::string cell;

                if (pos == goal) {
                    cell = " G ";
                } else if (std::find(obstacles.begin(), obstacles.end(), pos) != obstacles.end()) {
                    cell = " X ";
                } else if (policy.find(pos) != policy.end()) {
                    int action = policy.at(pos);
                    cell = " " + ACTION_NAMES[action] + " ";
                } else {
                    cell = "   ";
                }

                row_str += cell + "|";
            }
            std::cout << row_str << "\n";
            std::cout << std::string(grid_size * 4 + 1, '=') << "\n";
        }
        std::cout << "\n";
    }

    // 가치 함수 시각화
    void render_values(const std::map<State, double>& values) const {
        std::cout << "\n가치 함수 시각화:\n";
        std::cout << std::string(grid_size * 7 + 1, '=') << "\n";

        for (int r = 0; r < grid_size; r++) {
            std::string row_str = "|";
            for (int c = 0; c < grid_size; c++) {
                State pos = {r, c};

                if (values.find(pos) != values.end()) {
                    double value = values.at(pos);
                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(2) << std::setw(6) << value;
                    row_str += oss.str() + "|";
                } else {
                    row_str += "  N/A |";
                }
            }
            std::cout << row_str << "\n";
            std::cout << std::string(grid_size * 7 + 1, '=') << "\n";
        }
        std::cout << "\n";
    }

    // Getters
    const std::vector<State>& get_states() const { return states; }
    const State& get_goal() const { return goal; }
    const std::vector<State>& get_obstacles() const { return obstacles; }
    const State& get_start() const { return start; }
    int get_grid_size() const { return grid_size; }
};

// 테스트 함수들
void test_gridworld() {
    std::cout << "=== GridWorld 테스트 ===\n\n";

    GridWorld env(4);

    std::cout << "초기 상태:\n";
    env.render();

    std::cout << "\n행동 시퀀스 테스트:\n";
    std::vector<std::pair<int, std::string>> actions = {
        {GridWorld::ACTION_RIGHT, "오른쪽"},
        {GridWorld::ACTION_RIGHT, "오른쪽"},
        {GridWorld::ACTION_DOWN, "아래"},
        {GridWorld::ACTION_DOWN, "아래"}
    };

    double total_reward = 0;
    for (const auto& [action, action_name] : actions) {
        auto [next_state, reward, done] = env.step(nullptr, action);
        total_reward += reward;

        std::cout << "\n행동: " << action_name << "\n";
        std::cout << "다음 상태: (" << next_state.first << ", " << next_state.second << ")\n";
        std::cout << "보상: " << std::fixed << std::setprecision(2) << reward << "\n";
        std::cout << "종료: " << (done ? "true" : "false") << "\n";
        std::cout << "누적 보상: " << total_reward << "\n";

        env.render();

        if (done) {
            std::cout << "에피소드 종료!\n";
            break;
        }
    }

    // 장애물 테스트
    std::cout << "\n\n=== 장애물 테스트 ===\n";
    env.reset();
    std::cout << "\n초기 위치에서 아래로 이동 (장애물):\n";

    GridWorld::State test_state = {0, 0};
    auto [next_state, reward, done] = env.step(&test_state, GridWorld::ACTION_DOWN);
    std::cout << "다음 상태: (" << next_state.first << ", " << next_state.second << ")\n";
    std::cout << "보상: " << std::fixed << std::setprecision(2) << reward << "\n";
    std::cout << "종료: " << (done ? "true" : "false") << "\n";

    // 벽 테스트
    std::cout << "\n\n=== 벽 테스트 ===\n";
    std::cout << "(0, 0)에서 위로 이동 (벽):\n";
    test_state = {0, 0};
    auto [ns1, r1, d1] = env.step(&test_state, GridWorld::ACTION_UP);
    std::cout << "다음 상태: (" << ns1.first << ", " << ns1.second << ") (제자리)\n";
    std::cout << "보상: " << std::fixed << std::setprecision(2) << r1 << "\n";

    std::cout << "\n(0, 0)에서 왼쪽으로 이동 (벽):\n";
    auto [ns2, r2, d2] = env.step(&test_state, GridWorld::ACTION_LEFT);
    std::cout << "다음 상태: (" << ns2.first << ", " << ns2.second << ") (제자리)\n";
    std::cout << "보상: " << std::fixed << std::setprecision(2) << r2 << "\n";
}

void test_policy_visualization() {
    std::cout << "\n\n=== 정책 시각화 테스트 ===\n";

    GridWorld env(4);

    // 간단한 정책 (항상 오른쪽과 아래로)
    std::map<GridWorld::State, int> policy;
    for (int r = 0; r < env.get_grid_size(); r++) {
        for (int c = 0; c < env.get_grid_size(); c++) {
            GridWorld::State pos = {r, c};
            if (!env.is_terminal(pos)) {
                if (c < env.get_grid_size() - 1) {
                    policy[pos] = GridWorld::ACTION_RIGHT;
                } else {
                    policy[pos] = GridWorld::ACTION_DOWN;
                }
            }
        }
    }

    env.render_policy(policy);

    // 가치 함수 시각화
    std::map<GridWorld::State, double> values;
    for (int r = 0; r < env.get_grid_size(); r++) {
        for (int c = 0; c < env.get_grid_size(); c++) {
            GridWorld::State pos = {r, c};
            if (pos == env.get_goal()) {
                values[pos] = 1.0;
            } else if (std::find(env.get_obstacles().begin(), env.get_obstacles().end(), pos) != env.get_obstacles().end()) {
                values[pos] = -1.0;
            } else {
                // 목표까지의 맨해튼 거리
                int dist = std::abs(r - env.get_goal().first) + std::abs(c - env.get_goal().second);
                values[pos] = 1.0 - dist * 0.1;
            }
        }
    }

    env.render_values(values);
}

int main() {
    test_gridworld();
    test_policy_visualization();

    std::cout << "\n\n=== GridWorld 환경 준비 완료! ===\n";
    std::cout << "이제 value_iteration.cpp와 policy_iteration.cpp에서 사용할 수 있습니다.\n";

    return 0;
}
