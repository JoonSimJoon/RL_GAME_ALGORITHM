/*
 * Q-Learning을 사용한 FrozenLake 학습
 *
 * FrozenLake 환경을 C++로 구현하고 Q-Learning 알고리즘을 적용합니다.
 *
 * 실습 목표:
 * 1. Q-table 기반 Q-Learning 구현
 * 2. ε-greedy 탐험 전략 이해
 * 3. 학습 곡선 관찰
 * 4. 학습된 정책 시각화
 */

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <deque>
#include <string>
#include <fstream>

// FrozenLake Environment (4x4 grid)
class FrozenLakeEnv {
private:
    static constexpr int GRID_SIZE = 4;
    static constexpr int NUM_STATES = 16;
    static constexpr int NUM_ACTIONS = 4;  // LEFT=0, DOWN=1, RIGHT=2, UP=3

    bool is_slippery;
    int current_state;
    std::mt19937 rng;

    // Map layout: S=Start, F=Frozen, H=Hole, G=Goal
    // SFFF
    // FHFH
    // FFFH
    // HFFG
    std::array<char, NUM_STATES> grid = {
        'S', 'F', 'F', 'F',
        'F', 'H', 'F', 'H',
        'F', 'F', 'F', 'H',
        'H', 'F', 'F', 'G'
    };

public:
    FrozenLakeEnv(bool slippery = false, unsigned seed = std::random_device{}())
        : is_slippery(slippery), rng(seed) {
        reset();
    }

    int reset() {
        current_state = 0;  // Start position
        return current_state;
    }

    // Returns: (next_state, reward, terminated)
    std::tuple<int, double, bool> step(int action) {
        if (grid[current_state] == 'H' || grid[current_state] == 'G') {
            return {current_state, 0.0, true};
        }

        int actual_action = action;

        // Slippery: 33% chance each for intended, left, right
        if (is_slippery) {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            double prob = dis(rng);
            if (prob < 0.333) {
                actual_action = (action - 1 + NUM_ACTIONS) % NUM_ACTIONS;  // left
            } else if (prob < 0.666) {
                // intended action
            } else {
                actual_action = (action + 1) % NUM_ACTIONS;  // right
            }
        }

        int row = current_state / GRID_SIZE;
        int col = current_state % GRID_SIZE;

        // Apply action: LEFT=0, DOWN=1, RIGHT=2, UP=3
        if (actual_action == 0) col = std::max(0, col - 1);
        else if (actual_action == 1) row = std::min(GRID_SIZE - 1, row + 1);
        else if (actual_action == 2) col = std::min(GRID_SIZE - 1, col + 1);
        else if (actual_action == 3) row = std::max(0, row - 1);

        current_state = row * GRID_SIZE + col;

        double reward = 0.0;
        bool terminated = false;

        if (grid[current_state] == 'G') {
            reward = 1.0;
            terminated = true;
        } else if (grid[current_state] == 'H') {
            reward = 0.0;
            terminated = true;
        }

        return {current_state, reward, terminated};
    }

    int get_num_states() const { return NUM_STATES; }
    int get_num_actions() const { return NUM_ACTIONS; }
};

// Q-Learning Agent
class QLearningAgent {
private:
    int num_states;
    int num_actions;
    std::vector<std::vector<double>> Q;
    std::mt19937 rng;

public:
    QLearningAgent(int states, int actions, unsigned seed = std::random_device{}())
        : num_states(states), num_actions(actions), rng(seed) {
        Q.resize(num_states, std::vector<double>(num_actions, 0.0));
    }

    // ε-greedy action selection
    int epsilon_greedy(int state, double epsilon) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng) < epsilon) {
            // Explore: random action
            std::uniform_int_distribution<> action_dis(0, num_actions - 1);
            return action_dis(rng);
        } else {
            // Exploit: best action
            return std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
        }
    }

    // Greedy action selection (for evaluation)
    int greedy(int state) const {
        return std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
    }

    // Q-Learning update
    void update(int state, int action, double reward, int next_state, bool done,
                double alpha, double gamma) {
        double best_next_q = 0.0;
        if (!done) {
            best_next_q = *std::max_element(Q[next_state].begin(), Q[next_state].end());
        }

        double td_target = reward + gamma * best_next_q;
        double td_error = td_target - Q[state][action];
        Q[state][action] += alpha * td_error;
    }

    void print_policy() const {
        std::array<std::string, 4> symbols = {"←", "↓", "→", "↑"};
        std::cout << "\n학습된 정책 (화살표):\n";
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int state = i * 4 + j;
                int best_action = greedy(state);
                std::cout << symbols[best_action] << " ";
            }
            std::cout << "\n";
        }
    }

    void print_q_table(int num_states_to_show = 5) const {
        std::cout << "\nQ-table (상위 " << num_states_to_show << "개 상태):\n";
        std::cout << "State | Left   Down   Right  Up\n";
        std::cout << std::string(40, '-') << "\n";
        for (int state = 0; state < std::min(num_states_to_show, num_states); ++state) {
            std::cout << std::setw(5) << state << " | ";
            for (int action = 0; action < num_actions; ++action) {
                std::cout << std::fixed << std::setprecision(2) << std::setw(6) << Q[state][action] << " ";
            }
            std::cout << "\n";
        }
    }

    const std::vector<std::vector<double>>& get_q_table() const { return Q; }
};

// Evaluate policy
double evaluate_policy(FrozenLakeEnv& env, const QLearningAgent& agent, int num_episodes = 100) {
    int success_count = 0;

    for (int ep = 0; ep < num_episodes; ++ep) {
        int state = env.reset();
        bool done = false;

        while (!done) {
            int action = agent.greedy(state);
            auto [next_state, reward, terminated] = env.step(action);
            state = next_state;
            done = terminated;

            if (reward == 1.0) {
                success_count++;
            }
        }
    }

    return static_cast<double>(success_count) / num_episodes;
}

// Train Q-Learning
struct TrainingResult {
    std::vector<double> rewards_history;
    std::vector<std::pair<int, double>> success_history;
};

TrainingResult train_q_learning(FrozenLakeEnv& env, QLearningAgent& agent,
                                 int num_episodes = 10000,
                                 double alpha = 0.1,
                                 double gamma = 0.99,
                                 double epsilon_start = 1.0,
                                 double epsilon_min = 0.01,
                                 double epsilon_decay = 0.995,
                                 int eval_interval = 500,
                                 bool verbose = true) {
    TrainingResult result;
    double epsilon = epsilon_start;
    std::deque<double> recent_rewards;
    const int recent_window = 100;

    if (verbose) {
        std::cout << "Q-Learning 학습 시작\n";
        std::cout << "상태 개수: " << env.get_num_states()
                  << ", 행동 개수: " << env.get_num_actions() << "\n";
        std::cout << "하이퍼파라미터: α=" << alpha << ", γ=" << gamma
                  << ", ε=" << epsilon_start << "→" << epsilon_min << "\n";
        std::cout << std::string(70, '-') << "\n";
    }

    for (int episode = 0; episode < num_episodes; ++episode) {
        int state = env.reset();
        bool done = false;
        double total_reward = 0.0;

        // Episode execution
        while (!done) {
            int action = agent.epsilon_greedy(state, epsilon);
            auto [next_state, reward, terminated] = env.step(action);

            agent.update(state, action, reward, next_state, terminated, alpha, gamma);

            state = next_state;
            total_reward += reward;
            done = terminated;
        }

        result.rewards_history.push_back(total_reward);
        recent_rewards.push_back(total_reward);
        if (recent_rewards.size() > recent_window) {
            recent_rewards.pop_front();
        }

        // Decay epsilon
        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        // Periodic evaluation
        if ((episode + 1) % eval_interval == 0) {
            double success_rate = evaluate_policy(env, agent, 100);
            result.success_history.push_back({episode + 1, success_rate});

            if (verbose) {
                double avg_reward = 0.0;
                for (double r : recent_rewards) avg_reward += r;
                avg_reward /= recent_rewards.size();

                std::cout << "Episode " << std::setw(5) << (episode + 1) << " | "
                          << "ε=" << std::fixed << std::setprecision(3) << epsilon << " | "
                          << "Avg Reward=" << std::setprecision(3) << avg_reward << " | "
                          << "Success Rate=" << std::setprecision(1) << (success_rate * 100) << "%\n";
            }
        }
    }

    if (verbose) {
        std::cout << std::string(70, '-') << "\n";
        std::cout << "학습 완료!\n";
    }

    return result;
}

// Save results to file
void save_results(const TrainingResult& result, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << "\n";
        return;
    }

    file << "Episode,Reward\n";
    for (size_t i = 0; i < result.rewards_history.size(); ++i) {
        file << (i + 1) << "," << result.rewards_history[i] << "\n";
    }

    file.close();
    std::cout << "Results saved to " << filename << "\n";
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "FrozenLake Q-Learning 실습\n";
    std::cout << std::string(70, '=') << "\n";

    // Create environment
    FrozenLakeEnv env(false);  // is_slippery=false

    std::cout << "\n환경 정보:\n";
    std::cout << "  상태 공간: " << env.get_num_states() << " (4x4 격자)\n";
    std::cout << "  행동 공간: " << env.get_num_actions() << " (←↓→↑)\n";
    std::cout << "  is_slippery: false (결정적 환경)\n\n";

    // Create agent
    QLearningAgent agent(env.get_num_states(), env.get_num_actions());

    // Train
    auto result = train_q_learning(
        env, agent,
        10000,      // num_episodes
        0.1,        // alpha
        0.99,       // gamma
        1.0,        // epsilon_start
        0.01,       // epsilon_min
        0.995,      // epsilon_decay
        500,        // eval_interval
        true        // verbose
    );

    // Final evaluation
    std::cout << "\n=== 최종 성능 평가 ===\n";
    double final_success_rate = evaluate_policy(env, agent, 1000);
    std::cout << "1000번 테스트 성공률: " << std::fixed << std::setprecision(2)
              << (final_success_rate * 100) << "%\n";

    // Print learned policy
    agent.print_policy();

    // Print Q-table
    agent.print_q_table(5);

    // Save results
    save_results(result, "q_learning_results.csv");

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "실습 완료!\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n추가 실험 아이디어:\n";
    std::cout << "1. is_slippery=true로 변경하여 확률적 환경에서 학습\n";
    std::cout << "2. alpha를 0.01, 0.3, 0.5로 바꿔가며 학습 속도 비교\n";
    std::cout << "3. gamma를 0.5, 0.9로 바꿔가며 장기 계획의 중요성 확인\n";
    std::cout << "4. epsilon_decay를 0.99, 0.999로 바꿔가며 탐험 영향 확인\n";

    return 0;
}
