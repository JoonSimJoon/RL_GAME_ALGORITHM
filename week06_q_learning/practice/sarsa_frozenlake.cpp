/*
 * SARSA를 사용한 FrozenLake 학습
 * Q-Learning과 비교하여 On-policy 알고리즘인 SARSA를 구현합니다.
 *
 * 핵심 차이점:
 * - Q-Learning: max Q(s',a')를 사용 (Off-policy)
 * - SARSA: Q(s',a')를 사용 (On-policy, 실제로 선택한 행동)
 *
 * 실습 목표:
 * 1. SARSA 알고리즘 구현
 * 2. Q-Learning과의 차이점 이해
 * 3. 두 알고리즘의 학습 곡선 비교
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
#include <numeric>

// FrozenLake Environment (4x4 grid)
class FrozenLakeEnv {
private:
    static constexpr int GRID_SIZE = 4;
    static constexpr int NUM_STATES = 16;
    static constexpr int NUM_ACTIONS = 4;

    bool is_slippery;
    int current_state;
    std::mt19937 rng;

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
        current_state = 0;
        return current_state;
    }

    std::tuple<int, double, bool> step(int action) {
        if (grid[current_state] == 'H' || grid[current_state] == 'G') {
            return {current_state, 0.0, true};
        }

        int actual_action = action;

        if (is_slippery) {
            std::uniform_real_distribution<> dis(0.0, 1.0);
            double prob = dis(rng);
            if (prob < 0.333) {
                actual_action = (action - 1 + NUM_ACTIONS) % NUM_ACTIONS;
            } else if (prob < 0.666) {
                // intended
            } else {
                actual_action = (action + 1) % NUM_ACTIONS;
            }
        }

        int row = current_state / GRID_SIZE;
        int col = current_state % GRID_SIZE;

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

// Base Agent class
class Agent {
protected:
    int num_states;
    int num_actions;
    std::vector<std::vector<double>> Q;
    std::mt19937 rng;

public:
    Agent(int states, int actions, unsigned seed = std::random_device{}())
        : num_states(states), num_actions(actions), rng(seed) {
        Q.resize(num_states, std::vector<double>(num_actions, 0.0));
    }

    int epsilon_greedy(int state, double epsilon) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng) < epsilon) {
            std::uniform_int_distribution<> action_dis(0, num_actions - 1);
            return action_dis(rng);
        } else {
            return std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
        }
    }

    int greedy(int state) const {
        return std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
    }

    void print_policy() const {
        std::array<std::string, 4> symbols = {"←", "↓", "→", "↑"};
        std::cout << "\n학습된 정책 (화살표):\n";
        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < 4; ++j) {
                int state = i * 4 + j;
                std::cout << symbols[greedy(state)] << " ";
            }
            std::cout << "\n";
        }
    }

    virtual void update(int state, int action, double reward, int next_state,
                       int next_action, bool done, double alpha, double gamma) = 0;

    virtual ~Agent() = default;
};

// SARSA Agent (On-policy)
class SARSAAgent : public Agent {
public:
    using Agent::Agent;

    void update(int state, int action, double reward, int next_state,
               int next_action, bool done, double alpha, double gamma) override {
        // SARSA: Q(s,a) ← Q(s,a) + α[R + γ·Q(s',a') - Q(s,a)]
        // Uses the actual next action selected (on-policy)
        double next_q = done ? 0.0 : Q[next_state][next_action];
        double td_target = reward + gamma * next_q;
        double td_error = td_target - Q[state][action];
        Q[state][action] += alpha * td_error;
    }
};

// Q-Learning Agent (Off-policy)
class QLearningAgent : public Agent {
public:
    using Agent::Agent;

    void update(int state, int action, double reward, int next_state,
               int next_action, bool done, double alpha, double gamma) override {
        // Q-Learning: Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
        // Uses max Q value (off-policy)
        double best_next_q = 0.0;
        if (!done) {
            best_next_q = *std::max_element(Q[next_state].begin(), Q[next_state].end());
        }
        double td_target = reward + gamma * best_next_q;
        double td_error = td_target - Q[state][action];
        Q[state][action] += alpha * td_error;
    }
};

// Evaluate policy
double evaluate_policy(FrozenLakeEnv& env, const Agent& agent, int num_episodes = 100) {
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

// Training result structure
struct TrainingResult {
    std::vector<double> rewards_history;
    std::vector<std::pair<int, double>> success_history;
};

// Train SARSA
TrainingResult train_sarsa(FrozenLakeEnv& env, SARSAAgent& agent,
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
        std::cout << "SARSA 학습 시작\n";
        std::cout << "상태 개수: " << env.get_num_states()
                  << ", 행동 개수: " << env.get_num_actions() << "\n";
        std::cout << "하이퍼파라미터: α=" << alpha << ", γ=" << gamma
                  << ", ε=" << epsilon_start << "→" << epsilon_min << "\n";
        std::cout << std::string(70, '-') << "\n";
    }

    for (int episode = 0; episode < num_episodes; ++episode) {
        int state = env.reset();
        // SARSA: Select first action before loop
        int action = agent.epsilon_greedy(state, epsilon);

        bool done = false;
        double total_reward = 0.0;

        while (!done) {
            // Execute current action
            auto [next_state, reward, terminated] = env.step(action);

            // SARSA: Select next action (important!)
            int next_action = agent.epsilon_greedy(next_state, epsilon);

            // Update using actual next action
            agent.update(state, action, reward, next_state, next_action, terminated, alpha, gamma);

            // Transition: use already selected next_action
            state = next_state;
            action = next_action;
            total_reward += reward;
            done = terminated;
        }

        result.rewards_history.push_back(total_reward);
        recent_rewards.push_back(total_reward);
        if (recent_rewards.size() > recent_window) {
            recent_rewards.pop_front();
        }

        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        if ((episode + 1) % eval_interval == 0) {
            double success_rate = evaluate_policy(env, agent, 100);
            result.success_history.push_back({episode + 1, success_rate});

            if (verbose) {
                double avg_reward = std::accumulate(recent_rewards.begin(), recent_rewards.end(), 0.0) / recent_rewards.size();
                std::cout << "Episode " << std::setw(5) << (episode + 1) << " | "
                          << "ε=" << std::fixed << std::setprecision(3) << epsilon << " | "
                          << "Avg Reward=" << std::setprecision(3) << avg_reward << " | "
                          << "Success Rate=" << std::setprecision(1) << (success_rate * 100) << "%\n";
            }
        }
    }

    if (verbose) {
        std::cout << std::string(70, '-') << "\n학습 완료!\n";
    }

    return result;
}

// Train Q-Learning (for comparison)
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
        std::cout << std::string(70, '-') << "\n";
    }

    for (int episode = 0; episode < num_episodes; ++episode) {
        int state = env.reset();
        bool done = false;
        double total_reward = 0.0;

        while (!done) {
            int action = agent.epsilon_greedy(state, epsilon);
            auto [next_state, reward, terminated] = env.step(action);

            agent.update(state, action, reward, next_state, 0, terminated, alpha, gamma);

            state = next_state;
            total_reward += reward;
            done = terminated;
        }

        result.rewards_history.push_back(total_reward);
        recent_rewards.push_back(total_reward);
        if (recent_rewards.size() > recent_window) {
            recent_rewards.pop_front();
        }

        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        if ((episode + 1) % eval_interval == 0) {
            double success_rate = evaluate_policy(env, agent, 100);
            result.success_history.push_back({episode + 1, success_rate});

            if (verbose) {
                double avg_reward = std::accumulate(recent_rewards.begin(), recent_rewards.end(), 0.0) / recent_rewards.size();
                std::cout << "Episode " << std::setw(5) << (episode + 1) << " | "
                          << "ε=" << std::fixed << std::setprecision(3) << epsilon << " | "
                          << "Avg Reward=" << std::setprecision(3) << avg_reward << " | "
                          << "Success Rate=" << std::setprecision(1) << (success_rate * 100) << "%\n";
            }
        }
    }

    if (verbose) {
        std::cout << std::string(70, '-') << "\n학습 완료!\n";
    }

    return result;
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "SARSA vs Q-Learning 비교 실습\n";
    std::cout << std::string(70, '=') << "\n";

    FrozenLakeEnv env(false);

    std::cout << "\n환경 정보:\n";
    std::cout << "  상태 공간: " << env.get_num_states() << "\n";
    std::cout << "  행동 공간: " << env.get_num_actions() << "\n";
    std::cout << "  is_slippery: false\n\n";

    // Hyperparameters
    const double alpha = 0.1;
    const double gamma = 0.99;
    const double epsilon_start = 1.0;
    const double epsilon_min = 0.01;
    const double epsilon_decay = 0.995;
    const int eval_interval = 500;

    // 1. SARSA Training
    std::cout << std::string(70, '=') << "\n";
    std::cout << "1. SARSA 학습\n";
    std::cout << std::string(70, '=') << "\n";

    SARSAAgent sarsa_agent(env.get_num_states(), env.get_num_actions());
    auto sarsa_result = train_sarsa(env, sarsa_agent, 10000, alpha, gamma,
                                    epsilon_start, epsilon_min, epsilon_decay,
                                    eval_interval, true);

    double final_success_sarsa = evaluate_policy(env, sarsa_agent, 1000);
    std::cout << "\nSARSA 최종 성공률: " << std::fixed << std::setprecision(2)
              << (final_success_sarsa * 100) << "%\n";
    std::cout << "\nSARSA 학습된 정책:";
    sarsa_agent.print_policy();

    // 2. Q-Learning Training
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "2. Q-Learning 학습\n";
    std::cout << std::string(70, '=') << "\n";

    QLearningAgent qlearn_agent(env.get_num_states(), env.get_num_actions());
    auto qlearn_result = train_q_learning(env, qlearn_agent, 10000, alpha, gamma,
                                          epsilon_start, epsilon_min, epsilon_decay,
                                          eval_interval, true);

    double final_success_qlearn = evaluate_policy(env, qlearn_agent, 1000);
    std::cout << "\nQ-Learning 최종 성공률: " << std::fixed << std::setprecision(2)
              << (final_success_qlearn * 100) << "%\n";
    std::cout << "\nQ-Learning 학습된 정책:";
    qlearn_agent.print_policy();

    // 3. Analysis
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "분석 결과\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "\nFrozenLake (is_slippery=false) 환경에서:\n";
    std::cout << "- Q-Learning과 SARSA의 성능이 매우 유사합니다\n";
    std::cout << "- 두 알고리즘 모두 최적 정책을 잘 학습합니다\n\n";
    std::cout << "이유:\n";
    std::cout << "- 결정적 환경이라 탐험 중 실수가 적음\n";
    std::cout << "- 간단한 환경이라 차이가 두드러지지 않음\n\n";
    std::cout << "차이가 나는 경우:\n";
    std::cout << "- is_slippery=true (확률적 환경)\n";
    std::cout << "- 위험한 상태가 많은 환경 (절벽 문제 등)\n";
    std::cout << "- SARSA가 더 보수적이고 안전한 정책 학습\n";

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "실습 완료!\n";
    std::cout << std::string(70, '=') << "\n";

    std::cout << "\n추가 실험 아이디어:\n";
    std::cout << "1. is_slippery=true로 변경하여 확률적 환경에서 비교\n";
    std::cout << "2. epsilon_min을 0.1로 높여서 탐험이 많을 때 차이 확인\n";
    std::cout << "3. gamma를 낮춰서 (0.5) 단기 보상 중시 시 차이 확인\n";

    return 0;
}
