/*
 * 하이퍼파라미터 실험
 * Q-Learning의 핵심 하이퍼파라미터(α, γ, ε)를 변화시키며
 * 학습 성능에 미치는 영향을 분석합니다.
 *
 * 실험 하이퍼파라미터:
 * 1. α (학습률): 새로운 정보를 얼마나 빠르게 받아들일지
 * 2. γ (할인율): 미래 보상을 얼마나 중요하게 볼지
 * 3. ε_decay (탐험률 감소): 탐험을 얼마나 빠르게 줄일지
 *
 * 실습 목표:
 * 1. 각 하이퍼파라미터의 역할 이해
 * 2. 하이퍼파라미터 변화가 학습에 미치는 영향 관찰
 * 3. 최적 하이퍼파라미터 조합 찾기
 */

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <string>
#include <map>
#include <numeric>

// FrozenLake Environment
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
            if (prob < 0.333) actual_action = (action - 1 + NUM_ACTIONS) % NUM_ACTIONS;
            else if (prob >= 0.666) actual_action = (action + 1) % NUM_ACTIONS;
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

    int epsilon_greedy(int state, double epsilon) {
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(rng) < epsilon) {
            std::uniform_int_distribution<> action_dis(0, num_actions - 1);
            return action_dis(rng);
        }
        return std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
    }

    int greedy(int state) const {
        return std::max_element(Q[state].begin(), Q[state].end()) - Q[state].begin();
    }

    void update(int state, int action, double reward, int next_state, bool done,
                double alpha, double gamma) {
        double best_next_q = done ? 0.0 : *std::max_element(Q[next_state].begin(), Q[next_state].end());
        double td_target = reward + gamma * best_next_q;
        Q[state][action] += alpha * (td_target - Q[state][action]);
    }
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
            if (reward == 1.0) success_count++;
        }
    }
    return static_cast<double>(success_count) / num_episodes;
}

// Training function
std::vector<std::pair<int, double>> train_q_learning(
    FrozenLakeEnv& env,
    QLearningAgent& agent,
    int num_episodes,
    double alpha,
    double gamma,
    double epsilon_start,
    double epsilon_min,
    double epsilon_decay,
    int eval_interval = 250) {

    std::vector<std::pair<int, double>> success_history;
    double epsilon = epsilon_start;

    for (int episode = 0; episode < num_episodes; ++episode) {
        int state = env.reset();
        bool done = false;

        while (!done) {
            int action = agent.epsilon_greedy(state, epsilon);
            auto [next_state, reward, terminated] = env.step(action);
            agent.update(state, action, reward, next_state, terminated, alpha, gamma);
            state = next_state;
            done = terminated;
        }

        epsilon = std::max(epsilon_min, epsilon * epsilon_decay);

        if ((episode + 1) % eval_interval == 0) {
            double success_rate = evaluate_policy(env, agent, 100);
            success_history.push_back({episode + 1, success_rate});
        }
    }

    return success_history;
}

// Experiment with alpha
std::map<double, std::vector<std::pair<int, double>>> experiment_alpha(
    const std::vector<double>& alphas,
    int num_episodes = 5000,
    int num_runs = 3) {

    std::cout << std::string(70, '=') << "\n";
    std::cout << "실험 1: 학습률(α) 변화\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "테스트할 α 값: ";
    for (double a : alphas) std::cout << a << " ";
    std::cout << "\n각 설정당 " << num_runs << "회 반복\n\n";

    std::map<double, std::vector<std::pair<int, double>>> results;

    for (double alpha : alphas) {
        std::cout << "α = " << alpha << " 실험 중... " << std::flush;

        std::map<int, std::vector<double>> avg_history;

        for (int run = 0; run < num_runs; ++run) {
            FrozenLakeEnv env(false);
            QLearningAgent agent(env.get_num_states(), env.get_num_actions());
            auto history = train_q_learning(env, agent, num_episodes, alpha, 0.99, 1.0, 0.01, 0.995, 250);

            for (const auto& [ep, rate] : history) {
                avg_history[ep].push_back(rate);
            }
        }

        // Calculate averages
        for (const auto& [ep, rates] : avg_history) {
            double avg = std::accumulate(rates.begin(), rates.end(), 0.0) / rates.size();
            results[alpha].push_back({ep, avg});
        }

        std::cout << "완료! 최종 성공률: " << std::fixed << std::setprecision(1)
                  << (results[alpha].back().second * 100) << "%\n";
    }

    return results;
}

// Experiment with gamma
std::map<double, std::vector<std::pair<int, double>>> experiment_gamma(
    const std::vector<double>& gammas,
    int num_episodes = 5000,
    int num_runs = 3) {

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "실험 2: 할인율(γ) 변화\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "테스트할 γ 값: ";
    for (double g : gammas) std::cout << g << " ";
    std::cout << "\n각 설정당 " << num_runs << "회 반복\n\n";

    std::map<double, std::vector<std::pair<int, double>>> results;

    for (double gamma : gammas) {
        std::cout << "γ = " << gamma << " 실험 중... " << std::flush;

        std::map<int, std::vector<double>> avg_history;

        for (int run = 0; run < num_runs; ++run) {
            FrozenLakeEnv env(false);
            QLearningAgent agent(env.get_num_states(), env.get_num_actions());
            auto history = train_q_learning(env, agent, num_episodes, 0.1, gamma, 1.0, 0.01, 0.995, 250);

            for (const auto& [ep, rate] : history) {
                avg_history[ep].push_back(rate);
            }
        }

        for (const auto& [ep, rates] : avg_history) {
            double avg = std::accumulate(rates.begin(), rates.end(), 0.0) / rates.size();
            results[gamma].push_back({ep, avg});
        }

        std::cout << "완료! 최종 성공률: " << std::fixed << std::setprecision(1)
                  << (results[gamma].back().second * 100) << "%\n";
    }

    return results;
}

// Experiment with epsilon_decay
std::map<double, std::vector<std::pair<int, double>>> experiment_epsilon_decay(
    const std::vector<double>& decays,
    int num_episodes = 5000,
    int num_runs = 3) {

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "실험 3: 탐험률 감소율(ε_decay) 변화\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "테스트할 ε_decay 값: ";
    for (double d : decays) std::cout << d << " ";
    std::cout << "\n각 설정당 " << num_runs << "회 반복\n\n";

    std::map<double, std::vector<std::pair<int, double>>> results;

    for (double decay : decays) {
        std::cout << "ε_decay = " << decay << " 실험 중... " << std::flush;

        std::map<int, std::vector<double>> avg_history;

        for (int run = 0; run < num_runs; ++run) {
            FrozenLakeEnv env(false);
            QLearningAgent agent(env.get_num_states(), env.get_num_actions());
            auto history = train_q_learning(env, agent, num_episodes, 0.1, 0.99, 1.0, 0.01, decay, 250);

            for (const auto& [ep, rate] : history) {
                avg_history[ep].push_back(rate);
            }
        }

        for (const auto& [ep, rates] : avg_history) {
            double avg = std::accumulate(rates.begin(), rates.end(), 0.0) / rates.size();
            results[decay].push_back({ep, avg});
        }

        std::cout << "완료! 최종 성공률: " << std::fixed << std::setprecision(1)
                  << (results[decay].back().second * 100) << "%\n";
    }

    return results;
}

// Grid search
struct GridSearchResult {
    double alpha;
    double gamma;
    double epsilon_decay;
    double score;

    bool operator<(const GridSearchResult& other) const {
        return score > other.score;  // descending order
    }
};

std::vector<GridSearchResult> grid_search(
    const std::vector<double>& alphas,
    const std::vector<double>& gammas,
    const std::vector<double>& decays,
    int num_episodes = 5000) {

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "그리드 서치: 최적 하이퍼파라미터 찾기\n";
    std::cout << std::string(70, '=') << "\n";

    int total_combinations = alphas.size() * gammas.size() * decays.size();
    std::cout << "총 " << total_combinations << "개 조합 테스트\n";
    std::cout << std::string(70, '-') << "\n";

    std::vector<GridSearchResult> all_results;
    int idx = 0;

    for (double alpha : alphas) {
        for (double gamma : gammas) {
            for (double decay : decays) {
                idx++;
                std::cout << "[" << idx << "/" << total_combinations << "] 테스트 중: "
                          << "α=" << alpha << ", γ=" << gamma << ", ε_decay=" << decay << "\n";

                std::vector<double> scores;
                for (int run = 0; run < 3; ++run) {
                    FrozenLakeEnv env(false);
                    QLearningAgent agent(env.get_num_states(), env.get_num_actions());
                    auto history = train_q_learning(env, agent, num_episodes, alpha, gamma, 1.0, 0.01, decay, 250);
                    double final_score = history.back().second;
                    scores.push_back(final_score);
                }

                double avg_score = std::accumulate(scores.begin(), scores.end(), 0.0) / scores.size();
                all_results.push_back({alpha, gamma, decay, avg_score});

                std::cout << "  평균 성공률: " << std::fixed << std::setprecision(2)
                          << (avg_score * 100) << "%\n";
            }
        }
    }

    std::sort(all_results.begin(), all_results.end());

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "그리드 서치 완료!\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << "최적 파라미터: α=" << all_results[0].alpha
              << ", γ=" << all_results[0].gamma
              << ", ε_decay=" << all_results[0].epsilon_decay << "\n";
    std::cout << "최고 성공률: " << std::fixed << std::setprecision(2)
              << (all_results[0].score * 100) << "%\n";

    return all_results;
}

int main() {
    std::cout << std::string(70, '=') << "\n";
    std::cout << "Q-Learning 하이퍼파라미터 실험\n";
    std::cout << std::string(70, '=') << "\n";

    FrozenLakeEnv env(false);
    std::cout << "\n환경: FrozenLake-v1 (is_slippery=false)\n";
    std::cout << "상태: " << env.get_num_states() << ", 행동: " << env.get_num_actions() << "\n\n";

    // Experiment 1: Alpha
    auto alpha_results = experiment_alpha({0.01, 0.05, 0.1, 0.3, 0.5}, 5000, 3);

    std::cout << "\n[분석] 학습률(α):\n";
    std::cout << "- α=0.01: 학습이 너무 느림 (안정적이지만 비효율적)\n";
    std::cout << "- α=0.1~0.3: 적절한 속도로 학습 (권장)\n";
    std::cout << "- α=0.5: 빠르지만 불안정할 수 있음 (진동)\n";

    // Experiment 2: Gamma
    auto gamma_results = experiment_gamma({0.5, 0.7, 0.9, 0.95, 0.99}, 5000, 3);

    std::cout << "\n[분석] 할인율(γ):\n";
    std::cout << "- γ=0.5: 단기 보상만 고려 (긴 경로 학습 어려움)\n";
    std::cout << "- γ=0.9: 중간 미래까지 고려 (10스텝 정도)\n";
    std::cout << "- γ=0.99: 먼 미래까지 고려 (100스텝, 권장)\n";

    // Experiment 3: Epsilon Decay
    auto decay_results = experiment_epsilon_decay({0.99, 0.995, 0.999}, 5000, 3);

    std::cout << "\n[분석] 탐험률 감소(ε_decay):\n";
    std::cout << "- decay=0.99: 빠른 수렴 (조기 수렴 위험)\n";
    std::cout << "- decay=0.995: 균형 잡힌 탐험 (권장)\n";
    std::cout << "- decay=0.999: 충분한 탐험 (느린 학습)\n";

    // Grid Search (optional)
    std::cout << "\n그리드 서치를 진행하시겠습니까? (y/n): " << std::flush;
    std::string response;
    std::getline(std::cin, response);

    if (response == "y" || response == "Y") {
        auto grid_results = grid_search(
            {0.05, 0.1, 0.2},
            {0.95, 0.99},
            {0.99, 0.995},
            5000
        );

        std::cout << "\n전체 결과 (성공률 높은 순):\n";
        std::cout << std::string(70, '-') << "\n";
        std::cout << std::left << std::setw(6) << "Rank"
                  << std::setw(8) << "Alpha"
                  << std::setw(8) << "Gamma"
                  << std::setw(8) << "Decay"
                  << std::setw(15) << "Success Rate" << "\n";
        std::cout << std::string(70, '-') << "\n";

        for (size_t i = 0; i < std::min(size_t(10), grid_results.size()); ++i) {
            std::cout << std::left << std::setw(6) << (i + 1)
                      << std::setw(8) << grid_results[i].alpha
                      << std::setw(8) << grid_results[i].gamma
                      << std::setw(8) << grid_results[i].epsilon_decay
                      << std::fixed << std::setprecision(2)
                      << (grid_results[i].score * 100) << "%\n";
        }
    }

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "최종 권장 하이퍼파라미터 (FrozenLake 기준)\n";
    std::cout << std::string(70, '=') << "\n";
    std::cout << R"(
    alpha = 0.1          // 학습률 (중간 속도)
    gamma = 0.99         // 할인율 (장기 계획)
    epsilon_start = 1.0  // 초기 탐험률 (완전 탐험)
    epsilon_min = 0.01   // 최소 탐험률 (약간 유지)
    epsilon_decay = 0.995  // 감소율 (균형)
)" << "\n";

    std::cout << "주의사항:\n";
    std::cout << "- 환경마다 최적 파라미터가 다를 수 있습니다\n";
    std::cout << "- 실험을 통해 자신의 환경에 맞는 값을 찾으세요\n";
    std::cout << "- 위 값은 시작점으로 좋은 기본값입니다\n";

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "실험 완료!\n";
    std::cout << std::string(70, '=') << "\n";

    return 0;
}
