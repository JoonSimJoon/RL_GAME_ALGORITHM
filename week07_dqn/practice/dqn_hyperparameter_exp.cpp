/*
DQN Hyperparameter Experiment (C++20)
Week 7 Practice Code

This code compares DQN performance with different hyperparameters.

Experiments:
1. Learning Rate (0.0001, 0.001, 0.01)
2. Target Update Frequency (100, 1000, 10000)
3. Batch Size (16, 32, 64, 128)
4. Replay Buffer Size (1000, 5000, 10000)
*/

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <deque>
#include <iomanip>
#include <fstream>
#include <map>
#include <string>

// ============================================================================
// Matrix Class (Same as dqn_cartpole.cpp)
// ============================================================================

class Matrix {
public:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    void xavier_init(std::mt19937& gen, size_t fan_in, size_t fan_out) {
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& row : data) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
    }

    static Matrix multiply(const Matrix& a, const Matrix& b) {
        Matrix result(a.rows, b.cols);
        for (size_t i = 0; i < a.rows; ++i) {
            for (size_t j = 0; j < b.cols; ++j) {
                for (size_t k = 0; k < a.cols; ++k) {
                    result.data[i][j] += a.data[i][k] * b.data[k][j];
                }
            }
        }
        return result;
    }

    void relu() {
        for (auto& row : data) {
            for (auto& val : row) {
                val = std::max(0.0, val);
            }
        }
    }

    void copy_from(const Matrix& other) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = other.data[i][j];
            }
        }
    }
};

// ============================================================================
// Q-Network (Same as dqn_cartpole.cpp)
// ============================================================================

class QNetwork {
public:
    Matrix w1, b1, w2, b2, w3, b3;
    Matrix a1, z2, a2, z3, a3;

    QNetwork(size_t state_size, size_t action_size, std::mt19937& gen)
        : w1(128, state_size), b1(128, 1), w2(128, 128), b2(128, 1),
          w3(action_size, 128), b3(action_size, 1),
          a1(state_size, 1), z2(128, 1), a2(128, 1), z3(128, 1), a3(128, 1) {
        w1.xavier_init(gen, state_size, 128);
        w2.xavier_init(gen, 128, 128);
        w3.xavier_init(gen, 128, action_size);
    }

    std::vector<double> forward(const std::vector<double>& state) {
        a1 = Matrix(state.size(), 1);
        for (size_t i = 0; i < state.size(); ++i) {
            a1.data[i][0] = state[i];
        }

        z2 = Matrix::multiply(w1, a1);
        for (size_t i = 0; i < z2.rows; ++i) {
            z2.data[i][0] += b1.data[i][0];
        }
        a2 = z2;
        a2.relu();

        z3 = Matrix::multiply(w2, a2);
        for (size_t i = 0; i < z3.rows; ++i) {
            z3.data[i][0] += b2.data[i][0];
        }
        a3 = z3;
        a3.relu();

        Matrix output = Matrix::multiply(w3, a3);
        for (size_t i = 0; i < output.rows; ++i) {
            output.data[i][0] += b3.data[i][0];
        }

        std::vector<double> q_values(output.rows);
        for (size_t i = 0; i < output.rows; ++i) {
            q_values[i] = output.data[i][0];
        }
        return q_values;
    }

    void copy_weights_from(const QNetwork& other) {
        w1.copy_from(other.w1);
        b1.copy_from(other.b1);
        w2.copy_from(other.w2);
        b2.copy_from(other.b2);
        w3.copy_from(other.w3);
        b3.copy_from(other.b3);
    }
};

// ============================================================================
// Replay Buffer
// ============================================================================

struct Experience {
    std::vector<double> state;
    int action;
    double reward;
    std::vector<double> next_state;
    bool done;
};

class ReplayBuffer {
private:
    std::deque<Experience> buffer;
    size_t capacity;

public:
    ReplayBuffer(size_t cap = 10000) : capacity(cap) {}

    void store(const std::vector<double>& state, int action, double reward,
               const std::vector<double>& next_state, bool done) {
        buffer.push_back({state, action, reward, next_state, done});
        if (buffer.size() > capacity) {
            buffer.pop_front();
        }
    }

    std::vector<Experience> sample(size_t batch_size, std::mt19937& gen) {
        std::vector<Experience> batch;
        std::uniform_int_distribution<size_t> dist(0, buffer.size() - 1);
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer[dist(gen)]);
        }
        return batch;
    }

    size_t size() const { return buffer.size(); }
};

// ============================================================================
// CartPole Environment
// ============================================================================

class CartPoleEnv {
private:
    std::mt19937 gen;
    double x, x_dot, theta, theta_dot;
    const double gravity = 9.8;
    const double cart_mass = 1.0;
    const double pole_mass = 0.1;
    const double total_mass = cart_mass + pole_mass;
    const double pole_length = 0.5;
    const double pole_mass_length = pole_mass * pole_length;
    const double force_mag = 10.0;
    const double tau = 0.02;
    const double theta_threshold = 12.0 * M_PI / 180.0;
    const double x_threshold = 2.4;

public:
    CartPoleEnv(unsigned seed = 42) : gen(seed) {}

    std::vector<double> reset() {
        std::uniform_real_distribution<double> dist(-0.05, 0.05);
        x = dist(gen);
        x_dot = dist(gen);
        theta = dist(gen);
        theta_dot = dist(gen);
        return {x, x_dot, theta, theta_dot};
    }

    std::tuple<std::vector<double>, double, bool> step(int action) {
        double force = (action == 1) ? force_mag : -force_mag;
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);

        double temp = (force + pole_mass_length * theta_dot * theta_dot * sin_theta) / total_mass;
        double theta_acc = (gravity * sin_theta - cos_theta * temp) /
                          (pole_length * (4.0/3.0 - pole_mass * cos_theta * cos_theta / total_mass));
        double x_acc = temp - pole_mass_length * theta_acc * cos_theta / total_mass;

        x += tau * x_dot;
        x_dot += tau * x_acc;
        theta += tau * theta_dot;
        theta_dot += tau * theta_acc;

        bool done = (x < -x_threshold || x > x_threshold ||
                    theta < -theta_threshold || theta > theta_threshold);
        double reward = done ? 0.0 : 1.0;

        return {{x, x_dot, theta, theta_dot}, reward, done};
    }
};

// ============================================================================
// DQN Agent
// ============================================================================

class DQNAgent {
private:
    size_t state_size, action_size;
    double gamma, lr;
    QNetwork q_network, target_network;
    ReplayBuffer replay_buffer;
    std::mt19937 gen;
    int step_count;

public:
    DQNAgent(size_t state_sz, size_t action_sz, double learning_rate, double discount,
             size_t buffer_capacity, unsigned seed = 42)
        : state_size(state_sz), action_size(action_sz), gamma(discount), lr(learning_rate),
          q_network(state_sz, action_sz, gen), target_network(state_sz, action_sz, gen),
          replay_buffer(buffer_capacity), gen(seed), step_count(0) {
        target_network.copy_weights_from(q_network);
    }

    int select_action(const std::vector<double>& state, double epsilon) {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        if (dist(gen) < epsilon) {
            std::uniform_int_distribution<int> action_dist(0, action_size - 1);
            return action_dist(gen);
        }
        auto q_values = q_network.forward(state);
        return std::max_element(q_values.begin(), q_values.end()) - q_values.begin();
    }

    double learn(size_t batch_size) {
        if (replay_buffer.size() < batch_size) return 0.0;

        auto batch = replay_buffer.sample(batch_size, gen);
        double total_loss = 0.0;

        for (const auto& exp : batch) {
            auto current_q_values = q_network.forward(exp.state);
            double current_q = current_q_values[exp.action];

            auto next_q_values = target_network.forward(exp.next_state);
            double max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
            double target_q = exp.reward + gamma * max_next_q * (exp.done ? 0.0 : 1.0);

            double loss = (current_q - target_q) * (current_q - target_q);
            total_loss += loss;

            double grad = 2.0 * (current_q - target_q) / batch_size;

            for (size_t i = 0; i < q_network.w3.rows; ++i) {
                for (size_t j = 0; j < q_network.w3.cols; ++j) {
                    double g = grad * q_network.a3.data[j][0];
                    q_network.w3.data[i][j] -= lr * g;
                }
            }
            for (size_t i = 0; i < q_network.b3.rows; ++i) {
                q_network.b3.data[i][0] -= lr * grad;
            }
        }

        return total_loss / batch_size;
    }

    void update_target_network() {
        target_network.copy_weights_from(q_network);
    }

    void store_experience(const std::vector<double>& state, int action, double reward,
                         const std::vector<double>& next_state, bool done) {
        replay_buffer.store(state, action, reward, next_state, done);
    }

    void increment_step() { step_count++; }
    int get_step_count() const { return step_count; }
};

// ============================================================================
// Training with Configuration
// ============================================================================

struct Config {
    double lr;
    double gamma;
    double epsilon_start;
    double epsilon_end;
    double epsilon_decay;
    size_t batch_size;
    int target_update_freq;
    size_t buffer_size;
};

std::vector<double> train_with_config(const Config& config, int num_episodes = 300, bool verbose = false) {
    CartPoleEnv env(42);
    DQNAgent agent(4, 2, config.lr, config.gamma, config.buffer_size, 42);

    double epsilon = config.epsilon_start;
    std::vector<double> episode_rewards;
    std::vector<double> moving_avg_rewards;

    for (int episode = 0; episode < num_episodes; ++episode) {
        auto state = env.reset();
        double episode_reward = 0.0;

        for (int t = 0; t < 500; ++t) {
            int action = agent.select_action(state, epsilon);
            auto [next_state, reward, done] = env.step(action);

            agent.store_experience(state, action, reward, next_state, done);
            agent.learn(config.batch_size);

            agent.increment_step();
            if (agent.get_step_count() % config.target_update_freq == 0) {
                agent.update_target_network();
            }

            episode_reward += reward;
            state = next_state;

            if (done) break;
        }

        epsilon = std::max(config.epsilon_end, epsilon * config.epsilon_decay);
        episode_rewards.push_back(episode_reward);

        int window = std::min(static_cast<int>(episode_rewards.size()), 100);
        double avg = std::accumulate(episode_rewards.end() - window,
                                     episode_rewards.end(), 0.0) / window;
        moving_avg_rewards.push_back(avg);

        if (verbose && (episode + 1) % 50 == 0) {
            std::cout << "  Episode " << std::setw(3) << (episode + 1)
                     << " | Avg: " << std::setw(6) << std::fixed << std::setprecision(2) << avg << "\n";
        }
    }

    return moving_avg_rewards;
}

// ============================================================================
// Experiment 1: Learning Rate
// ============================================================================

void experiment_learning_rate() {
    std::cout << "============================================================\n";
    std::cout << "Experiment 1: Learning Rate Comparison\n";
    std::cout << "============================================================\n";

    std::vector<double> learning_rates = {0.0001, 0.001, 0.01};
    std::map<double, std::vector<double>> results;

    Config base_config = {0.001, 0.99, 1.0, 0.01, 0.995, 32, 1000, 10000};

    for (double lr : learning_rates) {
        std::cout << "\nLearning Rate = " << lr << " training...\n";
        base_config.lr = lr;

        auto moving_avg = train_with_config(base_config, 300, true);
        results[lr] = moving_avg;

        std::cout << "  Final average: " << std::fixed << std::setprecision(2)
                 << moving_avg.back() << "\n";
    }

    // Save results
    std::ofstream out("exp1_learning_rate.txt");
    out << "Episode";
    for (double lr : learning_rates) {
        out << ",LR_" << lr;
    }
    out << "\n";

    size_t max_size = results[learning_rates[0]].size();
    for (size_t i = 0; i < max_size; ++i) {
        out << (i + 1);
        for (double lr : learning_rates) {
            out << "," << results[lr][i];
        }
        out << "\n";
    }
    out.close();

    std::cout << "\nGraph saved: exp1_learning_rate.txt\n";
}

// ============================================================================
// Experiment 2: Target Update Frequency
// ============================================================================

void experiment_target_update_freq() {
    std::cout << "\n============================================================\n";
    std::cout << "Experiment 2: Target Update Frequency Comparison\n";
    std::cout << "============================================================\n";

    std::vector<int> update_freqs = {100, 1000, 10000};
    std::map<int, std::vector<double>> results;

    Config base_config = {0.001, 0.99, 1.0, 0.01, 0.995, 32, 1000, 10000};

    for (int freq : update_freqs) {
        std::cout << "\nTarget Update Frequency = " << freq << " training...\n";
        base_config.target_update_freq = freq;

        auto moving_avg = train_with_config(base_config, 300, true);
        results[freq] = moving_avg;

        std::cout << "  Final average: " << std::fixed << std::setprecision(2)
                 << moving_avg.back() << "\n";
    }

    // Save results
    std::ofstream out("exp2_target_update_freq.txt");
    out << "Episode";
    for (int freq : update_freqs) {
        out << ",Freq_" << freq;
    }
    out << "\n";

    size_t max_size = results[update_freqs[0]].size();
    for (size_t i = 0; i < max_size; ++i) {
        out << (i + 1);
        for (int freq : update_freqs) {
            out << "," << results[freq][i];
        }
        out << "\n";
    }
    out.close();

    std::cout << "\nGraph saved: exp2_target_update_freq.txt\n";
}

// ============================================================================
// Experiment 3: Batch Size
// ============================================================================

void experiment_batch_size() {
    std::cout << "\n============================================================\n";
    std::cout << "Experiment 3: Batch Size Comparison\n";
    std::cout << "============================================================\n";

    std::vector<size_t> batch_sizes = {16, 32, 64, 128};
    std::map<size_t, std::vector<double>> results;

    Config base_config = {0.001, 0.99, 1.0, 0.01, 0.995, 32, 1000, 10000};

    for (size_t batch : batch_sizes) {
        std::cout << "\nBatch Size = " << batch << " training...\n";
        base_config.batch_size = batch;

        auto moving_avg = train_with_config(base_config, 300, true);
        results[batch] = moving_avg;

        std::cout << "  Final average: " << std::fixed << std::setprecision(2)
                 << moving_avg.back() << "\n";
    }

    // Save results
    std::ofstream out("exp3_batch_size.txt");
    out << "Episode";
    for (size_t batch : batch_sizes) {
        out << ",Batch_" << batch;
    }
    out << "\n";

    size_t max_size = results[batch_sizes[0]].size();
    for (size_t i = 0; i < max_size; ++i) {
        out << (i + 1);
        for (size_t batch : batch_sizes) {
            out << "," << results[batch][i];
        }
        out << "\n";
    }
    out.close();

    std::cout << "\nGraph saved: exp3_batch_size.txt\n";
}

// ============================================================================
// Experiment 4: Buffer Size
// ============================================================================

void experiment_buffer_size() {
    std::cout << "\n============================================================\n";
    std::cout << "Experiment 4: Replay Buffer Size Comparison\n";
    std::cout << "============================================================\n";

    std::vector<size_t> buffer_sizes = {1000, 5000, 10000};
    std::map<size_t, std::vector<double>> results;

    Config base_config = {0.001, 0.99, 1.0, 0.01, 0.995, 32, 1000, 10000};

    for (size_t buffer : buffer_sizes) {
        std::cout << "\nBuffer Size = " << buffer << " training...\n";
        base_config.buffer_size = buffer;

        auto moving_avg = train_with_config(base_config, 300, true);
        results[buffer] = moving_avg;

        std::cout << "  Final average: " << std::fixed << std::setprecision(2)
                 << moving_avg.back() << "\n";
    }

    // Save results
    std::ofstream out("exp4_buffer_size.txt");
    out << "Episode";
    for (size_t buffer : buffer_sizes) {
        out << ",Buffer_" << buffer;
    }
    out << "\n";

    size_t max_size = results[buffer_sizes[0]].size();
    for (size_t i = 0; i < max_size; ++i) {
        out << (i + 1);
        for (size_t buffer : buffer_sizes) {
            out << "," << results[buffer][i];
        }
        out << "\n";
    }
    out.close();

    std::cout << "\nGraph saved: exp4_buffer_size.txt\n";
}

// ============================================================================
// Main Function
// ============================================================================

int main(int argc, char* argv[]) {
    std::cout << "\n";
    std::cout << "    ========================================================\n";
    std::cout << "              DQN Hyperparameter Experiment Program       \n";
    std::cout << "    ========================================================\n";
    std::cout << "\n";

    if (argc > 1) {
        std::string exp_name = argv[1];
        std::cout << "\nRunning experiment '" << exp_name << "'...\n\n";

        if (exp_name == "lr") {
            experiment_learning_rate();
        } else if (exp_name == "freq") {
            experiment_target_update_freq();
        } else if (exp_name == "batch") {
            experiment_batch_size();
        } else if (exp_name == "buffer") {
            experiment_buffer_size();
        } else {
            std::cout << "Invalid experiment name: " << exp_name << "\n";
            std::cout << "Available options: lr, freq, batch, buffer\n";
        }
    } else {
        std::cout << "Usage:\n";
        std::cout << "  All experiments:     ./dqn_hyperparameter_exp\n";
        std::cout << "  Single experiment:   ./dqn_hyperparameter_exp [lr|freq|batch|buffer]\n";
        std::cout << "\nRunning all experiments...\n\n";

        experiment_learning_rate();
        experiment_target_update_freq();
        experiment_batch_size();
        experiment_buffer_size();

        std::cout << "\n============================================================\n";
        std::cout << "All experiments completed!\n";
        std::cout << "============================================================\n";
    }

    return 0;
}
