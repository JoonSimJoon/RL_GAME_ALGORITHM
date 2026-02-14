/*
CartPole DQN Complete Implementation (C++20)
Week 7 Practice Code

This code implements DQN learning in the CartPole environment.
Goal: Keep the pole upright for 200+ timesteps

Key Components:
1. QNetwork: Neural network to approximate Q-values
2. ReplayBuffer: Store and sample experiences
3. DQNAgent: Action selection and learning
4. Training Loop: Execute episodes and train agent
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

// ============================================================================
// Matrix Class for Neural Network
// ============================================================================

class Matrix {
public:
    std::vector<std::vector<double>> data;
    size_t rows, cols;

    Matrix(size_t r, size_t c) : rows(r), cols(c) {
        data.resize(rows, std::vector<double>(cols, 0.0));
    }

    // Xavier initialization
    void xavier_init(std::mt19937& gen, size_t fan_in, size_t fan_out) {
        double limit = std::sqrt(6.0 / (fan_in + fan_out));
        std::uniform_real_distribution<double> dist(-limit, limit);
        for (auto& row : data) {
            for (auto& val : row) {
                val = dist(gen);
            }
        }
    }

    // Matrix multiplication
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

    // Element-wise operations
    void add(const Matrix& other) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] += other.data[i][j];
            }
        }
    }

    void scale(double scalar) {
        for (auto& row : data) {
            for (auto& val : row) {
                val *= scalar;
            }
        }
    }

    // ReLU activation
    void relu() {
        for (auto& row : data) {
            for (auto& val : row) {
                val = std::max(0.0, val);
            }
        }
    }

    // ReLU derivative (for backprop)
    Matrix relu_derivative() const {
        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] > 0 ? 1.0 : 0.0;
            }
        }
        return result;
    }

    // Copy
    void copy_from(const Matrix& other) {
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                data[i][j] = other.data[i][j];
            }
        }
    }
};

// ============================================================================
// Q-Network: Approximates Q-values with a neural network
// ============================================================================

class QNetwork {
public:
    // Network architecture: state_size -> 128 -> 128 -> action_size
    Matrix w1, b1, w2, b2, w3, b3;

    // Cached activations for backprop
    Matrix a1, z2, a2, z3, a3;

    QNetwork(size_t state_size, size_t action_size, std::mt19937& gen)
        : w1(128, state_size), b1(128, 1),
          w2(128, 128), b2(128, 1),
          w3(action_size, 128), b3(action_size, 1),
          a1(state_size, 1), z2(128, 1), a2(128, 1),
          z3(128, 1), a3(128, 1) {
        // Xavier initialization
        w1.xavier_init(gen, state_size, 128);
        w2.xavier_init(gen, 128, 128);
        w3.xavier_init(gen, 128, action_size);
    }

    // Forward pass
    std::vector<double> forward(const std::vector<double>& state) {
        // Input layer
        a1 = Matrix(state.size(), 1);
        for (size_t i = 0; i < state.size(); ++i) {
            a1.data[i][0] = state[i];
        }

        // Hidden layer 1
        z2 = Matrix::multiply(w1, a1);
        for (size_t i = 0; i < z2.rows; ++i) {
            z2.data[i][0] += b1.data[i][0];
        }
        a2 = z2;
        a2.relu();

        // Hidden layer 2
        z3 = Matrix::multiply(w2, a2);
        for (size_t i = 0; i < z3.rows; ++i) {
            z3.data[i][0] += b2.data[i][0];
        }
        a3 = z3;
        a3.relu();

        // Output layer
        Matrix output = Matrix::multiply(w3, a3);
        for (size_t i = 0; i < output.rows; ++i) {
            output.data[i][0] += b3.data[i][0];
        }

        // Return Q-values
        std::vector<double> q_values(output.rows);
        for (size_t i = 0; i < output.rows; ++i) {
            q_values[i] = output.data[i][0];
        }
        return q_values;
    }

    // Copy weights from another network
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
// Replay Buffer: Stores and samples experiences
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
// CartPole Environment (Simple Physics Simulation)
// ============================================================================

class CartPoleEnv {
private:
    std::mt19937 gen;
    double x, x_dot, theta, theta_dot;

    // Physics constants
    const double gravity = 9.8;
    const double cart_mass = 1.0;
    const double pole_mass = 0.1;
    const double total_mass = cart_mass + pole_mass;
    const double pole_length = 0.5;
    const double pole_mass_length = pole_mass * pole_length;
    const double force_mag = 10.0;
    const double tau = 0.02;  // timestep

    // Thresholds
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

    // Adam optimizer state
    Matrix m_w1, v_w1, m_w2, v_w2, m_w3, v_w3;
    Matrix m_b1, v_b1, m_b2, v_b2, m_b3, v_b3;
    double beta1 = 0.9, beta2 = 0.999, epsilon_adam = 1e-8;
    int adam_t = 0;

public:
    DQNAgent(size_t state_sz, size_t action_sz, double learning_rate = 0.001,
             double discount = 0.99, unsigned seed = 42)
        : state_size(state_sz), action_size(action_sz), gamma(discount),
          lr(learning_rate), q_network(state_sz, action_sz, gen),
          target_network(state_sz, action_sz, gen), gen(seed), step_count(0),
          m_w1(128, state_sz), v_w1(128, state_sz), m_w2(128, 128), v_w2(128, 128),
          m_w3(action_sz, 128), v_w3(action_sz, 128), m_b1(128, 1), v_b1(128, 1),
          m_b2(128, 1), v_b2(128, 1), m_b3(action_sz, 1), v_b3(action_sz, 1) {
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

        // Simple gradient descent (approximation for batch)
        for (const auto& exp : batch) {
            // Forward pass
            auto current_q_values = q_network.forward(exp.state);
            double current_q = current_q_values[exp.action];

            // Target
            auto next_q_values = target_network.forward(exp.next_state);
            double max_next_q = *std::max_element(next_q_values.begin(), next_q_values.end());
            double target_q = exp.reward + gamma * max_next_q * (exp.done ? 0.0 : 1.0);

            // Loss
            double loss = (current_q - target_q) * (current_q - target_q);
            total_loss += loss;

            // Gradient (simplified backprop)
            double grad = 2.0 * (current_q - target_q) / batch_size;

            // Update weights (simplified Adam update on output layer)
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
    size_t buffer_size() const { return replay_buffer.size(); }
};

// ============================================================================
// Training Function
// ============================================================================

std::pair<std::vector<double>, std::vector<double>> train_dqn(
    int num_episodes = 500,
    int max_steps = 500,
    size_t batch_size = 32,
    double epsilon_start = 1.0,
    double epsilon_end = 0.01,
    double epsilon_decay = 0.995,
    int target_update_freq = 1000)
{
    std::cout << "============================================================\n";
    std::cout << "DQN Training Start\n";
    std::cout << "============================================================\n";
    std::cout << "Episodes: " << num_episodes << "\n";
    std::cout << "Batch Size: " << batch_size << "\n";
    std::cout << "Target Update Freq: " << target_update_freq << "\n";
    std::cout << "============================================================\n";

    CartPoleEnv env(42);
    DQNAgent agent(4, 2, 0.001, 0.99, 42);

    double epsilon = epsilon_start;
    std::vector<double> episode_rewards;
    std::vector<double> moving_avg_rewards;

    for (int episode = 0; episode < num_episodes; ++episode) {
        auto state = env.reset();
        double episode_reward = 0.0;

        for (int t = 0; t < max_steps; ++t) {
            int action = agent.select_action(state, epsilon);
            auto [next_state, reward, done] = env.step(action);

            agent.store_experience(state, action, reward, next_state, done);
            agent.learn(batch_size);

            agent.increment_step();
            if (agent.get_step_count() % target_update_freq == 0) {
                agent.update_target_network();
            }

            episode_reward += reward;
            state = next_state;

            if (done) break;
        }

        epsilon = std::max(epsilon_end, epsilon * epsilon_decay);
        episode_rewards.push_back(episode_reward);

        // Moving average (last 100 episodes)
        int window = std::min(static_cast<int>(episode_rewards.size()), 100);
        double avg = std::accumulate(episode_rewards.end() - window,
                                     episode_rewards.end(), 0.0) / window;
        moving_avg_rewards.push_back(avg);

        if ((episode + 1) % 10 == 0) {
            std::cout << "Episode " << std::setw(4) << (episode + 1)
                     << " | Reward: " << std::setw(6) << std::fixed << std::setprecision(2)
                     << episode_reward
                     << " | Avg (100): " << std::setw(6) << avg
                     << " | epsilon: " << std::setw(6) << std::setprecision(4) << epsilon
                     << " | Buffer: " << std::setw(5) << agent.buffer_size() << "\n";
        }

        if (avg >= 195.0) {
            std::cout << "\n============================================================\n";
            std::cout << "Goal achieved! Episode " << (episode + 1)
                     << " with average " << avg << "!\n";
            std::cout << "============================================================\n";
            break;
        }
    }

    std::cout << "\nTraining completed!\n";
    std::cout << "Final 100-episode average: " << moving_avg_rewards.back() << "\n";

    return {episode_rewards, moving_avg_rewards};
}

// ============================================================================
// Main Function
// ============================================================================

int main() {
    auto [episode_rewards, moving_avg_rewards] = train_dqn(
        500,   // num_episodes
        500,   // max_steps
        32,    // batch_size
        1.0,   // epsilon_start
        0.01,  // epsilon_end
        0.995, // epsilon_decay
        1000   // target_update_freq
    );

    // Save results
    std::ofstream out("dqn_cartpole_results.txt");
    out << "Episode,Reward,MovingAvg\n";
    for (size_t i = 0; i < episode_rewards.size(); ++i) {
        out << (i + 1) << "," << episode_rewards[i] << ","
            << moving_avg_rewards[i] << "\n";
    }
    out.close();

    std::cout << "\n============================================================\n";
    std::cout << "All tasks completed!\n";
    std::cout << "Results saved to: dqn_cartpole_results.txt\n";
    std::cout << "============================================================\n";

    return 0;
}
