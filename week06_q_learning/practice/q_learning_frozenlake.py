"""
Q-Learning을 사용한 FrozenLake 학습

FrozenLake 환경을 구현하고 Q-Learning 알고리즘을 적용합니다.

실습 목표:
1. Q-table 기반 Q-Learning 구현
2. ε-greedy 탐험 전략 이해
3. 학습 곡선 관찰
4. 학습된 정책 시각화
"""

import random
import csv
from collections import deque


class FrozenLakeEnv:
    """FrozenLake 4x4 환경"""

    GRID_SIZE = 4
    NUM_STATES = 16
    NUM_ACTIONS = 4  # LEFT=0, DOWN=1, RIGHT=2, UP=3

    # SFFF / FHFH / FFFH / HFFG
    GRID = "SFFFFHFHFFFHHFFG"

    def __init__(self, is_slippery=False, seed=None):
        self.is_slippery = is_slippery
        self.rng = random.Random(seed)
        self.current_state = 0

    def reset(self):
        self.current_state = 0
        return self.current_state

    def step(self, action):
        if self.GRID[self.current_state] in ("H", "G"):
            return self.current_state, 0.0, True

        actual_action = action
        if self.is_slippery:
            p = self.rng.random()
            if p < 0.333:
                actual_action = (action - 1) % self.NUM_ACTIONS
            elif p >= 0.666:
                actual_action = (action + 1) % self.NUM_ACTIONS

        row = self.current_state // self.GRID_SIZE
        col = self.current_state % self.GRID_SIZE

        if actual_action == 0:
            col = max(0, col - 1)
        elif actual_action == 1:
            row = min(self.GRID_SIZE - 1, row + 1)
        elif actual_action == 2:
            col = min(self.GRID_SIZE - 1, col + 1)
        elif actual_action == 3:
            row = max(0, row - 1)

        self.current_state = row * self.GRID_SIZE + col

        reward = 0.0
        terminated = False
        if self.GRID[self.current_state] == "G":
            reward = 1.0
            terminated = True
        elif self.GRID[self.current_state] == "H":
            terminated = True

        return self.current_state, reward, terminated


class QLearningAgent:
    """Q-Learning 에이전트"""

    def __init__(self, num_states, num_actions, seed=None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.Q = [[0.0] * num_actions for _ in range(num_states)]
        self.rng = random.Random(seed)

    def epsilon_greedy(self, state, epsilon):
        if self.rng.random() < epsilon:
            return self.rng.randint(0, self.num_actions - 1)
        return max(range(self.num_actions), key=lambda a: self.Q[state][a])

    def greedy(self, state):
        return max(range(self.num_actions), key=lambda a: self.Q[state][a])

    def update(self, state, action, reward, next_state, done, alpha, gamma):
        best_next_q = 0.0 if done else max(self.Q[next_state])
        td_target = reward + gamma * best_next_q
        td_error = td_target - self.Q[state][action]
        self.Q[state][action] += alpha * td_error

    def print_policy(self):
        symbols = ["←", "↓", "→", "↑"]
        print("\n학습된 정책 (화살표):")
        for i in range(4):
            row = ""
            for j in range(4):
                state = i * 4 + j
                row += symbols[self.greedy(state)] + " "
            print(row)

    def print_q_table(self, num_states_to_show=5):
        print(f"\nQ-table (상위 {num_states_to_show}개 상태):")
        print("State | Left   Down   Right  Up")
        print("-" * 40)
        for state in range(min(num_states_to_show, self.num_states)):
            vals = "  ".join(f"{self.Q[state][a]:6.2f}" for a in range(self.num_actions))
            print(f"{state:5d} | {vals}")


def evaluate_policy(env, agent, num_episodes=100):
    success = 0
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.greedy(state)
            state, reward, done = env.step(action)
            if reward == 1.0:
                success += 1
    return success / num_episodes


def train_q_learning(
    env,
    agent,
    num_episodes=10000,
    alpha=0.1,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_min=0.01,
    epsilon_decay=0.995,
    eval_interval=500,
    verbose=True,
):
    if verbose:
        print("Q-Learning 학습 시작")
        print(f"상태 개수: {env.NUM_STATES}, 행동 개수: {env.NUM_ACTIONS}")
        print(f"하이퍼파라미터: α={alpha}, γ={gamma}, ε={epsilon_start}→{epsilon_min}")
        print("-" * 70)

    epsilon = epsilon_start
    rewards_history = []
    success_history = []
    recent = deque(maxlen=100)

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.epsilon_greedy(state, epsilon)
            next_state, reward, terminated = env.step(action)
            agent.update(state, action, reward, next_state, terminated, alpha, gamma)
            state = next_state
            total_reward += reward
            done = terminated

        rewards_history.append(total_reward)
        recent.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_policy(env, agent, 100)
            success_history.append((episode + 1, success_rate))
            if verbose:
                avg_reward = sum(recent) / len(recent)
                print(
                    f"Episode {episode + 1:5d} | ε={epsilon:.3f} | "
                    f"Avg Reward={avg_reward:.3f} | Success Rate={success_rate * 100:.1f}%"
                )

    if verbose:
        print("-" * 70)
        print("학습 완료!")

    return rewards_history, success_history


def save_results(rewards_history, filename):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Reward"])
        for i, r in enumerate(rewards_history):
            writer.writerow([i + 1, r])
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    print("=" * 70)
    print("FrozenLake Q-Learning 실습")
    print("=" * 70)

    env = FrozenLakeEnv(is_slippery=False)

    print("\n환경 정보:")
    print(f"  상태 공간: {env.NUM_STATES} (4x4 격자)")
    print(f"  행동 공간: {env.NUM_ACTIONS} (←↓→↑)")
    print("  is_slippery: false (결정적 환경)\n")

    agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)

    rewards_history, success_history = train_q_learning(
        env,
        agent,
        num_episodes=10000,
        alpha=0.1,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        eval_interval=500,
        verbose=True,
    )

    print("\n=== 최종 성능 평가 ===")
    final_success = evaluate_policy(env, agent, 1000)
    print(f"1000번 테스트 성공률: {final_success * 100:.2f}%")

    agent.print_policy()
    agent.print_q_table(5)
    save_results(rewards_history, "q_learning_results.csv")

    print("\n" + "=" * 70)
    print("실습 완료!")
    print("=" * 70)
    print("\n추가 실험 아이디어:")
    print("1. is_slippery=True로 변경하여 확률적 환경에서 학습")
    print("2. alpha를 0.01, 0.3, 0.5로 바꿔가며 학습 속도 비교")
    print("3. gamma를 0.5, 0.9로 바꿔가며 장기 계획의 중요성 확인")
    print("4. epsilon_decay를 0.99, 0.999로 바꿔가며 탐험 영향 확인")
