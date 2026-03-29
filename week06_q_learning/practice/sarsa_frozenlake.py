"""
SARSA를 사용한 FrozenLake 학습
Q-Learning과 비교하여 On-policy 알고리즘인 SARSA를 구현합니다.

핵심 차이점:
- Q-Learning: max Q(s',a')를 사용 (Off-policy)
- SARSA: Q(s',a')를 사용 (On-policy, 실제로 선택한 행동)

실습 목표:
1. SARSA 알고리즘 구현
2. Q-Learning과의 차이점 이해
3. 두 알고리즘의 학습 곡선 비교
"""

import random
from collections import deque


class FrozenLakeEnv:
    GRID_SIZE = 4
    NUM_STATES = 16
    NUM_ACTIONS = 4

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


class Agent:
    """Base Agent"""

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

    def print_policy(self):
        symbols = ["←", "↓", "→", "↑"]
        print("\n학습된 정책 (화살표):")
        for i in range(4):
            row = ""
            for j in range(4):
                state = i * 4 + j
                row += symbols[self.greedy(state)] + " "
            print(row)


class SARSAAgent(Agent):
    """SARSA Agent (On-policy)"""

    def update(self, state, action, reward, next_state, next_action, done, alpha, gamma):
        # SARSA: Q(s,a) ← Q(s,a) + α[R + γ·Q(s',a') - Q(s,a)]
        next_q = 0.0 if done else self.Q[next_state][next_action]
        td_target = reward + gamma * next_q
        self.Q[state][action] += alpha * (td_target - self.Q[state][action])


class QLearningAgent(Agent):
    """Q-Learning Agent (Off-policy)"""

    def update(self, state, action, reward, next_state, next_action, done, alpha, gamma):
        # Q-Learning: Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
        best_next_q = 0.0 if done else max(self.Q[next_state])
        td_target = reward + gamma * best_next_q
        self.Q[state][action] += alpha * (td_target - self.Q[state][action])


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


def train_sarsa(
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
        print("SARSA 학습 시작")
        print(f"상태 개수: {env.NUM_STATES}, 행동 개수: {env.NUM_ACTIONS}")
        print(f"하이퍼파라미터: α={alpha}, γ={gamma}, ε={epsilon_start}→{epsilon_min}")
        print("-" * 70)

    epsilon = epsilon_start
    rewards_history = []
    success_history = []
    recent = deque(maxlen=100)

    for episode in range(num_episodes):
        state = env.reset()
        action = agent.epsilon_greedy(state, epsilon)  # SARSA: 첫 행동 선택
        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, terminated = env.step(action)
            next_action = agent.epsilon_greedy(next_state, epsilon)  # SARSA: 다음 행동
            agent.update(state, action, reward, next_state, next_action, terminated, alpha, gamma)
            state = next_state
            action = next_action
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
            agent.update(state, action, reward, next_state, 0, terminated, alpha, gamma)
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


if __name__ == "__main__":
    print("=" * 70)
    print("SARSA vs Q-Learning 비교 실습")
    print("=" * 70)

    env = FrozenLakeEnv(is_slippery=False)

    print("\n환경 정보:")
    print(f"  상태 공간: {env.NUM_STATES}")
    print(f"  행동 공간: {env.NUM_ACTIONS}")
    print("  is_slippery: false\n")

    alpha = 0.1
    gamma = 0.99
    epsilon_start = 1.0
    epsilon_min = 0.01
    epsilon_decay = 0.995
    eval_interval = 500

    # 1. SARSA
    print("=" * 70)
    print("1. SARSA 학습")
    print("=" * 70)

    sarsa_agent = SARSAAgent(env.NUM_STATES, env.NUM_ACTIONS)
    train_sarsa(env, sarsa_agent, 10000, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, eval_interval)

    final_sarsa = evaluate_policy(env, sarsa_agent, 1000)
    print(f"\nSARSA 최종 성공률: {final_sarsa * 100:.2f}%")
    print("\nSARSA 학습된 정책:")
    sarsa_agent.print_policy()

    # 2. Q-Learning
    print("\n" + "=" * 70)
    print("2. Q-Learning 학습")
    print("=" * 70)

    qlearn_agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)
    train_q_learning(env, qlearn_agent, 10000, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, eval_interval)

    final_qlearn = evaluate_policy(env, qlearn_agent, 1000)
    print(f"\nQ-Learning 최종 성공률: {final_qlearn * 100:.2f}%")
    print("\nQ-Learning 학습된 정책:")
    qlearn_agent.print_policy()

    # 3. 분석
    print("\n" + "=" * 70)
    print("분석 결과")
    print("=" * 70)
    print("\nFrozenLake (is_slippery=false) 환경에서:")
    print("- Q-Learning과 SARSA의 성능이 매우 유사합니다")
    print("- 두 알고리즘 모두 최적 정책을 잘 학습합니다\n")
    print("이유:")
    print("- 결정적 환경이라 탐험 중 실수가 적음")
    print("- 간단한 환경이라 차이가 두드러지지 않음\n")
    print("차이가 나는 경우:")
    print("- is_slippery=true (확률적 환경)")
    print("- 위험한 상태가 많은 환경 (절벽 문제 등)")
    print("- SARSA가 더 보수적이고 안전한 정책 학습")

    print("\n" + "=" * 70)
    print("실습 완료!")
    print("=" * 70)
    print("\n추가 실험 아이디어:")
    print("1. is_slippery=True로 변경하여 확률적 환경에서 비교")
    print("2. epsilon_min을 0.1로 높여서 탐험이 많을 때 차이 확인")
    print("3. gamma를 낮춰서 (0.5) 단기 보상 중시 시 차이 확인")
