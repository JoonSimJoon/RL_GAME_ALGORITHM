"""
하이퍼파라미터 실험
Q-Learning의 핵심 하이퍼파라미터(α, γ, ε)를 변화시키며
학습 성능에 미치는 영향을 분석합니다.

실험 하이퍼파라미터:
1. α (학습률): 새로운 정보를 얼마나 빠르게 받아들일지
2. γ (할인율): 미래 보상을 얼마나 중요하게 볼지
3. ε_decay (탐험률 감소): 탐험을 얼마나 빠르게 줄일지

실습 목표:
1. 각 하이퍼파라미터의 역할 이해
2. 하이퍼파라미터 변화가 학습에 미치는 영향 관찰
3. 최적 하이퍼파라미터 조합 찾기
"""

import random
from collections import defaultdict


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


class QLearningAgent:
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


def train_q_learning(env, agent, num_episodes, alpha, gamma, epsilon_start, epsilon_min, epsilon_decay, eval_interval=250):
    success_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.epsilon_greedy(state, epsilon)
            next_state, reward, terminated = env.step(action)
            agent.update(state, action, reward, next_state, terminated, alpha, gamma)
            state = next_state
            done = terminated
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_policy(env, agent, 100)
            success_history.append((episode + 1, success_rate))

    return success_history


def experiment_alpha(alphas, num_episodes=5000, num_runs=3):
    print("=" * 70)
    print("실험 1: 학습률(α) 변화")
    print("=" * 70)
    print(f"테스트할 α 값: {alphas}")
    print(f"각 설정당 {num_runs}회 반복\n")

    results = {}
    for alpha in alphas:
        print(f"α = {alpha} 실험 중... ", end="", flush=True)
        avg_history = defaultdict(list)

        for _ in range(num_runs):
            env = FrozenLakeEnv(is_slippery=False)
            agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)
            history = train_q_learning(env, agent, num_episodes, alpha, 0.99, 1.0, 0.01, 0.995, 250)
            for ep, rate in history:
                avg_history[ep].append(rate)

        results[alpha] = [(ep, sum(rates) / len(rates)) for ep, rates in sorted(avg_history.items())]
        print(f"완료! 최종 성공률: {results[alpha][-1][1] * 100:.1f}%")

    return results


def experiment_gamma(gammas, num_episodes=5000, num_runs=3):
    print("\n" + "=" * 70)
    print("실험 2: 할인율(γ) 변화")
    print("=" * 70)
    print(f"테스트할 γ 값: {gammas}")
    print(f"각 설정당 {num_runs}회 반복\n")

    results = {}
    for gamma in gammas:
        print(f"γ = {gamma} 실험 중... ", end="", flush=True)
        avg_history = defaultdict(list)

        for _ in range(num_runs):
            env = FrozenLakeEnv(is_slippery=False)
            agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)
            history = train_q_learning(env, agent, num_episodes, 0.1, gamma, 1.0, 0.01, 0.995, 250)
            for ep, rate in history:
                avg_history[ep].append(rate)

        results[gamma] = [(ep, sum(rates) / len(rates)) for ep, rates in sorted(avg_history.items())]
        print(f"완료! 최종 성공률: {results[gamma][-1][1] * 100:.1f}%")

    return results


def experiment_epsilon_decay(decays, num_episodes=5000, num_runs=3):
    print("\n" + "=" * 70)
    print("실험 3: 탐험률 감소율(ε_decay) 변화")
    print("=" * 70)
    print(f"테스트할 ε_decay 값: {decays}")
    print(f"각 설정당 {num_runs}회 반복\n")

    results = {}
    for decay in decays:
        print(f"ε_decay = {decay} 실험 중... ", end="", flush=True)
        avg_history = defaultdict(list)

        for _ in range(num_runs):
            env = FrozenLakeEnv(is_slippery=False)
            agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)
            history = train_q_learning(env, agent, num_episodes, 0.1, 0.99, 1.0, 0.01, decay, 250)
            for ep, rate in history:
                avg_history[ep].append(rate)

        results[decay] = [(ep, sum(rates) / len(rates)) for ep, rates in sorted(avg_history.items())]
        print(f"완료! 최종 성공률: {results[decay][-1][1] * 100:.1f}%")

    return results


def grid_search(alphas, gammas, decays, num_episodes=5000):
    print("\n" + "=" * 70)
    print("그리드 서치: 최적 하이퍼파라미터 찾기")
    print("=" * 70)

    total = len(alphas) * len(gammas) * len(decays)
    print(f"총 {total}개 조합 테스트")
    print("-" * 70)

    all_results = []
    idx = 0

    for alpha in alphas:
        for gamma in gammas:
            for decay in decays:
                idx += 1
                print(f"[{idx}/{total}] 테스트 중: α={alpha}, γ={gamma}, ε_decay={decay}")

                scores = []
                for _ in range(3):
                    env = FrozenLakeEnv(is_slippery=False)
                    agent = QLearningAgent(env.NUM_STATES, env.NUM_ACTIONS)
                    history = train_q_learning(env, agent, num_episodes, alpha, gamma, 1.0, 0.01, decay, 250)
                    scores.append(history[-1][1])

                avg_score = sum(scores) / len(scores)
                all_results.append((alpha, gamma, decay, avg_score))
                print(f"  평균 성공률: {avg_score * 100:.2f}%")

    all_results.sort(key=lambda x: -x[3])

    print("\n" + "=" * 70)
    print("그리드 서치 완료!")
    print("=" * 70)
    best = all_results[0]
    print(f"최적 파라미터: α={best[0]}, γ={best[1]}, ε_decay={best[2]}")
    print(f"최고 성공률: {best[3] * 100:.2f}%")

    return all_results


if __name__ == "__main__":
    print("=" * 70)
    print("Q-Learning 하이퍼파라미터 실험")
    print("=" * 70)

    env = FrozenLakeEnv(is_slippery=False)
    print(f"\n환경: FrozenLake-v1 (is_slippery=false)")
    print(f"상태: {env.NUM_STATES}, 행동: {env.NUM_ACTIONS}\n")

    # 실험 1: Alpha
    alpha_results = experiment_alpha([0.01, 0.05, 0.1, 0.3, 0.5], 5000, 3)

    print("\n[분석] 학습률(α):")
    print("- α=0.01: 학습이 너무 느림 (안정적이지만 비효율적)")
    print("- α=0.1~0.3: 적절한 속도로 학습 (권장)")
    print("- α=0.5: 빠르지만 불안정할 수 있음 (진동)")

    # 실험 2: Gamma
    gamma_results = experiment_gamma([0.5, 0.7, 0.9, 0.95, 0.99], 5000, 3)

    print("\n[분석] 할인율(γ):")
    print("- γ=0.5: 단기 보상만 고려 (긴 경로 학습 어려움)")
    print("- γ=0.9: 중간 미래까지 고려 (10스텝 정도)")
    print("- γ=0.99: 먼 미래까지 고려 (100스텝, 권장)")

    # 실험 3: Epsilon Decay
    decay_results = experiment_epsilon_decay([0.99, 0.995, 0.999], 5000, 3)

    print("\n[분석] 탐험률 감소(ε_decay):")
    print("- decay=0.99: 빠른 수렴 (조기 수렴 위험)")
    print("- decay=0.995: 균형 잡힌 탐험 (권장)")
    print("- decay=0.999: 충분한 탐험 (느린 학습)")

    # 그리드 서치
    response = input("\n그리드 서치를 진행하시겠습니까? (y/n): ").strip()

    if response.lower() == "y":
        grid_results = grid_search([0.05, 0.1, 0.2], [0.95, 0.99], [0.99, 0.995], 5000)

        print("\n전체 결과 (성공률 높은 순):")
        print("-" * 70)
        print(f"{'Rank':<6}{'Alpha':<8}{'Gamma':<8}{'Decay':<8}{'Success Rate':<15}")
        print("-" * 70)

        for i, (alpha, gamma, decay, score) in enumerate(grid_results[:10]):
            print(f"{i + 1:<6}{alpha:<8}{gamma:<8}{decay:<8}{score * 100:.2f}%")

    print("\n" + "=" * 70)
    print("최종 권장 하이퍼파라미터 (FrozenLake 기준)")
    print("=" * 70)
    print("""
    alpha = 0.1          # 학습률 (중간 속도)
    gamma = 0.99         # 할인율 (장기 계획)
    epsilon_start = 1.0  # 초기 탐험률 (완전 탐험)
    epsilon_min = 0.01   # 최소 탐험률 (약간 유지)
    epsilon_decay = 0.995  # 감소율 (균형)
""")

    print("주의사항:")
    print("- 환경마다 최적 파라미터가 다를 수 있습니다")
    print("- 실험을 통해 자신의 환경에 맞는 값을 찾으세요")
    print("- 위 값은 시작점으로 좋은 기본값입니다")

    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)
