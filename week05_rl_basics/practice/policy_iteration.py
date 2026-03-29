"""
Policy Iteration 구현
정책 평가와 정책 개선을 번갈아 수행하여 최적 정책을 찾는 알고리즘
"""

import math
from gridworld import GridWorld


def policy_evaluation(env, policy, gamma=0.9, theta=0.001, max_iterations=1000):
    """정책 평가: 주어진 정책의 가치 함수 계산"""
    V = {state: 0.0 for state in env.states}

    for iteration in range(1, max_iterations + 1):
        delta = 0
        for state in env.states:
            if env.is_terminal(state):
                continue

            v = V[state]
            new_value = 0.0

            for action in range(env.n_actions):
                action_prob = policy[state][action]
                if action_prob == 0:
                    continue
                next_state, reward, _ = env.step(action, state=state)
                new_value += action_prob * (reward + gamma * V[next_state])

            V[state] = new_value
            delta = max(delta, abs(v - V[state]))

        if delta < theta and iteration > 1:
            break

    return V


def policy_improvement(env, V, gamma=0.9):
    """정책 개선: 가치 함수로부터 개선된 정책 생성"""
    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            policy[state] = [0.25] * 4
            continue

        q_values = []
        for action in range(env.n_actions):
            next_state, reward, _ = env.step(action, state=state)
            q_values.append(reward + gamma * V[next_state])

        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if abs(q - max_q) < 1e-10]

        action_probs = [0.0] * env.n_actions
        for a in best_actions:
            action_probs[a] = 1.0 / len(best_actions)

        policy[state] = action_probs

    return policy


def policy_iteration(env, gamma=0.9, theta=0.001, max_iterations=100):
    """Policy Iteration 알고리즘"""
    print("=== Policy Iteration 시작 ===\n")

    # 1. 초기 정책 (균등)
    policy = {state: [0.25] * 4 for state in env.states}
    print("초기 정책: 균등 정책 (모든 행동 25%)")

    for iteration in range(1, max_iterations + 1):
        print(f"\n{'=' * 60}")
        print(f"반복 {iteration}")
        print("=" * 60)

        # 2. 정책 평가
        print("정책 평가 중...")
        V = policy_evaluation(env, policy, gamma, theta)

        # 3. 정책 개선
        print("정책 개선 중...")
        new_policy = policy_improvement(env, V, gamma)

        # 4. 수렴 확인
        policy_stable = True
        for state in env.states:
            if env.is_terminal(state):
                continue
            old_best = max(range(4), key=lambda a: policy[state][a])
            new_best = max(range(4), key=lambda a: new_policy[state][a])
            if old_best != new_best:
                policy_stable = False
                break

        if policy_stable:
            print(f"\n정책이 수렴했습니다! (반복 {iteration}회)")
            policy = new_policy
            break

        policy = new_policy

    # 결정적 정책 변환
    policy_det = {}
    for state in env.states:
        if not env.is_terminal(state):
            policy_det[state] = max(range(4), key=lambda a: policy[state][a])

    V = policy_evaluation(env, policy, gamma, theta)
    return policy_det, V


def evaluate_policy(env, policy, gamma=0.9, n_episodes=100):
    """정책 평가 (시뮬레이션)"""
    total_rewards = []
    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        discount = 1.0
        for _ in range(100):
            if state not in policy or env.is_terminal(state):
                break
            action = policy[state]
            next_state, reward, done = env.step(action)
            episode_reward += discount * reward
            discount *= gamma
            state = next_state
            if done:
                break
        total_rewards.append(episode_reward)
    return sum(total_rewards) / len(total_rewards)


def simple_example():
    print("\n=== 간단한 예시 ===\n")
    env = GridWorld(4)
    policy, V = policy_iteration(env, 0.9)

    print("\n최적 정책:")
    env.render_policy(policy)

    print("\n최적 경로 시뮬레이션:")
    state = env.reset()
    total_reward = 0
    print(f"시작: {state}")

    for step in range(20):
        if env.is_terminal(state):
            print(f"\n목표 도달! (총 {step}걸음)")
            break
        if state not in policy:
            print("\n정책에 없는 상태!")
            break

        action = policy[state]
        next_state, reward, done = env.step(action, state=state)
        total_reward += reward
        print(f"  → {GridWorld.ACTION_NAMES[action]} → {next_state} (보상: {reward:.2f})")
        state = next_state

    print(f"\n총 보상: {total_reward:.4f}")


if __name__ == "__main__":
    print("=" * 60)
    print("Policy Iteration 알고리즘 구현")
    print("=" * 60)

    env = GridWorld(4)
    print("\n환경 초기화:")
    env.render()

    print("\n" + "=" * 60)
    policy, V = policy_iteration(env, 0.9, 0.001)

    print("\n\n=== 최종 결과 ===\n")
    print("최적 가치 함수:")
    env.render_values(V)
    print("최적 정책:")
    env.render_policy(policy)

    avg_reward = evaluate_policy(env, policy, 0.9)
    print(f"정책 성능 (평균 누적 보상): {avg_reward:.4f}\n")

    simple_example()

    print("\n" + "=" * 60)
    print("Policy Iteration 완료!")
    print("=" * 60)
