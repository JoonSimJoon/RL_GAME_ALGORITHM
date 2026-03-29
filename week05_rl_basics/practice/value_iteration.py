"""
Value Iteration 구현
벨만 최적 방정식을 반복 적용하여 최적 가치 함수와 정책을 찾는 알고리즘
"""

import math
from gridworld import GridWorld


def extract_policy(env, V, gamma):
    """가치 함수에서 정책 추출"""
    policy = {}
    for state in env.states:
        if env.is_terminal(state):
            continue
        q_values = []
        for action in range(env.n_actions):
            next_state, reward, _ = env.step(action, state=state)
            q_values.append(reward + gamma * V[next_state])
        policy[state] = max(range(len(q_values)), key=lambda i: q_values[i])
    return policy


def value_iteration(env, gamma=0.9, theta=0.001, max_iterations=1000):
    """Value Iteration 알고리즘"""
    print("=== Value Iteration 시작 ===\n")

    # 1. 가치 함수 초기화
    V = {state: 0.0 for state in env.states}

    # 2. 가치 반복
    for iteration in range(1, max_iterations + 1):
        delta = 0

        for state in env.states:
            if env.is_terminal(state):
                continue

            v = V[state]

            # 벨만 최적 방정식: V(s) = max_a [R + γV(s')]
            max_value = -math.inf
            for action in range(env.n_actions):
                next_state, reward, _ = env.step(action, state=state)
                q_value = reward + gamma * V[next_state]
                max_value = max(max_value, q_value)

            V[state] = max_value
            delta = max(delta, abs(v - V[state]))

        if iteration % 5 == 0 or iteration == 1:
            print(f"반복 {iteration:3d}: delta = {delta:.6f}")

        # 3. 수렴 확인
        if delta < theta:
            print(f"\n수렴 완료! (반복 {iteration}회)")
            break

    # 4. 최적 정책 추출
    policy = extract_policy(env, V, gamma)
    return V, policy


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
    V, policy = value_iteration(env, 0.9)

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
    print("Value Iteration 알고리즘 구현")
    print("=" * 60)

    env = GridWorld(4)
    print("\n환경 초기화:")
    env.render()

    print("\n" + "=" * 60)
    V, policy = value_iteration(env, 0.9, 0.001)

    print("\n\n=== 최종 결과 ===\n")
    print("최적 가치 함수:")
    env.render_values(V)
    print("최적 정책:")
    env.render_policy(policy)

    avg_reward = evaluate_policy(env, policy, 0.9, 100)
    print(f"정책 성능 (평균 누적 보상): {avg_reward:.4f}\n")

    simple_example()

    print("\n" + "=" * 60)
    print("Value Iteration 완료!")
    print("=" * 60)
