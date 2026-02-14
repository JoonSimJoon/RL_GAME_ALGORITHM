"""
Policy Iteration 구현
정책 평가와 정책 개선을 번갈아 수행하여 최적 정책을 찾는 알고리즘
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List
from gridworld import GridWorld


def policy_evaluation(
    env: GridWorld,
    policy: Dict[Tuple[int, int], List[float]],
    gamma: float = 0.9,
    theta: float = 0.001,
    max_iterations: int = 1000
) -> Dict[Tuple[int, int], float]:
    """
    정책 평가: 주어진 정책에 대한 가치 함수 계산

    Args:
        env: GridWorld 환경
        policy: 평가할 정책 {state: [p(a0), p(a1), p(a2), p(a3)]}
        gamma: 감가율
        theta: 수렴 임계값
        max_iterations: 최대 반복 횟수

    Returns:
        V: 가치 함수 {state: value}
    """
    # 가치 함수 초기화
    V = {state: 0.0 for state in env.states}

    iteration = 0
    while iteration < max_iterations:
        delta = 0
        iteration += 1

        # 모든 상태에 대해 업데이트
        for state in env.states:
            if env.is_terminal(state):
                continue

            v = V[state]  # 이전 가치 저장

            # 벨만 기대 방정식: V^π(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a)[R + γV(s')]
            new_value = 0.0

            for action in range(env.n_actions):
                # 정책에 따른 행동 확률
                action_prob = policy[state][action]

                if action_prob == 0:
                    continue

                # 다음 상태와 보상
                next_state, reward, _ = env.step(state, action)

                # 기대값 계산
                new_value += action_prob * (reward + gamma * V[next_state])

            V[state] = new_value

            # 변화량 계산
            delta = max(delta, abs(v - V[state]))

        # 수렴 확인
        if delta < theta:
            if iteration > 1:  # 최소 2번은 반복
                break

    return V


def policy_improvement(
    env: GridWorld,
    V: Dict[Tuple[int, int], float],
    gamma: float = 0.9
) -> Dict[Tuple[int, int], List[float]]:
    """
    정책 개선: 가치 함수를 기반으로 탐욕 정책 생성

    Args:
        env: GridWorld 환경
        V: 가치 함수
        gamma: 감가율

    Returns:
        policy: 개선된 정책 {state: [p(a0), p(a1), p(a2), p(a3)]}
    """
    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            # 종료 상태는 균등 정책 (사용되지 않음)
            policy[state] = [0.25, 0.25, 0.25, 0.25]
            continue

        # 각 행동의 Q값 계산
        q_values = []
        for action in range(env.n_actions):
            next_state, reward, _ = env.step(state, action)
            q_value = reward + gamma * V[next_state]
            q_values.append(q_value)

        # 최대 Q값을 가진 행동 찾기
        max_q = max(q_values)
        best_actions = [i for i, q in enumerate(q_values) if abs(q - max_q) < 1e-10]

        # 결정적 정책 생성 (최선의 행동에 확률 1)
        action_probs = [0.0] * env.n_actions

        # 동점인 행동이 여러 개면 균등 분배
        for action in best_actions:
            action_probs[action] = 1.0 / len(best_actions)

        policy[state] = action_probs

    return policy


def policy_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 0.001,
    max_iterations: int = 100
) -> Tuple[Dict, Dict]:
    """
    Policy Iteration 알고리즘

    Args:
        env: GridWorld 환경
        gamma: 감가율
        theta: 수렴 임계값
        max_iterations: 최대 반복 횟수

    Returns:
        policy_deterministic: 최적 정책 {state: action}
        V: 최적 가치 함수 {state: value}
    """
    print("=== Policy Iteration 시작 ===\n")

    # 1. 초기 정책 생성 (균등 정책)
    policy = {}
    for state in env.states:
        policy[state] = [0.25, 0.25, 0.25, 0.25]  # 모든 행동 동일 확률

    print("초기 정책: 균등 정책 (모든 행동 25%)")

    iteration = 0
    while iteration < max_iterations:
        iteration += 1
        print(f"\n{'=' * 60}")
        print(f"반복 {iteration}")
        print(f"{'=' * 60}")

        # 2. 정책 평가
        print("정책 평가 중...")
        V = policy_evaluation(env, policy, gamma, theta)

        # 3. 정책 개선
        print("정책 개선 중...")
        new_policy = policy_improvement(env, V, gamma)

        # 4. 수렴 확인 (정책이 변하지 않으면 종료)
        policy_stable = True
        for state in env.states:
            if env.is_terminal(state):
                continue

            # 최선의 행동이 바뀌었는지 확인
            old_best = np.argmax(policy[state])
            new_best = np.argmax(new_policy[state])

            if old_best != new_best:
                policy_stable = False
                break

        if policy_stable:
            print(f"\n정책이 수렴했습니다! (반복 {iteration}회)")
            policy = new_policy
            break

        policy = new_policy

    # 최종 정책을 결정적 형태로 변환
    policy_deterministic = {}
    for state in env.states:
        if not env.is_terminal(state):
            policy_deterministic[state] = int(np.argmax(policy[state]))

    return policy_deterministic, V


def compare_with_value_iteration(env: GridWorld, gamma: float = 0.9):
    """
    Policy Iteration과 Value Iteration 비교

    Args:
        env: GridWorld 환경
        gamma: 감가율
    """
    print("\n\n" + "=" * 60)
    print("Policy Iteration vs Value Iteration 비교")
    print("=" * 60)

    # Policy Iteration 실행
    print("\n[1] Policy Iteration 실행")
    policy_pi, V_pi = policy_iteration(env, gamma=gamma)

    # Value Iteration 실행 (간단 버전)
    print("\n\n[2] Value Iteration 실행")
    print("=== Value Iteration 시작 ===\n")

    V_vi = {state: 0.0 for state in env.states}
    iteration = 0
    theta = 0.001

    while iteration < 1000:
        delta = 0
        iteration += 1

        for state in env.states:
            if env.is_terminal(state):
                continue

            v = V_vi[state]
            max_value = float('-inf')

            for action in range(env.n_actions):
                next_state, reward, _ = env.step(state, action)
                q_value = reward + gamma * V_vi[next_state]
                max_value = max(max_value, q_value)

            V_vi[state] = max_value
            delta = max(delta, abs(v - V_vi[state]))

        if iteration % 5 == 0 or iteration == 1:
            print(f"반복 {iteration:3d}: delta = {delta:.6f}")

        if delta < theta:
            print(f"\n수렴 완료! (반복 {iteration}회)")
            break

    # Value Iteration 정책 추출
    policy_vi = {}
    for state in env.states:
        if env.is_terminal(state):
            continue

        q_values = []
        for action in range(env.n_actions):
            next_state, reward, _ = env.step(state, action)
            q_value = reward + gamma * V_vi[next_state]
            q_values.append(q_value)

        policy_vi[state] = int(np.argmax(q_values))

    # 결과 비교
    print("\n\n" + "=" * 60)
    print("비교 결과")
    print("=" * 60)

    print("\n[Policy Iteration 정책]")
    env.render_policy(policy_pi)

    print("[Value Iteration 정책]")
    env.render_policy(policy_vi)

    # 정책 일치 여부
    policies_match = all(
        policy_pi.get(s) == policy_vi.get(s)
        for s in env.states
        if not env.is_terminal(s)
    )

    print(f"\n정책 일치 여부: {'✓ 동일' if policies_match else '✗ 다름'}")

    # 가치 함수 차이
    max_value_diff = max(
        abs(V_pi.get(s, 0) - V_vi.get(s, 0))
        for s in env.states
    )

    print(f"최대 가치 차이: {max_value_diff:.6f}")


def analyze_policy_changes(env: GridWorld, gamma: float = 0.9):
    """
    정책 변화 과정 분석

    Args:
        env: GridWorld 환경
        gamma: 감가율
    """
    print("\n\n" + "=" * 60)
    print("정책 변화 과정 분석")
    print("=" * 60)

    # 초기 정책
    policy = {state: [0.25, 0.25, 0.25, 0.25] for state in env.states}

    policies_history = []
    iteration = 0

    while iteration < 10:
        iteration += 1

        # 정책 평가
        V = policy_evaluation(env, policy, gamma, theta=0.001)

        # 결정적 정책 추출 (시각화용)
        policy_det = {}
        for state in env.states:
            if not env.is_terminal(state):
                policy_det[state] = int(np.argmax(policy[state]))

        policies_history.append(policy_det.copy())

        # 정책 개선
        new_policy = policy_improvement(env, V, gamma)

        # 수렴 확인
        policy_stable = all(
            np.argmax(policy[s]) == np.argmax(new_policy[s])
            for s in env.states
            if not env.is_terminal(s)
        )

        if policy_stable:
            print(f"\n정책 수렴: {iteration}회 반복")
            break

        policy = new_policy

    # 각 반복의 정책 시각화
    print("\n정책 변화 과정:")
    for i, pol in enumerate(policies_history, 1):
        print(f"\n[반복 {i}]")
        env.render_policy(pol)


def evaluate_policy(env: GridWorld, policy: Dict, gamma: float = 0.9) -> float:
    """
    정책 성능 평가 (시뮬레이션)

    Args:
        env: GridWorld 환경
        policy: 평가할 정책
        gamma: 감가율

    Returns:
        평균 누적 보상
    """
    n_episodes = 100
    total_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        discount = 1.0
        steps = 0
        max_steps = 100

        while steps < max_steps:
            if state not in policy or env.is_terminal(state):
                break

            action = policy[state]
            next_state, reward, done = env.step(None, action)

            episode_reward += discount * reward
            discount *= gamma

            state = next_state
            steps += 1

            if done:
                break

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Policy Iteration 알고리즘 구현")
    print("=" * 60)

    # 환경 생성
    env = GridWorld(grid_size=4)

    print("\n환경 초기화:")
    env.render()

    # Policy Iteration 실행
    print("\n" + "=" * 60)
    policy, V = policy_iteration(env, gamma=0.9, theta=0.001)

    # 결과 출력
    print("\n\n=== 최종 결과 ===\n")

    print("최적 가치 함수:")
    env.render_values(V)

    print("최적 정책:")
    env.render_policy(policy)

    # 정책 평가
    avg_reward = evaluate_policy(env, policy, gamma=0.9)
    print(f"정책 성능 (평균 누적 보상): {avg_reward:.4f}\n")

    # 정책 변화 과정 분석
    analyze_policy_changes(env, gamma=0.9)

    # Value Iteration과 비교
    compare_with_value_iteration(env, gamma=0.9)

    print("\n" + "=" * 60)
    print("Policy Iteration 완료!")
    print("=" * 60)


def simple_example():
    """간단한 예시 실행"""
    print("\n=== 간단한 예시 ===\n")

    env = GridWorld(grid_size=4)
    policy, V = policy_iteration(env, gamma=0.9)

    print("\n최적 정책:")
    env.render_policy(policy)

    print("\n최적 경로 시뮬레이션:")
    state = env.reset()
    steps = 0
    total_reward = 0

    print(f"시작: {state}")

    while steps < 20:
        if env.is_terminal(state):
            print(f"\n목표 도달! (총 {steps}걸음)")
            break

        if state not in policy:
            print("\n정책에 없는 상태!")
            break

        action = policy[state]
        next_state, reward, done = env.step(state, action)
        total_reward += reward

        print(f"  → {GridWorld.ACTION_NAMES[action]} → {next_state} (보상: {reward:.2f})")

        state = next_state
        steps += 1

    print(f"\n총 보상: {total_reward:.4f}")


def compare_gamma_effect():
    """Gamma 값에 따른 정책 비교"""
    print("\n\n" + "=" * 60)
    print("Gamma 값에 따른 정책 비교")
    print("=" * 60)

    env = GridWorld(grid_size=4)
    gamma_values = [0.5, 0.7, 0.9, 0.99]

    for gamma in gamma_values:
        print(f"\n\n{'=' * 60}")
        print(f"γ = {gamma}")
        print(f"{'=' * 60}")

        policy, V = policy_iteration(env, gamma=gamma, theta=0.001)

        print(f"\n최적 정책 (γ={gamma}):")
        env.render_policy(policy)

        print(f"\n가치 함수 (γ={gamma}):")
        env.render_values(V)

        avg_reward = evaluate_policy(env, policy, gamma=gamma)
        print(f"\n평균 누적 보상: {avg_reward:.4f}")


if __name__ == "__main__":
    # 전체 실행
    main()

    # Gamma 비교
    compare_gamma_effect()

    # 또는 간단한 예시만 실행하려면:
    # simple_example()
