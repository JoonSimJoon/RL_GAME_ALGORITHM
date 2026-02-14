"""
Value Iteration 구현
벨만 최적 방정식을 반복 적용하여 최적 가치 함수와 정책을 찾는 알고리즘
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from gridworld import GridWorld


def value_iteration(
    env: GridWorld,
    gamma: float = 0.9,
    theta: float = 0.001,
    max_iterations: int = 1000
) -> Tuple[Dict, Dict]:
    """
    Value Iteration 알고리즘

    Args:
        env: GridWorld 환경
        gamma: 감가율 (discount factor)
        theta: 수렴 임계값
        max_iterations: 최대 반복 횟수

    Returns:
        V: 최적 가치 함수 {state: value}
        policy: 최적 정책 {state: action}
    """
    print("=== Value Iteration 시작 ===\n")

    # 1. 가치 함수 초기화
    V = {}
    for state in env.states:
        V[state] = 0.0

    # 2. 가치 반복
    iteration = 0

    while iteration < max_iterations:
        delta = 0
        iteration += 1

        # 모든 상태에 대해 업데이트
        for state in env.states:
            # 종료 상태는 건너뛰기
            if env.is_terminal(state):
                continue

            v = V[state]  # 이전 가치 저장

            # 벨만 최적 방정식: V(s) = max_a Σ P(s'|s,a)[R + γV(s')]
            # 결정적 환경이므로 P(s'|s,a) = 1
            max_value = float('-inf')

            for action in range(env.n_actions):
                # 다음 상태와 보상
                next_state, reward, _ = env.step(state, action)

                # Q(s, a) 계산
                q_value = reward + gamma * V[next_state]

                # 최대값 찾기
                max_value = max(max_value, q_value)

            V[state] = max_value

            # 변화량 계산
            delta = max(delta, abs(v - V[state]))

        # 진행 상황 출력 (5번마다)
        if iteration % 5 == 0 or iteration == 1:
            print(f"반복 {iteration:3d}: delta = {delta:.6f}")

        # 3. 수렴 확인
        if delta < theta:
            print(f"\n수렴 완료! (반복 {iteration}회)")
            break

    # 4. 최적 정책 추출
    policy = extract_policy(env, V, gamma)

    return V, policy


def extract_policy(env: GridWorld, V: Dict, gamma: float) -> Dict:
    """
    가치 함수로부터 탐욕 정책 추출

    Args:
        env: GridWorld 환경
        V: 가치 함수
        gamma: 감가율

    Returns:
        policy: 정책 {state: action}
    """
    policy = {}

    for state in env.states:
        if env.is_terminal(state):
            continue

        # 각 행동의 Q값 계산
        q_values = []
        for action in range(env.n_actions):
            next_state, reward, _ = env.step(state, action)
            q_value = reward + gamma * V[next_state]
            q_values.append(q_value)

        # 최대 Q값을 주는 행동 선택
        best_action = int(np.argmax(q_values))
        policy[state] = best_action

    return policy


def evaluate_policy(env: GridWorld, policy: Dict, gamma: float = 0.9, n_episodes: int = 100) -> float:
    """
    정책의 성능 평가 (시뮬레이션)

    Args:
        env: GridWorld 환경
        policy: 평가할 정책
        gamma: 감가율
        n_episodes: 시뮬레이션 횟수

    Returns:
        평균 누적 보상
    """
    total_rewards = []

    for _ in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        discount = 1.0
        steps = 0
        max_steps = 100  # 무한 루프 방지

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


def visualize_values_heatmap(env: GridWorld, V: Dict, title: str = "가치 함수"):
    """
    가치 함수를 히트맵으로 시각화

    Args:
        env: GridWorld 환경
        V: 가치 함수
        title: 그래프 제목
    """
    # 가치 행렬 생성
    value_matrix = np.zeros((env.grid_size, env.grid_size))

    for r in range(env.grid_size):
        for c in range(env.grid_size):
            state = (r, c)
            value_matrix[r, c] = V.get(state, 0)

    # 히트맵 그리기
    plt.figure(figsize=(8, 7))
    im = plt.imshow(value_matrix, cmap='RdYlGn', interpolation='nearest')

    # 각 셀에 값 표시
    for r in range(env.grid_size):
        for c in range(env.grid_size):
            state = (r, c)
            value = V.get(state, 0)

            # 텍스트 색상 (배경에 따라)
            text_color = 'white' if value < 0 else 'black'

            # 특수 상태 표시
            if state == env.goal:
                text = f"G\n{value:.2f}"
            elif state in env.obstacles:
                text = f"X\n{value:.2f}"
            else:
                text = f"{value:.2f}"

            plt.text(c, r, text, ha='center', va='center',
                    color=text_color, fontsize=12, fontweight='bold')

    plt.colorbar(im, label='가치')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('열')
    plt.ylabel('행')
    plt.xticks(range(env.grid_size))
    plt.yticks(range(env.grid_size))
    plt.tight_layout()
    plt.savefig('value_iteration_heatmap.png', dpi=150, bbox_inches='tight')
    print("\n히트맵 저장: value_iteration_heatmap.png")
    plt.show()


def compare_gamma_values(env: GridWorld):
    """
    다양한 gamma 값에 따른 정책 비교

    Args:
        env: GridWorld 환경
    """
    gamma_values = [0.5, 0.7, 0.9, 0.99]

    print("\n\n=== Gamma 값 비교 ===\n")

    for gamma in gamma_values:
        print(f"\n--- γ = {gamma} ---")

        V, policy = value_iteration(env, gamma=gamma, theta=0.001)

        print(f"\n최적 정책 (γ={gamma}):")
        env.render_policy(policy)

        # 성능 평가
        avg_reward = evaluate_policy(env, policy, gamma=gamma)
        print(f"평균 누적 보상: {avg_reward:.4f}")


def test_convergence_analysis(env: GridWorld):
    """
    수렴 과정 분석

    Args:
        env: GridWorld 환경
    """
    print("\n\n=== 수렴 과정 분석 ===\n")

    V = {state: 0.0 for state in env.states}
    gamma = 0.9
    theta = 0.001

    # 특정 상태들의 가치 변화 추적
    tracked_states = [(0, 0), (1, 2), (2, 3), (3, 3)]
    value_history = {state: [] for state in tracked_states}

    iteration = 0
    max_iterations = 100

    while iteration < max_iterations:
        delta = 0
        iteration += 1

        for state in env.states:
            if env.is_terminal(state):
                continue

            v = V[state]
            max_value = float('-inf')

            for action in range(env.n_actions):
                next_state, reward, _ = env.step(state, action)
                q_value = reward + gamma * V[next_state]
                max_value = max(max_value, q_value)

            V[state] = max_value
            delta = max(delta, abs(v - V[state]))

        # 추적 상태 기록
        for state in tracked_states:
            value_history[state].append(V[state])

        if delta < theta:
            print(f"수렴 완료! (반복 {iteration}회)")
            break

    # 수렴 그래프 그리기
    plt.figure(figsize=(10, 6))

    for state in tracked_states:
        if state == env.goal:
            label = f"목표 {state}"
        elif state in env.obstacles:
            label = f"장애물 {state}"
        else:
            label = f"상태 {state}"

        plt.plot(value_history[state], marker='o', label=label)

    plt.xlabel('반복 횟수')
    plt.ylabel('가치')
    plt.title('Value Iteration 수렴 과정')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('value_iteration_convergence.png', dpi=150, bbox_inches='tight')
    print("\n수렴 그래프 저장: value_iteration_convergence.png")
    plt.show()


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("Value Iteration 알고리즘 구현")
    print("=" * 60)

    # 환경 생성
    env = GridWorld(grid_size=4)

    print("\n환경 초기화:")
    env.render()

    # Value Iteration 실행
    print("\n" + "=" * 60)
    V, policy = value_iteration(env, gamma=0.9, theta=0.001)

    # 결과 출력
    print("\n\n=== 최종 결과 ===\n")

    print("최적 가치 함수:")
    env.render_values(V)

    print("최적 정책:")
    env.render_policy(policy)

    # 정책 평가
    avg_reward = evaluate_policy(env, policy, gamma=0.9, n_episodes=100)
    print(f"정책 성능 (평균 누적 보상): {avg_reward:.4f}\n")

    # 시각화
    print("\n가치 함수 히트맵 생성 중...")
    visualize_values_heatmap(env, V, title="Value Iteration - 최적 가치 함수")

    # 수렴 과정 분석
    test_convergence_analysis(env)

    # 다양한 gamma 값 비교
    compare_gamma_values(env)

    print("\n" + "=" * 60)
    print("Value Iteration 완료!")
    print("=" * 60)


def simple_example():
    """간단한 예시 실행"""
    print("\n=== 간단한 예시 ===\n")

    env = GridWorld(grid_size=4)
    V, policy = value_iteration(env, gamma=0.9)

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


if __name__ == "__main__":
    # 전체 실행
    main()

    # 또는 간단한 예시만 실행하려면:
    # simple_example()
