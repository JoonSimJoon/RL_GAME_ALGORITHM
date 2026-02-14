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

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque
import itertools


def epsilon_greedy(Q, state, epsilon, num_actions):
    """ε-greedy 정책으로 행동 선택"""
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(Q[state])


def evaluate_policy(env, Q, num_episodes=100):
    """학습된 정책의 성능 평가"""
    success_count = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            if reward == 1.0:
                success_count += 1
    return success_count / num_episodes


def train_q_learning(env, num_episodes, alpha, gamma, epsilon_start,
                     epsilon_min, epsilon_decay, eval_interval=100):
    """
    Q-Learning 학습 (간단 버전)

    Returns:
        success_history: [(에피소드, 성공률), ...]
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    success_history = []
    epsilon = epsilon_start

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = epsilon_greedy(Q, state, epsilon, num_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            best_next_q = np.max(Q[next_state]) if not done else 0.0
            td_target = reward + gamma * best_next_q
            Q[state, action] += alpha * (td_target - Q[state, action])

            state = next_state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 평가
        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_policy(env, Q, num_episodes=100)
            success_history.append((episode + 1, success_rate))

    return success_history


def experiment_alpha(env, alphas, num_episodes=5000, num_runs=3):
    """
    학습률(α) 변화 실험

    Args:
        env: Gymnasium 환경
        alphas: 실험할 α 값들
        num_episodes: 에피소드 수
        num_runs: 각 설정당 반복 횟수

    Returns:
        results: {alpha: [(episode, success_rate), ...]}
    """
    print("=" * 70)
    print("실험 1: 학습률(α) 변화")
    print("=" * 70)
    print(f"테스트할 α 값: {alphas}")
    print(f"각 설정당 {num_runs}회 반복\n")

    results = {}

    for alpha in alphas:
        print(f"α = {alpha} 실험 중...", end=' ')
        all_histories = []

        for run in range(num_runs):
            history = train_q_learning(
                env,
                num_episodes=num_episodes,
                alpha=alpha,
                gamma=0.99,  # 고정
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,  # 고정
                eval_interval=250
            )
            all_histories.append(history)

        # 평균 계산
        avg_history = {}
        for history in all_histories:
            for ep, rate in history:
                if ep not in avg_history:
                    avg_history[ep] = []
                avg_history[ep].append(rate)

        results[alpha] = [(ep, np.mean(rates)) for ep, rates in sorted(avg_history.items())]
        print(f"완료! 최종 성공률: {results[alpha][-1][1]*100:.1f}%")

    return results


def experiment_gamma(env, gammas, num_episodes=5000, num_runs=3):
    """
    할인율(γ) 변화 실험

    Args:
        env: Gymnasium 환경
        gammas: 실험할 γ 값들
        num_episodes: 에피소드 수
        num_runs: 각 설정당 반복 횟수

    Returns:
        results: {gamma: [(episode, success_rate), ...]}
    """
    print("\n" + "=" * 70)
    print("실험 2: 할인율(γ) 변화")
    print("=" * 70)
    print(f"테스트할 γ 값: {gammas}")
    print(f"각 설정당 {num_runs}회 반복\n")

    results = {}

    for gamma in gammas:
        print(f"γ = {gamma} 실험 중...", end=' ')
        all_histories = []

        for run in range(num_runs):
            history = train_q_learning(
                env,
                num_episodes=num_episodes,
                alpha=0.1,  # 고정
                gamma=gamma,
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=0.995,  # 고정
                eval_interval=250
            )
            all_histories.append(history)

        # 평균 계산
        avg_history = {}
        for history in all_histories:
            for ep, rate in history:
                if ep not in avg_history:
                    avg_history[ep] = []
                avg_history[ep].append(rate)

        results[gamma] = [(ep, np.mean(rates)) for ep, rates in sorted(avg_history.items())]
        print(f"완료! 최종 성공률: {results[gamma][-1][1]*100:.1f}%")

    return results


def experiment_epsilon_decay(env, decays, num_episodes=5000, num_runs=3):
    """
    탐험률 감소율(ε_decay) 변화 실험

    Args:
        env: Gymnasium 환경
        decays: 실험할 ε_decay 값들
        num_episodes: 에피소드 수
        num_runs: 각 설정당 반복 횟수

    Returns:
        results: {decay: [(episode, success_rate), ...]}
    """
    print("\n" + "=" * 70)
    print("실험 3: 탐험률 감소율(ε_decay) 변화")
    print("=" * 70)
    print(f"테스트할 ε_decay 값: {decays}")
    print(f"각 설정당 {num_runs}회 반복\n")

    results = {}

    for decay in decays:
        print(f"ε_decay = {decay} 실험 중...", end=' ')
        all_histories = []

        for run in range(num_runs):
            history = train_q_learning(
                env,
                num_episodes=num_episodes,
                alpha=0.1,  # 고정
                gamma=0.99,  # 고정
                epsilon_start=1.0,
                epsilon_min=0.01,
                epsilon_decay=decay
            )
            all_histories.append(history)

        # 평균 계산
        avg_history = {}
        for history in all_histories:
            for ep, rate in history:
                if ep not in avg_history:
                    avg_history[ep] = []
                avg_history[ep].append(rate)

        results[decay] = [(ep, np.mean(rates)) for ep, rates in sorted(avg_history.items())]
        print(f"완료! 최종 성공률: {results[decay][-1][1]*100:.1f}%")

    return results


def plot_experiment_results(results, param_name, ylabel="Success Rate (%)"):
    """
    실험 결과를 그래프로 시각화

    Args:
        results: {param_value: [(episode, success_rate), ...]}
        param_name: 파라미터 이름 (예: "α", "γ")
        ylabel: y축 레이블
    """
    plt.figure(figsize=(10, 6))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, (param_value, history) in enumerate(sorted(results.items())):
        episodes, success_rates = zip(*history)
        plt.plot(episodes, [sr * 100 for sr in success_rates],
                marker='o', label=f'{param_name}={param_value}',
                color=colors[i], linewidth=2, markersize=6)

    plt.xlabel('Episode', fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(f'Impact of {param_name} on Learning', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 105])

    filename = f'experiment_{param_name}.png'
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    print(f"  그래프 저장: {filename}")
    plt.show()


def grid_search(env, param_grid, num_episodes=5000):
    """
    그리드 서치로 최적 하이퍼파라미터 조합 찾기

    Args:
        env: Gymnasium 환경
        param_grid: {param_name: [values]}
        num_episodes: 에피소드 수

    Returns:
        best_params: 최고 성능 파라미터
        best_score: 최고 성공률
        all_results: 모든 조합의 결과
    """
    print("\n" + "=" * 70)
    print("그리드 서치: 최적 하이퍼파라미터 찾기")
    print("=" * 70)

    # 모든 조합 생성
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))

    print(f"총 {len(combinations)}개 조합 테스트")
    print("-" * 70)

    best_score = -1
    best_params = None
    all_results = []

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"[{i+1}/{len(combinations)}] 테스트 중: {params}")

        # 3회 반복 평균
        scores = []
        for run in range(3):
            history = train_q_learning(env, num_episodes=num_episodes, **params)
            final_score = history[-1][1]  # 마지막 성공률
            scores.append(final_score)

        avg_score = np.mean(scores)
        all_results.append((params, avg_score))

        print(f"  평균 성공률: {avg_score*100:.2f}%")

        if avg_score > best_score:
            best_score = avg_score
            best_params = params
            print(f"  ★ 새로운 최고 기록!")

    print("\n" + "=" * 70)
    print("그리드 서치 완료!")
    print("=" * 70)
    print(f"최적 파라미터: {best_params}")
    print(f"최고 성공률: {best_score*100:.2f}%")

    return best_params, best_score, all_results


def visualize_grid_search_results(all_results):
    """
    그리드 서치 결과를 표로 출력

    Args:
        all_results: [(params, score), ...]
    """
    print("\n전체 결과 (성공률 높은 순):")
    print("-" * 70)
    print(f"{'Rank':<6} {'Alpha':<8} {'Gamma':<8} {'Decay':<8} {'Success Rate':<15}")
    print("-" * 70)

    # 성공률 높은 순으로 정렬
    sorted_results = sorted(all_results, key=lambda x: x[1], reverse=True)

    for rank, (params, score) in enumerate(sorted_results[:10], 1):
        print(f"{rank:<6} "
              f"{params.get('alpha', '-'):<8} "
              f"{params.get('gamma', '-'):<8} "
              f"{params.get('epsilon_decay', '-'):<8} "
              f"{score*100:.2f}%")


def main():
    """
    메인 함수: 하이퍼파라미터 실험
    """
    print("=" * 70)
    print("Q-Learning 하이퍼파라미터 실험")
    print("=" * 70)

    # 환경 생성
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)

    print(f"\n환경: FrozenLake-v1 (is_slippery=False)")
    print(f"상태: {env.observation_space.n}, 행동: {env.action_space.n}\n")

    # 실험 1: 학습률(α) 변화
    alpha_results = experiment_alpha(
        env,
        alphas=[0.01, 0.05, 0.1, 0.3, 0.5],
        num_episodes=5000,
        num_runs=3
    )
    plot_experiment_results(alpha_results, "α (Learning Rate)")

    # 분석
    print("\n[분석] 학습률(α):")
    print("- α=0.01: 학습이 너무 느림 (안정적이지만 비효율적)")
    print("- α=0.1~0.3: 적절한 속도로 학습 (권장)")
    print("- α=0.5: 빠르지만 불안정할 수 있음 (진동)")
    print()

    # 실험 2: 할인율(γ) 변화
    gamma_results = experiment_gamma(
        env,
        gammas=[0.5, 0.7, 0.9, 0.95, 0.99],
        num_episodes=5000,
        num_runs=3
    )
    plot_experiment_results(gamma_results, "γ (Discount Factor)")

    # 분석
    print("\n[분석] 할인율(γ):")
    print("- γ=0.5: 단기 보상만 고려 (긴 경로 학습 어려움)")
    print("- γ=0.9: 중간 미래까지 고려 (10스텝 정도)")
    print("- γ=0.99: 먼 미래까지 고려 (100스텝, 권장)")
    print()

    # 실험 3: 탐험률 감소(ε_decay) 변화
    decay_results = experiment_epsilon_decay(
        env,
        decays=[0.99, 0.995, 0.999],
        num_episodes=5000,
        num_runs=3
    )
    plot_experiment_results(decay_results, "ε_decay (Exploration Decay)")

    # 분석
    print("\n[분석] 탐험률 감소(ε_decay):")
    print("- decay=0.99: 빠른 수렴 (조기 수렴 위험)")
    print("- decay=0.995: 균형 잡힌 탐험 (권장)")
    print("- decay=0.999: 충분한 탐험 (느린 학습)")
    print()

    # 실험 4: 그리드 서치 (선택)
    print("\n진행하시겠습니까? (y/n): ", end='')
    response = input().strip().lower()

    if response == 'y':
        param_grid = {
            'alpha': [0.05, 0.1, 0.2],
            'gamma': [0.95, 0.99],
            'epsilon_start': [1.0],
            'epsilon_min': [0.01],
            'epsilon_decay': [0.99, 0.995],
            'eval_interval': [250]
        }

        best_params, best_score, all_results = grid_search(
            env, param_grid, num_episodes=5000
        )

        visualize_grid_search_results(all_results)

    env.close()

    # 최종 권장사항
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

    print("\n주의사항:")
    print("- 환경마다 최적 파라미터가 다를 수 있습니다")
    print("- 실험을 통해 자신의 환경에 맞는 값을 찾으세요")
    print("- 위 값은 시작점으로 좋은 기본값입니다")

    print("\n" + "=" * 70)
    print("실험 완료!")
    print("=" * 70)


if __name__ == "__main__":
    main()


"""
실습 문제:

1. 학습률(α) 이해:
   - α=0.01과 α=0.5의 학습 곡선을 비교하세요
   - 어떤 차이가 있나요? 왜 그럴까요?
   - 최적 α 값은 얼마인가요?

2. 할인율(γ) 이해:
   - γ=0.5와 γ=0.99의 학습 결과 비교
   - γ가 작으면 왜 성능이 나쁠까요?
   - FrozenLake에서 최적 γ는?

3. 탐험률 감소(ε_decay) 이해:
   - decay=0.99와 0.999 비교
   - 빠른 감소의 장단점은?
   - 느린 감소의 장단점은?

4. 그리드 서치 실험:
   - 위 param_grid를 수정하여 다른 조합 실험
   - 최적 조합을 찾았나요?
   - 왜 그 조합이 최적인가요?

5. 도전 과제:
   - is_slippery=True로 변경하여 실험
   - 최적 파라미터가 바뀌나요?
   - 확률적 환경에서는 어떤 파라미터가 중요한가요?

6. 분석 질문:
   - 하이퍼파라미터 튜닝이 왜 중요한가요?
   - 실무에서는 어떻게 최적 파라미터를 찾나요?
   - 시간이 제한적이면 어떤 파라미터를 먼저 튜닝하나요?
"""
