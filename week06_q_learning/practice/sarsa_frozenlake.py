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

import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import deque


def epsilon_greedy(Q, state, epsilon, num_actions):
    """
    ε-greedy 정책으로 행동 선택

    Args:
        Q: Q-table
        state: 현재 상태
        epsilon: 탐험 확률
        num_actions: 가능한 행동 개수

    Returns:
        선택된 행동
    """
    if random.random() < epsilon:
        return random.randint(0, num_actions - 1)
    else:
        return np.argmax(Q[state])


def print_policy(Q, shape=(4, 4)):
    """학습된 Q-table로부터 정책을 시각화"""
    symbols = ['←', '↓', '→', '↑']
    policy = np.argmax(Q, axis=1)

    print("\n학습된 정책 (화살표):")
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = i * shape[1] + j
            print(symbols[policy[state]], end=' ')
        print()


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


def train_sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99,
                epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                eval_interval=100, verbose=True):
    """
    SARSA 알고리즘으로 학습

    핵심 차이점:
    - Q-Learning: best_next_q = max(Q[next_state])
    - SARSA: next_q = Q[next_state, next_action] (실제 선택한 행동)

    Args:
        env: Gymnasium 환경
        num_episodes: 학습 에피소드 수
        alpha: 학습률
        gamma: 할인율
        epsilon_start: 초기 탐험률
        epsilon_min: 최소 탐험률
        epsilon_decay: 탐험률 감소율
        eval_interval: 평가 주기
        verbose: 진행상황 출력 여부

    Returns:
        Q: 학습된 Q-table
        rewards_history: 에피소드별 보상 기록
        success_history: 평가 시점별 성공률 기록
    """
    # Q-table 초기화
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    # 학습 기록
    rewards_history = []
    success_history = []
    epsilon = epsilon_start
    recent_rewards = deque(maxlen=100)

    if verbose:
        print(f"SARSA 학습 시작")
        print(f"상태 개수: {num_states}, 행동 개수: {num_actions}")
        print(f"하이퍼파라미터: α={alpha}, γ={gamma}, ε={epsilon_start}→{epsilon_min}")
        print("-" * 70)

    for episode in range(num_episodes):
        state, _ = env.reset()

        # SARSA 특징: 에피소드 시작 시 첫 행동을 미리 선택
        action = epsilon_greedy(Q, state, epsilon, num_actions)

        done = False
        total_reward = 0
        steps = 0

        # 에피소드 실행
        while not done:
            # 1. 현재 행동(action)을 실행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 2. 다음 행동(next_action)을 미리 선택 (중요!)
            #    이게 SARSA의 핵심: 실제로 할 행동을 미리 정함
            next_action = epsilon_greedy(Q, next_state, epsilon, num_actions)

            # 3. SARSA 업데이트
            # Q(s,a) ← Q(s,a) + α[R + γ·Q(s',a') - Q(s,a)]
            #                              ^^^^^^^^
            #                              실제 선택한 행동의 Q값!
            if not done:
                # 다음 상태의 Q값: 실제로 선택한 행동(next_action)의 Q값
                next_q = Q[next_state, next_action]
            else:
                # 종료 상태면 미래 보상 없음
                next_q = 0.0

            # TD target 계산
            td_target = reward + gamma * next_q
            td_error = td_target - Q[state, action]

            # Q-table 업데이트
            Q[state, action] += alpha * td_error

            # 4. 상태 및 행동 전이
            #    이미 선택한 next_action을 다음 스텝에서 사용!
            state = next_state
            action = next_action  # 핵심: 이미 선택한 행동을 사용

            total_reward += reward
            steps += 1

        # 에피소드 종료 후 기록
        rewards_history.append(total_reward)
        recent_rewards.append(total_reward)

        # ε 감소
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 주기적으로 성능 평가
        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_policy(env, Q, num_episodes=100)
            success_history.append((episode + 1, success_rate))

            if verbose:
                avg_reward = np.mean(recent_rewards)
                print(f"Episode {episode + 1:5d} | "
                      f"ε={epsilon:.3f} | "
                      f"Avg Reward={avg_reward:.3f} | "
                      f"Success Rate={success_rate*100:.1f}%")

    if verbose:
        print("-" * 70)
        print(f"학습 완료!")

    return Q, rewards_history, success_history


def train_q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99,
                     epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                     eval_interval=100, verbose=False):
    """
    비교를 위한 Q-Learning 구현 (간단 버전)

    Args:
        env: Gymnasium 환경
        (기타 파라미터는 SARSA와 동일)

    Returns:
        Q: 학습된 Q-table
        rewards_history: 에피소드별 보상 기록
        success_history: 평가 시점별 성공률 기록
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    rewards_history = []
    success_history = []
    epsilon = epsilon_start
    recent_rewards = deque(maxlen=100)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0

        while not done:
            # 행동 선택
            action = epsilon_greedy(Q, state, epsilon, num_actions)

            # 환경 step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning 업데이트: max Q(s',a') 사용
            if not done:
                best_next_q = np.max(Q[next_state])
            else:
                best_next_q = 0.0

            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        recent_rewards.append(total_reward)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % eval_interval == 0:
            success_rate = evaluate_policy(env, Q, num_episodes=100)
            success_history.append((episode + 1, success_rate))

    return Q, rewards_history, success_history


def compare_algorithms(env, num_episodes=10000, num_runs=5, **kwargs):
    """
    Q-Learning과 SARSA를 여러 번 실행하여 평균 비교

    Args:
        env: Gymnasium 환경
        num_episodes: 에피소드 수
        num_runs: 반복 실행 횟수 (평균을 위해)
        **kwargs: 하이퍼파라미터

    Returns:
        q_results: Q-Learning 결과 (평균 보상, 평균 성공률)
        sarsa_results: SARSA 결과 (평균 보상, 평균 성공률)
    """
    print("=" * 70)
    print(f"Q-Learning vs SARSA 비교 ({num_runs}회 반복)")
    print("=" * 70)

    q_rewards_all = []
    q_success_all = []
    sarsa_rewards_all = []
    sarsa_success_all = []

    for run in range(num_runs):
        print(f"\n[Run {run + 1}/{num_runs}]")

        # Q-Learning
        print("  Q-Learning 학습 중...")
        Q_q, rewards_q, success_q = train_q_learning(
            env, num_episodes=num_episodes, verbose=False, **kwargs
        )
        q_rewards_all.append(rewards_q)
        q_success_all.append(success_q)

        # SARSA
        print("  SARSA 학습 중...")
        Q_s, rewards_s, success_s = train_sarsa(
            env, num_episodes=num_episodes, verbose=False, **kwargs
        )
        sarsa_rewards_all.append(rewards_s)
        sarsa_success_all.append(success_s)

    # 평균 계산
    q_rewards_mean = np.mean(q_rewards_all, axis=0)
    sarsa_rewards_mean = np.mean(sarsa_rewards_all, axis=0)

    # 성공률 평균 (에피소드 번호별로)
    q_success_mean = {}
    sarsa_success_mean = {}

    for success_list in q_success_all:
        for ep, rate in success_list:
            if ep not in q_success_mean:
                q_success_mean[ep] = []
            q_success_mean[ep].append(rate)

    for success_list in sarsa_success_all:
        for ep, rate in success_list:
            if ep not in sarsa_success_mean:
                sarsa_success_mean[ep] = []
            sarsa_success_mean[ep].append(rate)

    q_success_mean = [(ep, np.mean(rates)) for ep, rates in sorted(q_success_mean.items())]
    sarsa_success_mean = [(ep, np.mean(rates)) for ep, rates in sorted(sarsa_success_mean.items())]

    return (q_rewards_mean, q_success_mean), (sarsa_rewards_mean, sarsa_success_mean)


def plot_comparison(q_results, sarsa_results, title="Q-Learning vs SARSA"):
    """
    Q-Learning과 SARSA 결과를 그래프로 비교

    Args:
        q_results: (rewards_mean, success_mean) 튜플
        sarsa_results: (rewards_mean, success_mean) 튜플
        title: 그래프 제목
    """
    q_rewards, q_success = q_results
    sarsa_rewards, sarsa_success = sarsa_results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 보상 곡선 비교 (이동평균)
    window = 100

    if len(q_rewards) >= window:
        q_moving_avg = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(q_moving_avg, label='Q-Learning', color='blue', linewidth=2)

    if len(sarsa_rewards) >= window:
        sarsa_moving_avg = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(sarsa_moving_avg, label='SARSA', color='red', linewidth=2)

    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Training Reward (100-Episode Moving Average)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 성공률 비교
    if q_success:
        q_episodes, q_rates = zip(*q_success)
        ax2.plot(q_episodes, [r * 100 for r in q_rates],
                marker='o', label='Q-Learning', color='blue', linewidth=2)

    if sarsa_success:
        sarsa_episodes, sarsa_rates = zip(*sarsa_success)
        ax2.plot(sarsa_episodes, [r * 100 for r in sarsa_rates],
                marker='s', label='SARSA', color='red', linewidth=2)

    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Success Rate (%)')
    ax2.set_title('Evaluation Success Rate')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 105])

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('sarsa_comparison.png', dpi=100, bbox_inches='tight')
    print("\n그래프가 'sarsa_comparison.png'로 저장되었습니다.")
    plt.show()


def main():
    """
    메인 함수: Q-Learning과 SARSA 비교 실습
    """
    print("=" * 70)
    print("SARSA vs Q-Learning 비교 실습")
    print("=" * 70)

    # 환경 생성
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)

    print(f"\n환경 정보:")
    print(f"  상태 공간: {env.observation_space.n}")
    print(f"  행동 공간: {env.action_space.n}")
    print(f"  is_slippery: False\n")

    # 하이퍼파라미터
    params = {
        'alpha': 0.1,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_min': 0.01,
        'epsilon_decay': 0.995,
        'eval_interval': 500
    }

    # 1. SARSA 단독 학습
    print("\n" + "=" * 70)
    print("1. SARSA 학습")
    print("=" * 70)
    Q_sarsa, rewards_sarsa, success_sarsa = train_sarsa(
        env, num_episodes=10000, verbose=True, **params
    )

    final_success_sarsa = evaluate_policy(env, Q_sarsa, num_episodes=1000)
    print(f"\nSARSA 최종 성공률: {final_success_sarsa*100:.2f}%")
    print("\nSARSA 학습된 정책:")
    print_policy(Q_sarsa)

    # 2. Q-Learning 단독 학습
    print("\n" + "=" * 70)
    print("2. Q-Learning 학습")
    print("=" * 70)
    Q_qlearn, rewards_qlearn, success_qlearn = train_q_learning(
        env, num_episodes=10000, verbose=True, **params
    )

    final_success_qlearn = evaluate_policy(env, Q_qlearn, num_episodes=1000)
    print(f"\nQ-Learning 최종 성공률: {final_success_qlearn*100:.2f}%")
    print("\nQ-Learning 학습된 정책:")
    print_policy(Q_qlearn)

    # 3. 비교 분석
    print("\n" + "=" * 70)
    print("3. 알고리즘 비교")
    print("=" * 70)

    # 여러 번 실행하여 평균 비교 (더 신뢰성 있는 결과)
    q_results, sarsa_results = compare_algorithms(
        env, num_episodes=10000, num_runs=3, **params
    )

    # 결과 그래프
    plot_comparison(q_results, sarsa_results)

    # 최종 분석
    print("\n" + "=" * 70)
    print("분석 결과")
    print("=" * 70)
    print("\nFrozenLake (is_slippery=False) 환경에서:")
    print("- Q-Learning과 SARSA의 성능이 매우 유사합니다")
    print("- 두 알고리즘 모두 최적 정책을 잘 학습합니다")
    print()
    print("이유:")
    print("- 결정적 환경이라 탐험 중 실수가 적음")
    print("- 간단한 환경이라 차이가 두드러지지 않음")
    print()
    print("차이가 나는 경우:")
    print("- is_slippery=True (확률적 환경)")
    print("- 위험한 상태가 많은 환경 (절벽 문제 등)")
    print("- SARSA가 더 보수적이고 안전한 정책 학습")

    env.close()

    print("\n" + "=" * 70)
    print("실습 완료!")
    print("=" * 70)

    # 추가 실험 제안
    print("\n💡 추가 실험 아이디어:")
    print("1. is_slippery=True로 변경하여 확률적 환경에서 비교")
    print("2. CliffWalking 환경에서 비교 (절벽 옆 경로)")
    print("3. epsilon_min을 0.1로 높여서 탐험이 많을 때 차이 확인")
    print("4. gamma를 낮춰서 (0.5) 단기 보상 중시 시 차이 확인")


if __name__ == "__main__":
    main()


"""
실습 문제:

1. SARSA vs Q-Learning 이해:
   - 업데이트 수식의 차이를 코드에서 찾아보세요
   - 왜 SARSA는 next_action을 미리 선택하나요?
   - Off-policy와 On-policy의 의미는?

2. 실험:
   a) is_slippery=False (현재 설정):
      - 두 알고리즘의 성공률 비교
      - 학습 속도 비교
      - 학습된 정책 비교

   b) is_slippery=True로 변경:
      - 두 알고리즘의 차이가 더 커지나요?
      - 어떤 알고리즘이 더 안정적인가요?

3. 도전 과제:
   - CliffWalking-v0 환경 설치 및 실험
     (pip install gymnasium[toy-text])
   - 절벽 옆 경로에서 두 알고리즘의 차이 확인
   - SARSA가 더 안전한 경로를 학습하는지 검증

4. 분석 질문:
   - 어떤 상황에서 SARSA를 선택해야 하나요?
   - 어떤 상황에서 Q-Learning을 선택해야 하나요?
   - 실제 로봇에는 어떤 알고리즘이 더 적합한가요?
"""
