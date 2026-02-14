"""
Q-Learning을 사용한 FrozenLake 학습
OpenAI Gymnasium의 FrozenLake-v1 환경에서 Q-Learning 알고리즘을 구현합니다.

실습 목표:
1. Q-table 기반 Q-Learning 구현
2. ε-greedy 탐험 전략 이해
3. 학습 곡선 관찰
4. 학습된 정책 시각화
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
        Q: Q-table (numpy array)
        state: 현재 상태
        epsilon: 탐험 확률
        num_actions: 가능한 행동 개수

    Returns:
        선택된 행동
    """
    if random.random() < epsilon:
        # 탐험: 랜덤 행동
        return random.randint(0, num_actions - 1)
    else:
        # 활용: 최선의 행동
        return np.argmax(Q[state])


def print_policy(Q, shape=(4, 4)):
    """
    학습된 Q-table로부터 정책을 시각화

    Args:
        Q: Q-table
        shape: 격자 크기 (행, 열)
    """
    symbols = ['←', '↓', '→', '↑']
    policy = np.argmax(Q, axis=1)

    print("\n학습된 정책 (화살표):")
    for i in range(shape[0]):
        for j in range(shape[1]):
            state = i * shape[1] + j
            print(symbols[policy[state]], end=' ')
        print()


def evaluate_policy(env, Q, num_episodes=100, render=False):
    """
    학습된 정책의 성능 평가 (탐험 없이 greedy만 사용)

    Args:
        env: Gymnasium 환경
        Q: 학습된 Q-table
        num_episodes: 평가할 에피소드 수
        render: 렌더링 여부

    Returns:
        성공률 (0.0 ~ 1.0)
    """
    success_count = 0

    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            if render:
                env.render()

            # Greedy 정책 (탐험 없음)
            action = np.argmax(Q[state])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if reward == 1.0:
                success_count += 1

    return success_count / num_episodes


def train_q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99,
                     epsilon_start=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                     eval_interval=100, verbose=True):
    """
    Q-Learning 알고리즘으로 학습

    Args:
        env: Gymnasium 환경
        num_episodes: 학습 에피소드 수
        alpha: 학습률 (learning rate)
        gamma: 할인율 (discount factor)
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
    # Q-table 초기화 (모든 값을 0으로)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))

    # 학습 기록
    rewards_history = []
    success_history = []
    epsilon = epsilon_start

    # 최근 100개 에피소드의 평균 보상 (학습 진행 확인용)
    recent_rewards = deque(maxlen=100)

    if verbose:
        print(f"Q-Learning 학습 시작")
        print(f"상태 개수: {num_states}, 행동 개수: {num_actions}")
        print(f"하이퍼파라미터: α={alpha}, γ={gamma}, ε={epsilon_start}→{epsilon_min}")
        print("-" * 70)

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0

        # 에피소드 실행
        while not done:
            # 1. ε-greedy로 행동 선택
            action = epsilon_greedy(Q, state, epsilon, num_actions)

            # 2. 환경에서 행동 실행
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # 3. Q-Learning 업데이트
            # Q(s,a) ← Q(s,a) + α[R + γ·max Q(s',a') - Q(s,a)]
            if not done:
                # 종료 상태가 아니면 다음 상태의 최대 Q값 사용
                best_next_q = np.max(Q[next_state])
            else:
                # 종료 상태면 미래 보상 없음
                best_next_q = 0.0

            # TD Error 계산
            td_target = reward + gamma * best_next_q
            td_error = td_target - Q[state, action]

            # Q-table 업데이트
            Q[state, action] += alpha * td_error

            # 4. 상태 전이
            state = next_state
            total_reward += reward
            steps += 1

        # 에피소드 종료 후 기록
        rewards_history.append(total_reward)
        recent_rewards.append(total_reward)

        # ε 감소 (탐험 줄이기)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        # 주기적으로 성능 평가 및 출력
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


def plot_results(rewards_history, success_history, title="Q-Learning on FrozenLake"):
    """
    학습 결과를 그래프로 시각화

    Args:
        rewards_history: 에피소드별 보상
        success_history: (에피소드, 성공률) 리스트
        title: 그래프 제목
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. 보상 곡선 (이동평균)
    window = 100
    if len(rewards_history) >= window:
        moving_avg = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
        ax1.plot(moving_avg, label=f'{window}-Episode Moving Average')
    ax1.plot(rewards_history, alpha=0.3, label='Episode Reward')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Reward')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 성공률 곡선
    if success_history:
        episodes, success_rates = zip(*success_history)
        ax2.plot(episodes, [sr * 100 for sr in success_rates], marker='o')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_title('Evaluation Success Rate')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig('q_learning_results.png', dpi=100, bbox_inches='tight')
    print("그래프가 'q_learning_results.png'로 저장되었습니다.")
    plt.show()


def visualize_episode(env, Q, max_steps=100):
    """
    학습된 정책으로 한 에피소드를 실행하고 시각화

    Args:
        env: Gymnasium 환경
        Q: 학습된 Q-table
        max_steps: 최대 스텝 수
    """
    state, _ = env.reset()
    done = False
    steps = 0
    total_reward = 0

    print("\n=== 학습된 정책 실행 ===")
    env.render()

    while not done and steps < max_steps:
        action = np.argmax(Q[state])
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1

        env.render()

        if done:
            if reward == 1.0:
                print(f"\n성공! {steps} 스텝 만에 목표 도달")
            else:
                print(f"\n실패! 구멍에 빠짐 (스텝: {steps})")

    if not done:
        print(f"\n시간 초과 ({max_steps} 스텝)")

    print(f"총 보상: {total_reward}")


def main():
    """
    메인 함수: Q-Learning으로 FrozenLake 학습
    """
    print("=" * 70)
    print("FrozenLake Q-Learning 실습")
    print("=" * 70)

    # 환경 생성
    # is_slippery=False: 결정적 환경 (쉬움)
    # is_slippery=True: 확률적 환경 (어려움, 더 현실적)
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode=None)

    print(f"\n환경 정보:")
    print(f"  상태 공간: {env.observation_space.n} (4x4 격자)")
    print(f"  행동 공간: {env.action_space.n} (←↓→↑)")
    print(f"  is_slippery: False (결정적 환경)")
    print()

    # Q-Learning 학습
    Q, rewards_history, success_history = train_q_learning(
        env,
        num_episodes=10000,
        alpha=0.1,          # 학습률
        gamma=0.99,         # 할인율
        epsilon_start=1.0,  # 초기 탐험률
        epsilon_min=0.01,   # 최소 탐험률
        epsilon_decay=0.995,  # 탐험률 감소율
        eval_interval=500,  # 500 에피소드마다 평가
        verbose=True
    )

    # 최종 성능 평가
    print("\n=== 최종 성능 평가 ===")
    final_success_rate = evaluate_policy(env, Q, num_episodes=1000)
    print(f"1000번 테스트 성공률: {final_success_rate*100:.2f}%")

    # 학습된 정책 시각화
    print_policy(Q)

    # Q-table 출력 (일부)
    print("\nQ-table (상위 5개 상태):")
    print("State | Left   Down   Right  Up")
    print("-" * 40)
    for state in range(min(5, Q.shape[0])):
        print(f"  {state:2d}  | ", end='')
        for action in range(Q.shape[1]):
            print(f"{Q[state, action]:6.2f}", end=' ')
        print()

    # 결과 그래프
    plot_results(rewards_history, success_history)

    # 학습된 정책으로 한 에피소드 실행 (시각화)
    # env_vis = gym.make('FrozenLake-v1', is_slippery=False, render_mode='human')
    # visualize_episode(env_vis, Q)
    # env_vis.close()

    env.close()

    print("\n" + "=" * 70)
    print("실습 완료!")
    print("=" * 70)

    # 추가 실험 제안
    print("\n💡 추가 실험 아이디어:")
    print("1. is_slippery=True로 변경하여 확률적 환경에서 학습")
    print("2. alpha를 0.01, 0.3, 0.5로 바꿔가며 학습 속도 비교")
    print("3. gamma를 0.5, 0.9로 바꿔가며 장기 계획의 중요성 확인")
    print("4. epsilon_decay를 0.99, 0.999로 바꿔가며 탐험 영향 확인")
    print("5. 8x8 FrozenLake 환경 도전 (더 큰 맵)")


if __name__ == "__main__":
    main()


"""
실습 문제:

1. 기본 실습:
   - 코드를 실행하고 학습 과정을 관찰하세요
   - 학습된 정책이 최적 경로를 찾았는지 확인하세요
   - Q-table 값들이 어떻게 분포하는지 살펴보세요

2. 하이퍼파라미터 실험:
   a) alpha (학습률) 변화:
      - alpha = 0.01: 어떻게 되나요?
      - alpha = 0.5: 어떻게 되나요?

   b) gamma (할인율) 변화:
      - gamma = 0.5: 어떻게 되나요?
      - gamma = 0.99: 어떻게 되나요?

   c) epsilon_decay 변화:
      - decay = 0.99: 어떻게 되나요?
      - decay = 0.999: 어떻게 되나요?

3. 도전 과제:
   - is_slippery=True로 변경하여 학습
     (힌트: 더 많은 에피소드 필요, 20000+)
   - 성공률 90% 이상 달성하기
   - 학습 곡선을 비교하여 차이점 분석

4. 분석 질문:
   - Q-Learning은 항상 최적 정책을 찾나요?
   - epsilon을 0으로 하면 어떻게 되나요?
   - gamma=1.0으로 하면 어떤 문제가 생기나요?
"""
