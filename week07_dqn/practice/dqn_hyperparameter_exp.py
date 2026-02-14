"""
DQN 하이퍼파라미터 실험
Week 7 실습 코드

이 코드는 DQN의 주요 하이퍼파라미터를 변경하며
학습 성능을 비교합니다.

실험 항목:
1. Learning Rate (0.0001, 0.001, 0.01)
2. Target Update Frequency (100, 1000, 10000)
3. Batch Size (16, 32, 64, 128)
4. Replay Buffer Size (1000, 5000, 10000)
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from dqn_cartpole import QNetwork, ReplayBuffer, DQNAgent


def train_with_config(config, num_episodes=300, verbose=False):
    """
    특정 설정으로 DQN 학습

    Args:
        config: 하이퍼파라미터 딕셔너리
        num_episodes: 학습할 에피소드 수
        verbose: 상세 출력 여부

    Returns:
        moving_avg_rewards: 이동평균 보상 리스트
    """
    # 환경 생성
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # 에이전트 생성
    agent = DQNAgent(
        state_size,
        action_size,
        lr=config['lr'],
        gamma=config['gamma']
    )

    # Replay Buffer 크기 변경
    if 'buffer_size' in config:
        agent.replay_buffer = ReplayBuffer(capacity=config['buffer_size'])

    # 학습 변수
    epsilon = config['epsilon_start']
    episode_rewards = []
    moving_avg_rewards = []

    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(500):
            # 행동 선택
            action = agent.select_action(state, epsilon)

            # 환경 실행
            next_state, reward, done, truncated, _ = env.step(action)

            # 경험 저장
            agent.replay_buffer.store(state, action, reward, next_state, done or truncated)

            # 학습
            loss = agent.learn(config['batch_size'])

            # Target Network 업데이트
            agent.step_count += 1
            if agent.step_count % config['target_update_freq'] == 0:
                agent.update_target_network()

            episode_reward += reward
            state = next_state

            if done or truncated:
                break

        # ε 감소
        epsilon = max(config['epsilon_end'], epsilon * config['epsilon_decay'])

        # 기록
        episode_rewards.append(episode_reward)
        moving_avg = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
        moving_avg_rewards.append(moving_avg)

        # 진행 상황 출력
        if verbose and (episode + 1) % 50 == 0:
            print(f"  Episode {episode + 1:3d} | Avg: {moving_avg:6.2f}")

    env.close()
    return moving_avg_rewards


def experiment_learning_rate():
    """
    실험 1: Learning Rate 비교
    """
    print("=" * 60)
    print("실험 1: Learning Rate 비교")
    print("=" * 60)

    learning_rates = [0.0001, 0.001, 0.01]
    results = {}

    base_config = {
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_freq': 1000
    }

    for lr in learning_rates:
        print(f"\nLearning Rate = {lr} 학습 중...")
        config = base_config.copy()
        config['lr'] = lr

        moving_avg = train_with_config(config, num_episodes=300, verbose=True)
        results[lr] = moving_avg

        print(f"  최종 평균: {moving_avg[-1]:.2f}")

    # 시각화
    plt.figure(figsize=(10, 6))
    for lr, rewards in results.items():
        plt.plot(rewards, label=f"lr={lr}", linewidth=2)

    plt.axhline(y=195, color='r', linestyle='--', alpha=0.5, label='Goal (195)')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Moving Average Reward', fontsize=12)
    plt.title('Learning Rate Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp1_learning_rate.png', dpi=150)
    print("\n그래프 저장: exp1_learning_rate.png")
    plt.show()

    return results


def experiment_target_update_freq():
    """
    실험 2: Target Update Frequency 비교
    """
    print("\n" + "=" * 60)
    print("실험 2: Target Update Frequency 비교")
    print("=" * 60)

    update_freqs = [100, 1000, 10000]
    results = {}

    base_config = {
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32
    }

    for freq in update_freqs:
        print(f"\nTarget Update Frequency = {freq} 학습 중...")
        config = base_config.copy()
        config['target_update_freq'] = freq

        moving_avg = train_with_config(config, num_episodes=300, verbose=True)
        results[freq] = moving_avg

        print(f"  최종 평균: {moving_avg[-1]:.2f}")

    # 시각화
    plt.figure(figsize=(10, 6))
    for freq, rewards in results.items():
        plt.plot(rewards, label=f"freq={freq}", linewidth=2)

    plt.axhline(y=195, color='r', linestyle='--', alpha=0.5, label='Goal (195)')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Moving Average Reward', fontsize=12)
    plt.title('Target Update Frequency Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp2_target_update_freq.png', dpi=150)
    print("\n그래프 저장: exp2_target_update_freq.png")
    plt.show()

    return results


def experiment_batch_size():
    """
    실험 3: Batch Size 비교
    """
    print("\n" + "=" * 60)
    print("실험 3: Batch Size 비교")
    print("=" * 60)

    batch_sizes = [16, 32, 64, 128]
    results = {}

    base_config = {
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'target_update_freq': 1000
    }

    for batch_size in batch_sizes:
        print(f"\nBatch Size = {batch_size} 학습 중...")
        config = base_config.copy()
        config['batch_size'] = batch_size

        moving_avg = train_with_config(config, num_episodes=300, verbose=True)
        results[batch_size] = moving_avg

        print(f"  최종 평균: {moving_avg[-1]:.2f}")

    # 시각화
    plt.figure(figsize=(10, 6))
    for batch_size, rewards in results.items():
        plt.plot(rewards, label=f"batch={batch_size}", linewidth=2)

    plt.axhline(y=195, color='r', linestyle='--', alpha=0.5, label='Goal (195)')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Moving Average Reward', fontsize=12)
    plt.title('Batch Size Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp3_batch_size.png', dpi=150)
    print("\n그래프 저장: exp3_batch_size.png")
    plt.show()

    return results


def experiment_buffer_size():
    """
    실험 4: Replay Buffer Size 비교
    """
    print("\n" + "=" * 60)
    print("실험 4: Replay Buffer Size 비교")
    print("=" * 60)

    buffer_sizes = [1000, 5000, 10000]
    results = {}

    base_config = {
        'lr': 0.001,
        'gamma': 0.99,
        'epsilon_start': 1.0,
        'epsilon_end': 0.01,
        'epsilon_decay': 0.995,
        'batch_size': 32,
        'target_update_freq': 1000
    }

    for buffer_size in buffer_sizes:
        print(f"\nBuffer Size = {buffer_size} 학습 중...")
        config = base_config.copy()
        config['buffer_size'] = buffer_size

        moving_avg = train_with_config(config, num_episodes=300, verbose=True)
        results[buffer_size] = moving_avg

        print(f"  최종 평균: {moving_avg[-1]:.2f}")

    # 시각화
    plt.figure(figsize=(10, 6))
    for buffer_size, rewards in results.items():
        plt.plot(rewards, label=f"buffer={buffer_size}", linewidth=2)

    plt.axhline(y=195, color='r', linestyle='--', alpha=0.5, label='Goal (195)')
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Moving Average Reward', fontsize=12)
    plt.title('Replay Buffer Size Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('exp4_buffer_size.png', dpi=150)
    print("\n그래프 저장: exp4_buffer_size.png")
    plt.show()

    return results


def run_all_experiments():
    """
    모든 실험 실행 및 종합 결과 출력
    """
    print("\n" + "=" * 60)
    print("DQN 하이퍼파라미터 실험 시작")
    print("=" * 60)
    print("\n각 실험은 약 5-10분 소요됩니다.")
    print("총 4개 실험 진행 예정\n")

    # 랜덤 시드 설정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 실험 1: Learning Rate
    lr_results = experiment_learning_rate()

    # 실험 2: Target Update Frequency
    freq_results = experiment_target_update_freq()

    # 실험 3: Batch Size
    batch_results = experiment_batch_size()

    # 실험 4: Buffer Size
    buffer_results = experiment_buffer_size()

    # 종합 결과 출력
    print("\n" + "=" * 60)
    print("실험 결과 요약")
    print("=" * 60)

    print("\n1. Learning Rate:")
    for lr, rewards in lr_results.items():
        print(f"   lr={lr:7.4f} → 최종 평균: {rewards[-1]:6.2f}")

    print("\n2. Target Update Frequency:")
    for freq, rewards in freq_results.items():
        print(f"   freq={freq:5d} → 최종 평균: {rewards[-1]:6.2f}")

    print("\n3. Batch Size:")
    for batch, rewards in batch_results.items():
        print(f"   batch={batch:3d} → 최종 평균: {rewards[-1]:6.2f}")

    print("\n4. Replay Buffer Size:")
    for buffer, rewards in buffer_results.items():
        print(f"   buffer={buffer:5d} → 최종 평균: {rewards[-1]:6.2f}")

    # 최적 설정 찾기
    print("\n" + "=" * 60)
    print("권장 설정")
    print("=" * 60)

    best_lr = max(lr_results.items(), key=lambda x: x[1][-1])[0]
    best_freq = max(freq_results.items(), key=lambda x: x[1][-1])[0]
    best_batch = max(batch_results.items(), key=lambda x: x[1][-1])[0]
    best_buffer = max(buffer_results.items(), key=lambda x: x[1][-1])[0]

    print(f"\nLearning Rate:            {best_lr}")
    print(f"Target Update Frequency:  {best_freq}")
    print(f"Batch Size:               {best_batch}")
    print(f"Replay Buffer Size:       {best_buffer}")

    print("\n" + "=" * 60)
    print("모든 실험 완료!")
    print("=" * 60)


def run_single_experiment(experiment_name):
    """
    개별 실험 실행

    Args:
        experiment_name: 'lr', 'freq', 'batch', 'buffer' 중 하나
    """
    # 랜덤 시드 설정
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    experiments = {
        'lr': experiment_learning_rate,
        'freq': experiment_target_update_freq,
        'batch': experiment_batch_size,
        'buffer': experiment_buffer_size
    }

    if experiment_name not in experiments:
        print(f"올바르지 않은 실험 이름: {experiment_name}")
        print(f"가능한 옵션: {list(experiments.keys())}")
        return

    experiments[experiment_name]()


if __name__ == "__main__":
    import sys

    print("""
    ╔════════════════════════════════════════════════════════╗
    ║         DQN 하이퍼파라미터 실험 프로그램              ║
    ╚════════════════════════════════════════════════════════╝
    """)

    if len(sys.argv) > 1:
        # 명령줄 인자로 개별 실험 실행
        experiment_name = sys.argv[1]
        print(f"\n실험 '{experiment_name}' 실행 중...\n")
        run_single_experiment(experiment_name)
    else:
        # 모든 실험 실행
        print("\n사용법:")
        print("  전체 실험:     python dqn_hyperparameter_exp.py")
        print("  개별 실험:     python dqn_hyperparameter_exp.py [lr|freq|batch|buffer]")
        print("\n전체 실험을 시작합니다...\n")

        user_input = input("계속하시겠습니까? (y/n): ")
        if user_input.lower() == 'y':
            run_all_experiments()
        else:
            print("\n실험 취소")

            # 개별 실험 선택 옵션 제공
            print("\n개별 실험을 선택하세요:")
            print("  1. Learning Rate")
            print("  2. Target Update Frequency")
            print("  3. Batch Size")
            print("  4. Replay Buffer Size")
            print("  0. 종료")

            choice = input("\n선택 (0-4): ")

            if choice == '1':
                run_single_experiment('lr')
            elif choice == '2':
                run_single_experiment('freq')
            elif choice == '3':
                run_single_experiment('batch')
            elif choice == '4':
                run_single_experiment('buffer')
            else:
                print("\n프로그램 종료")
