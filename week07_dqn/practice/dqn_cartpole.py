"""
CartPole DQN 완전 구현
Week 7 실습 코드

이 코드는 CartPole-v1 환경에서 DQN을 사용하여 학습합니다.
목표: 막대를 쓰러뜨리지 않고 200 타임스텝 이상 유지하기

주요 구성요소:
1. QNetwork: Q값을 근사하는 신경망
2. ReplayBuffer: 경험을 저장하고 샘플링
3. DQNAgent: 행동 선택 및 학습
4. 학습 루프: 에피소드 실행 및 에이전트 학습
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


class QNetwork(nn.Module):
    """
    Q-Network: 상태를 입력받아 각 행동의 Q값을 출력

    구조:
    - Input layer: state_size (CartPole: 4)
    - Hidden layer 1: 128 neurons + ReLU
    - Hidden layer 2: 128 neurons + ReLU
    - Output layer: action_size (CartPole: 2)
    """

    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        """
        순전파

        Args:
            x: 상태 (batch_size, state_size)

        Returns:
            Q값 (batch_size, action_size)
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # 출력층은 활성화 함수 없음


class ReplayBuffer:
    """
    Experience Replay Buffer

    경험 (s, a, r, s', done)을 저장하고 랜덤 샘플링을 제공합니다.
    capacity에 도달하면 가장 오래된 경험부터 자동으로 삭제됩니다.
    """

    def __init__(self, capacity=10000):
        """
        Args:
            capacity: 버퍼의 최대 크기
        """
        self.buffer = deque(maxlen=capacity)

    def store(self, state, action, reward, next_state, done):
        """
        경험 저장

        Args:
            state: 현재 상태
            action: 선택한 행동
            reward: 받은 보상
            next_state: 다음 상태
            done: 에피소드 종료 여부
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        랜덤 샘플링

        Args:
            batch_size: 샘플링할 경험의 개수

        Returns:
            (states, actions, rewards, next_states, dones) 튜플
            각각 텐서로 변환되어 반환됨
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def size(self):
        """버퍼에 저장된 경험의 개수"""
        return len(self.buffer)


class DQNAgent:
    """
    DQN 에이전트

    Q-Network를 사용하여 행동을 선택하고 학습합니다.
    Experience Replay와 Target Network를 사용합니다.
    """

    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99):
        """
        Args:
            state_size: 상태의 차원
            action_size: 행동의 개수
            lr: 학습률
            gamma: 할인율
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        # Q-Network (학습용)
        self.q_network = QNetwork(state_size, action_size)

        # Target Network (목표값 계산용)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # 평가 모드로 설정

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=10000)

        # 스텝 카운터 (Target Network 업데이트용)
        self.step_count = 0

    def select_action(self, state, epsilon):
        """
        ε-greedy 정책으로 행동 선택

        Args:
            state: 현재 상태
            epsilon: 탐험 확률

        Returns:
            선택된 행동 (0 또는 1)
        """
        if random.random() < epsilon:
            # 탐험: 랜덤 행동
            return random.randint(0, self.action_size - 1)
        else:
            # 활용: Q값이 높은 행동
            with torch.no_grad():
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return q_values.argmax(1).item()

    def learn(self, batch_size):
        """
        미니배치를 사용하여 Q-Network 학습

        Args:
            batch_size: 배치 크기

        Returns:
            loss 값 (float), 버퍼가 부족하면 None
        """
        # 버퍼에 충분한 경험이 쌓일 때까지 대기
        if self.replay_buffer.size() < batch_size:
            return None

        # 미니배치 샘플링
        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(batch_size)

        # 현재 Q값 계산 (메인 네트워크)
        # gather: 각 상태에서 실제로 선택한 행동의 Q값만 가져오기
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 목표 Q값 계산 (타겟 네트워크)
        with torch.no_grad():  # 그래디언트 계산 안 함
            # 다음 상태의 최대 Q값
            max_next_q = self.target_network(next_states).max(1)[0]
            # Bellman 방정식: r + γ * max_a' Q(s', a')
            # done이면 다음 상태가 없으므로 r만 사용
            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # Loss 계산 (MSE)
        loss = nn.MSELoss()(current_q, target_q)

        # 역전파 및 가중치 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Target Network를 Q-Network로 업데이트
        (가중치 복사)
        """
        self.target_network.load_state_dict(self.q_network.state_dict())

    def save(self, path):
        """모델 저장"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        """모델 로드"""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])


def train_dqn(num_episodes=500,
              max_steps=500,
              batch_size=32,
              epsilon_start=1.0,
              epsilon_end=0.01,
              epsilon_decay=0.995,
              target_update_freq=1000,
              render=False):
    """
    DQN 학습 함수

    Args:
        num_episodes: 학습할 에피소드 수
        max_steps: 에피소드당 최대 스텝
        batch_size: 미니배치 크기
        epsilon_start: 초기 탐험 확률
        epsilon_end: 최소 탐험 확률
        epsilon_decay: 탐험 확률 감소율
        target_update_freq: Target Network 업데이트 주기
        render: 환경 렌더링 여부

    Returns:
        episode_rewards: 에피소드별 보상 리스트
        moving_avg_rewards: 100 에피소드 이동평균 리스트
    """
    # 환경 생성
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')

    # 에이전트 생성
    state_size = env.observation_space.shape[0]  # 4
    action_size = env.action_space.n  # 2
    agent = DQNAgent(state_size, action_size)

    # 학습 변수
    epsilon = epsilon_start
    episode_rewards = []
    moving_avg_rewards = []

    print("=" * 60)
    print("DQN 학습 시작")
    print("=" * 60)
    print(f"에피소드 수: {num_episodes}")
    print(f"배치 크기: {batch_size}")
    print(f"학습률: {agent.optimizer.param_groups[0]['lr']}")
    print(f"Gamma: {agent.gamma}")
    print(f"Target 업데이트 주기: {target_update_freq}")
    print("=" * 60)

    # 학습 루프
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(max_steps):
            # 행동 선택
            action = agent.select_action(state, epsilon)

            # 환경 실행
            next_state, reward, done, truncated, _ = env.step(action)

            # Replay Buffer에 저장
            agent.replay_buffer.store(state, action, reward, next_state, done or truncated)

            # 학습
            loss = agent.learn(batch_size)

            # Target Network 업데이트
            agent.step_count += 1
            if agent.step_count % target_update_freq == 0:
                agent.update_target_network()

            episode_reward += reward
            state = next_state

            if done or truncated:
                break

        # ε 감소 (탐험 → 활용)
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        # 기록
        episode_rewards.append(episode_reward)

        # 이동평균 계산 (최근 100 에피소드)
        moving_avg = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
        moving_avg_rewards.append(moving_avg)

        # 진행 상황 출력
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d} | "
                  f"Reward: {episode_reward:6.2f} | "
                  f"Avg (100): {moving_avg:6.2f} | "
                  f"ε: {epsilon:.4f} | "
                  f"Buffer: {agent.replay_buffer.size():5d}")

        # 목표 달성 확인 (100 에피소드 평균 195 이상)
        if moving_avg >= 195.0:
            print("\n" + "=" * 60)
            print(f"목표 달성! Episode {episode + 1}에서 평균 {moving_avg:.2f} 달성!")
            print("=" * 60)
            break

    env.close()

    print("\n학습 완료!")
    print(f"최종 100 에피소드 평균: {moving_avg_rewards[-1]:.2f}")

    return episode_rewards, moving_avg_rewards, agent


def plot_results(episode_rewards, moving_avg_rewards):
    """
    학습 결과 시각화

    Args:
        episode_rewards: 에피소드별 보상
        moving_avg_rewards: 이동평균 보상
    """
    plt.figure(figsize=(12, 5))

    # 왼쪽: 에피소드별 보상
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards, alpha=0.3, label='Episode Reward')
    plt.plot(moving_avg_rewards, label='Moving Average (100 episodes)', linewidth=2)
    plt.axhline(y=195, color='r', linestyle='--', label='Goal (195)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('DQN Training on CartPole-v1')
    plt.legend()
    plt.grid(alpha=0.3)

    # 오른쪽: 이동평균만
    plt.subplot(1, 2, 2)
    plt.plot(moving_avg_rewards, linewidth=2)
    plt.axhline(y=195, color='r', linestyle='--', label='Goal (195)')
    plt.xlabel('Episode')
    plt.ylabel('Moving Average Reward')
    plt.title('Moving Average Progress')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_cartpole_results.png', dpi=150)
    print("\n그래프 저장 완료: dqn_cartpole_results.png")
    plt.show()


def test_agent(agent, num_episodes=10, render=True):
    """
    학습된 에이전트 테스트

    Args:
        agent: 학습된 DQNAgent
        num_episodes: 테스트할 에피소드 수
        render: 렌더링 여부
    """
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')

    print("\n" + "=" * 60)
    print("학습된 에이전트 테스트")
    print("=" * 60)

    test_rewards = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0

        for t in range(500):
            # ε=0으로 행동 선택 (순수 활용)
            action = agent.select_action(state, epsilon=0.0)
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward

            if done or truncated:
                break

        test_rewards.append(episode_reward)
        print(f"Test Episode {episode + 1}: Reward = {episode_reward}")

    env.close()

    avg_reward = sum(test_rewards) / len(test_rewards)
    print("=" * 60)
    print(f"평균 테스트 보상: {avg_reward:.2f}")
    print("=" * 60)

    return test_rewards


if __name__ == "__main__":
    # 랜덤 시드 설정 (재현성)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 학습
    episode_rewards, moving_avg_rewards, agent = train_dqn(
        num_episodes=500,
        max_steps=500,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        target_update_freq=1000,
        render=False  # 학습 중에는 렌더링 끄기 (속도 향상)
    )

    # 결과 시각화
    plot_results(episode_rewards, moving_avg_rewards)

    # 모델 저장
    agent.save('dqn_cartpole.pth')
    print("\n모델 저장 완료: dqn_cartpole.pth")

    # 학습된 에이전트 테스트
    print("\n학습된 에이전트를 테스트합니다...")
    test_rewards = test_agent(agent, num_episodes=10, render=False)

    print("\n" + "=" * 60)
    print("모든 작업 완료!")
    print("=" * 60)
    print("\n다음을 시도해보세요:")
    print("1. 하이퍼파라미터를 변경하여 재학습")
    print("2. dqn_hyperparameter_exp.py로 하이퍼파라미터 비교")
    print("3. 학습된 모델로 렌더링하며 플레이 관찰")
    print("=" * 60)
