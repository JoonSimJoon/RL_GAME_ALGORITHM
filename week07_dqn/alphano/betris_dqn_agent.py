"""
Betris ALPHANO 에이전트 (DQN 기반 컨셉)
Week 7 ALPHANO 문제 3

주의사항:
1. 이 코드는 Betris의 정확한 프로토콜을 알지 못한 상태에서 작성된 개념 구현입니다.
2. 실제 ALPHANO 제출 시 Betris의 입출력 프로토콜을 정확히 확인해야 합니다.
3. DQN을 실전에 적용하려면 사전 학습(offline training)이 필요합니다.
4. 여기서는 휴리스틱 전략으로 대체하며, DQN 적용 방법을 주석으로 설명합니다.

ALPHANO 기본 프로토콜 (추정):
- READY FIRST/SECOND → 출력: OK
- TURN my_time opp_time → 출력: MOVE <베팅> <배치 정보>
- OPP <상대 행동> → 상태 업데이트
- FINISH <결과> → 종료

Betris 게임 규칙:
- 5×5 보드에 블록 배치
- 줄이 완성되면 제거 후 점수 획득
- 각 턴마다 코인 베팅
- 베팅 금액에 따라 점수 배수 증가
"""

import sys
import random


class BetrisBoard:
    """
    Betris 보드 관리 클래스
    """

    def __init__(self):
        self.board = [[0] * 5 for _ in range(5)]  # 5×5 보드
        self.score = 0
        self.coins = 100  # 초기 코인 (추정)

    def place_block(self, block, row, col):
        """
        블록 배치 (간단한 구현)

        Args:
            block: 블록 정보
            row: 배치 행
            col: 배치 열

        Returns:
            배치 성공 여부
        """
        # 실제로는 블록 형태에 따라 복잡한 로직 필요
        if 0 <= row < 5 and 0 <= col < 5:
            if self.board[row][col] == 0:
                self.board[row][col] = 1
                return True
        return False

    def clear_lines(self):
        """
        완성된 줄 제거 및 점수 계산

        Returns:
            제거된 줄 수
        """
        lines_cleared = 0

        # 가로줄 확인
        for row in range(5):
            if all(self.board[row][col] != 0 for col in range(5)):
                self.board[row] = [0] * 5
                lines_cleared += 1

        # 세로줄 확인
        for col in range(5):
            if all(self.board[row][col] != 0 for row in range(5)):
                for row in range(5):
                    self.board[row][col] = 0
                lines_cleared += 1

        return lines_cleared

    def get_empty_cells(self):
        """빈 칸 개수 반환"""
        count = 0
        for row in range(5):
            for col in range(5):
                if self.board[row][col] == 0:
                    count += 1
        return count


class BetrisAgent:
    """
    Betris 에이전트 (휴리스틱 기반)

    실제 DQN 적용 시:
    1. 사전 학습(Offline Training):
       - Betris 시뮬레이터 구현
       - 수천~수만 에피소드 학습
       - 학습된 Q-Network 저장

    2. 온라인 플레이:
       - 저장된 모델 로드
       - ε=0 (순수 활용)
       - 빠른 추론
    """

    def __init__(self):
        self.board = BetrisBoard()
        self.turn_count = 0

    def select_bet(self):
        """
        베팅 금액 선택 (휴리스틱)

        DQN 적용 시:
        - 상태: [board, score, coins, turn_count]
        - 행동: [0, 1, 2, 5, 10]
        - Q-Network로 베팅 금액 선택

        현재 전략:
        - 보수적 베팅 (안정성 우선)
        """
        empty_cells = self.board.get_empty_cells()

        # 빈 칸이 많으면 적게 베팅
        if empty_cells > 15:
            bet = min(1, self.board.coins)
        elif empty_cells > 10:
            bet = min(2, self.board.coins)
        elif empty_cells > 5:
            bet = min(5, self.board.coins)
        else:
            # 보드가 거의 차면 큰 베팅
            bet = min(10, self.board.coins)

        return bet

    def select_placement(self, block):
        """
        블록 배치 위치 선택 (휴리스틱)

        DQN 적용 시:
        - 상태: [board, block]
        - 행동: [(row, col, rotation) for all valid positions]
        - Q-Network로 최선의 배치 선택

        현재 전략:
        - Greedy: 즉시 줄을 완성할 수 있는 위치 우선
        - 없으면 빈 칸 중 랜덤
        """
        # 전략 1: 줄을 완성할 수 있는 위치 찾기
        for row in range(5):
            for col in range(5):
                if self.board.board[row][col] == 0:
                    # 이 위치에 놓으면 줄이 완성되는지 확인
                    if self.would_complete_line(row, col):
                        return (row, col, 0)  # rotation=0

        # 전략 2: 빈 칸 중 랜덤
        empty_positions = []
        for row in range(5):
            for col in range(5):
                if self.board.board[row][col] == 0:
                    empty_positions.append((row, col))

        if empty_positions:
            row, col = random.choice(empty_positions)
            return (row, col, 0)

        # 더 이상 놓을 곳이 없음
        return (0, 0, 0)

    def would_complete_line(self, row, col):
        """
        특정 위치에 블록을 놓으면 줄이 완성되는지 확인

        Args:
            row: 행
            col: 열

        Returns:
            줄 완성 여부
        """
        # 가로줄 확인
        row_complete = True
        for c in range(5):
            if c != col and self.board.board[row][c] == 0:
                row_complete = False
                break

        # 세로줄 확인
        col_complete = True
        for r in range(5):
            if r != row and self.board.board[r][col] == 0:
                col_complete = False
                break

        return row_complete or col_complete

    def make_move(self, block):
        """
        행동 결정 (베팅 + 배치)

        Returns:
            (bet, row, col, rotation)
        """
        bet = self.select_bet()
        row, col, rotation = self.select_placement(block)
        return (bet, row, col, rotation)


# ============================================================================
# DQN 적용 시 필요한 구성요소 (주석으로 설명)
# ============================================================================

"""
1. Q-Network 구조 (예시)

import torch
import torch.nn as nn

class BetrisQNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # 보드를 CNN으로 처리
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Flatten 후 FC
        self.fc1 = nn.Linear(64 * 5 * 5 + 4, 256)  # +4: score, coins, turn, block
        self.fc2 = nn.Linear(256, 256)

        # 베팅 출력 (5가지: 0, 1, 2, 5, 10)
        self.bet_head = nn.Linear(256, 5)

        # 배치 출력 (5×5×4 = 100가지)
        self.place_head = nn.Linear(256, 100)

    def forward(self, board, meta):
        # board: (batch, 1, 5, 5)
        # meta: (batch, 4) - [score, coins, turn, block_type]

        x = F.relu(self.conv1(board))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        x = torch.cat([x, meta], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        bet_q = self.bet_head(x)
        place_q = self.place_head(x)

        return bet_q, place_q


2. 학습 과정 (Offline Training)

def train_betris_dqn():
    # 환경 시뮬레이터
    env = BetrisSimulator()

    # DQN 에이전트
    agent = DQNAgent(state_size=..., action_size=...)

    # 학습 루프
    for episode in range(10000):
        state = env.reset()

        for t in range(max_steps):
            # 1. 행동 선택 (ε-greedy)
            action = agent.select_action(state, epsilon)

            # 2. 환경 실행
            next_state, reward, done = env.step(action)

            # 3. 경험 저장
            agent.replay_buffer.store(state, action, reward, next_state, done)

            # 4. 학습
            agent.learn(batch_size=32)

            # 5. Target Network 업데이트
            if t % 1000 == 0:
                agent.update_target_network()

            state = next_state
            if done:
                break

        # 진행 상황 출력
        if episode % 100 == 0:
            print(f"Episode {episode}, Avg Score: {avg_score}")

    # 모델 저장
    torch.save(agent.q_network.state_dict(), 'betris_dqn.pth')


3. 온라인 플레이 (Competition)

class BetrisDQNAgent:
    def __init__(self):
        # 학습된 모델 로드
        self.q_network = BetrisQNetwork()
        self.q_network.load_state_dict(torch.load('betris_dqn.pth'))
        self.q_network.eval()

    def select_action(self, state):
        # ε=0 (순수 활용)
        with torch.no_grad():
            board_tensor = self.state_to_tensor(state)
            bet_q, place_q = self.q_network(board_tensor)

            bet = bet_q.argmax().item()  # 최선의 베팅
            place = place_q.argmax().item()  # 최선의 배치

            return bet, place


4. 보상 설계 (Reward Shaping)

def calculate_reward(state, action, next_state):
    reward = 0

    # 1. 줄 제거 보너스
    lines_cleared = next_state.lines_cleared - state.lines_cleared
    reward += 10 * lines_cleared

    # 2. 점수 증가
    score_gained = next_state.score - state.score
    reward += score_gained

    # 3. 빈 공간 유지 보너스
    empty_cells = next_state.get_empty_cells()
    reward += 0.1 * empty_cells

    # 4. 게임 오버 페널티
    if next_state.game_over:
        reward -= 50

    return reward


5. 학습 팁

- Episode 수: 최소 10000+
- Replay Buffer: 100000
- Batch Size: 64
- Learning Rate: 0.0001 (낮게 시작)
- Epsilon Decay: 0.999 (천천히)
- Target Update: 2000 steps

- 다양한 초기 상태에서 학습
- 적절한 Reward Shaping
- Curriculum Learning (쉬운 것부터)
"""


# ============================================================================
# ALPHANO 프로토콜 처리 (메인 로직)
# ============================================================================

def main():
    """
    ALPHANO 메인 함수
    """
    agent = BetrisAgent()

    # 표준 입력으로부터 명령 읽기
    for line in sys.stdin:
        line = line.strip()

        if line.startswith("READY"):
            # 게임 시작
            parts = line.split()
            position = parts[1]  # FIRST or SECOND
            print("OK")
            sys.stdout.flush()

        elif line.startswith("TURN"):
            # 내 턴
            parts = line.split()
            my_time = int(parts[1])
            opp_time = int(parts[2])

            # 블록 정보 읽기 (다음 줄에서 올 것으로 가정)
            # 실제 프로토콜에 따라 수정 필요
            block_line = sys.stdin.readline().strip()
            block_info = parse_block_info(block_line)

            # 행동 결정
            bet, row, col, rotation = agent.make_move(block_info)

            # 실제 배치 (내부 상태 업데이트)
            agent.board.place_block(block_info, row, col)
            agent.board.coins -= bet

            # 줄 제거 및 점수 계산
            lines_cleared = agent.board.clear_lines()
            if lines_cleared > 0:
                agent.board.score += lines_cleared * 10 * bet

            agent.turn_count += 1

            # 행동 출력 (실제 프로토콜에 맞게 수정 필요)
            print(f"MOVE {bet} {row} {col} {rotation}")
            sys.stdout.flush()

        elif line.startswith("OPP"):
            # 상대 행동 (필요시 상태 업데이트)
            parts = line.split()
            # 상대 행동 파싱 및 처리
            pass

        elif line.startswith("FINISH"):
            # 게임 종료
            parts = line.split()
            result = parts[1]  # WIN, LOSE, DRAW
            break

        else:
            # 알 수 없는 명령
            pass


def parse_block_info(line):
    """
    블록 정보 파싱 (프로토콜에 따라 구현)

    Args:
        line: 블록 정보 문자열

    Returns:
        블록 정보 딕셔너리
    """
    # 실제 프로토콜에 맞게 구현 필요
    # 예시:
    # "BLOCK I" -> {'type': 'I', 'shape': [...]}
    return {'type': 'unknown', 'shape': []}


# ============================================================================
# 실행
# ============================================================================

if __name__ == "__main__":
    # 실제 제출용
    main()

    # 로컬 테스트용 (주석 해제하여 사용)
    # test_agent()


def test_agent():
    """
    로컬 테스트 함수
    """
    print("=" * 60)
    print("Betris Agent 로컬 테스트")
    print("=" * 60)

    agent = BetrisAgent()

    # 시뮬레이션 예시
    for turn in range(10):
        print(f"\n===== Turn {turn + 1} =====")

        # 더미 블록
        block = {'type': 'I', 'shape': [[1, 1, 1, 1]]}

        # 행동 결정
        bet, row, col, rotation = agent.make_move(block)

        print(f"Bet: {bet} coins")
        print(f"Place: ({row}, {col}), Rotation: {rotation}")

        # 배치
        agent.board.place_block(block, row, col)
        agent.board.coins -= bet

        # 줄 제거
        lines = agent.board.clear_lines()
        if lines > 0:
            score_gained = lines * 10 * bet
            agent.board.score += score_gained
            print(f"Lines cleared: {lines}, Score gained: {score_gained}")

        # 보드 출력
        print("\nBoard:")
        for row in agent.board.board:
            print(" ".join(str(cell) for cell in row))

        print(f"\nScore: {agent.board.score}, Coins: {agent.board.coins}")

    print("\n" + "=" * 60)
    print("테스트 완료")
    print("=" * 60)


"""
실제 ALPHANO 제출 전 체크리스트:

[ ] Betris 공식 프로토콜 확인
[ ] 블록 종류 및 형태 파악
[ ] 베팅 규칙 상세 확인
[ ] 점수 계산 방식 확인
[ ] 입출력 형식 정확히 매칭
[ ] 시간 제한 내 동작 확인
[ ] 예외 처리 추가

DQN 적용 시 추가 체크리스트:

[ ] Betris 시뮬레이터 구현
[ ] 충분한 학습 (10000+ episodes)
[ ] 학습 곡선 확인
[ ] 다양한 상황 테스트
[ ] 모델 크기 최적화 (추론 속도)
[ ] 모델 파일 제출 방법 확인
"""
