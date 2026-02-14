"""
쥐를 잡자 - ALPHANO 휴리스틱 에이전트

게임 규칙 (추정):
- 보드: 7×11 격자
- 고양이 (선공): 쥐를 잡는 것이 목표
- 쥐 (후공): 고양이를 피하는 것이 목표
- 4방향 이동 가능

전략:
- 고양이: 쥐와의 거리를 최소화 (Manhattan Distance)
- 쥐: 고양이와의 거리를 최대화

주의: 실제 ALPHANO 문제의 세부 규칙을 확인하고
      프로토콜에 맞게 수정이 필요할 수 있습니다.

ALPHANO 표준 프로토콜:
- 입력: READY FIRST 또는 READY SECOND
- 출력: OK
- 입력: TURN my_time opp_time
- 출력: MOVE x y (1-indexed 좌표)
- 입력: OPP x y
- 출력: (상대 이동 정보 업데이트)
- 입력: FINISH
- 출력: (종료)
"""

import sys


class CatchMouseAgent:
    """
    쥐를 잡자 휴리스틱 에이전트

    간단한 거리 기반 전략:
    - 고양이: 쥐에게 가까이 가기
    - 쥐: 고양이에게서 멀어지기
    """

    def __init__(self, board_height=7, board_width=11):
        """
        Args:
            board_height: 보드 높이 (기본 7)
            board_width: 보드 너비 (기본 11)
        """
        self.height = board_height
        self.width = board_width
        self.is_first = None  # True: 고양이(선공), False: 쥐(후공)
        self.my_pos = None    # (x, y) 1-indexed
        self.opp_pos = None   # (x, y) 1-indexed

        # 디버그 모드 (stderr로 출력)
        self.debug = True

    def log(self, message):
        """디버그 메시지를 stderr로 출력"""
        if self.debug:
            print(f"[DEBUG] {message}", file=sys.stderr, flush=True)

    def manhattan_distance(self, pos1, pos2):
        """
        Manhattan Distance 계산

        Args:
            pos1: (x, y)
            pos2: (x, y)

        Returns:
            거리 (int)
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def is_valid_position(self, x, y):
        """
        유효한 위치인지 확인

        Args:
            x, y: 1-indexed 좌표

        Returns:
            True if valid
        """
        return 1 <= x <= self.width and 1 <= y <= self.height

    def get_neighbors(self, pos):
        """
        주변 4방향 이동 가능 위치 반환

        Args:
            pos: (x, y) 현재 위치

        Returns:
            [(x, y), ...] 이동 가능한 위치들
        """
        x, y = pos
        directions = [
            (0, -1),  # 위
            (0, 1),   # 아래
            (-1, 0),  # 왼쪽
            (1, 0)    # 오른쪽
        ]

        neighbors = []
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            if self.is_valid_position(new_x, new_y):
                neighbors.append((new_x, new_y))

        return neighbors

    def get_best_move(self):
        """
        최선의 이동 계산

        고양이: 쥐와 가장 가까워지는 이동
        쥐: 고양이와 가장 멀어지는 이동

        Returns:
            (x, y) 이동할 위치 (1-indexed)
        """
        if self.my_pos is None or self.opp_pos is None:
            self.log("Error: 위치 정보 없음!")
            return self.my_pos

        neighbors = self.get_neighbors(self.my_pos)

        if not neighbors:
            self.log("Warning: 이동 가능한 곳이 없음!")
            return self.my_pos

        best_pos = None
        best_distance = float('inf') if self.is_first else -float('inf')

        for neighbor in neighbors:
            dist = self.manhattan_distance(neighbor, self.opp_pos)

            if self.is_first:  # 고양이: 거리 최소화
                if dist < best_distance:
                    best_distance = dist
                    best_pos = neighbor
            else:  # 쥐: 거리 최대화
                if dist > best_distance:
                    best_distance = dist
                    best_pos = neighbor

        if best_pos is None:
            best_pos = neighbors[0]  # 기본값

        self.log(f"내 위치: {self.my_pos}, 상대 위치: {self.opp_pos}")
        self.log(f"선택: {best_pos}, 거리: {best_distance}")

        return best_pos

    def handle_ready(self, command):
        """
        READY 명령 처리

        Args:
            command: "READY FIRST" 또는 "READY SECOND"
        """
        parts = command.strip().split()
        if len(parts) != 2 or parts[0] != "READY":
            self.log(f"Invalid READY command: {command}")
            return

        if parts[1] == "FIRST":
            self.is_first = True
            # 고양이 초기 위치 (추정: 왼쪽 중앙)
            self.my_pos = (1, self.height // 2)
            # 쥐 초기 위치 (추정: 오른쪽 중앙)
            self.opp_pos = (self.width, self.height // 2)
            self.log("고양이 (선공) 시작")
        else:
            self.is_first = False
            # 쥐 초기 위치
            self.my_pos = (self.width, self.height // 2)
            # 고양이 초기 위치
            self.opp_pos = (1, self.height // 2)
            self.log("쥐 (후공) 시작")

        print("OK", flush=True)

    def handle_turn(self, command):
        """
        TURN 명령 처리

        Args:
            command: "TURN my_time opp_time"
        """
        parts = command.strip().split()
        if len(parts) != 3 or parts[0] != "TURN":
            self.log(f"Invalid TURN command: {command}")
            return

        my_time = int(parts[1])
        opp_time = int(parts[2])
        self.log(f"시간: 나={my_time}ms, 상대={opp_time}ms")

        # 최선의 이동 계산
        new_pos = self.get_best_move()

        # 이동 명령 출력
        # ALPHANO 프로토콜에 따라 "MOVE x y" 형식
        # 주의: 실제 문제의 프로토콜 확인 필요!
        print(f"MOVE {new_pos[0]} {new_pos[1]}", flush=True)

        # 내 위치 업데이트
        self.my_pos = new_pos
        self.log(f"이동: {new_pos}")

    def handle_opp(self, command):
        """
        OPP 명령 처리 (상대방 이동)

        Args:
            command: "OPP x y"
        """
        parts = command.strip().split()
        if len(parts) != 3 or parts[0] != "OPP":
            self.log(f"Invalid OPP command: {command}")
            return

        x = int(parts[1])
        y = int(parts[2])
        self.opp_pos = (x, y)
        self.log(f"상대 이동: {self.opp_pos}")

    def handle_finish(self):
        """FINISH 명령 처리"""
        self.log("게임 종료")
        sys.exit(0)

    def run(self):
        """
        메인 루프: 표준 입력에서 명령을 읽고 처리
        """
        self.log("쥐를 잡자 에이전트 시작")

        try:
            while True:
                # 명령 읽기
                line = sys.stdin.readline()
                if not line:
                    break

                command = line.strip()
                self.log(f"수신: {command}")

                # 명령 처리
                if command.startswith("READY"):
                    self.handle_ready(command)
                elif command.startswith("TURN"):
                    self.handle_turn(command)
                elif command.startswith("OPP"):
                    self.handle_opp(command)
                elif command == "FINISH":
                    self.handle_finish()
                else:
                    self.log(f"Unknown command: {command}")

        except Exception as e:
            self.log(f"Error: {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)


def main():
    """
    메인 함수
    """
    # 보드 크기 설정 (실제 문제에 맞게 수정)
    agent = CatchMouseAgent(board_height=7, board_width=11)
    agent.run()


if __name__ == "__main__":
    main()


"""
사용 방법:

1. 로컬 테스트:
   - 테스트 입력 파일 준비 (test_input.txt):
     READY FIRST
     TURN 1000 1000
     OPP 11 4
     TURN 1000 1000
     OPP 10 4
     FINISH

   - 실행:
     python catch_mouse_heuristic.py < test_input.txt

2. ALPHANO 제출:
   - 실제 문제의 프로토콜 확인
   - 필요시 코드 수정
   - 제출

3. 개선 아이디어:
   - 벽이나 장애물 고려
   - 코너로 몰기 전략 (고양이)
   - 코너 피하기 전략 (쥐)
   - 상대방의 다음 수 예측
   - Minimax 알고리즘 적용
   - Q-Learning으로 학습

4. Q-Learning 적용 (도전 과제):
   - 상태: (cat_x, cat_y, mouse_x, mouse_y) 또는 상대적 위치
   - 행동: 4방향 이동
   - 보상: 고양이=잡으면 +100, 쥐=잡히면 -100
   - Self-play로 학습

주의사항:
- 실제 ALPHANO 문제의 규칙을 정확히 확인하세요
- 초기 위치, 이동 규칙, 승리 조건 등
- 프로토콜 형식 (MOVE 명령 등)
- 시간 제한 고려
"""
