#!/usr/bin/env python3
"""
MCTS 에이전트 테스트 스크립트

사용법:
    python3 test_mcts.py
"""

import time
import random
from mcts_agent import AtaxxBoard, MCTSNode, mcts_search


def test_board_basic():
    """보드 기본 기능 테스트"""
    print("=" * 60)
    print("테스트 1: 보드 기본 기능")
    print("=" * 60)

    board = AtaxxBoard()

    # 초기 보드 출력
    print("\n초기 보드:")
    print_board(board)

    # 가능한 수 확인
    moves = board.get_legal_moves()
    print(f"\n가능한 수 개수: {len(moves)}")
    print(f"처음 5개 수: {moves[:5]}")

    # 수 적용 테스트
    if moves:
        move = moves[0]
        print(f"\n수 적용: {move}")
        new_board = board.apply_move(move)
        print_board(new_board)

    print("\n✓ 보드 기본 기능 정상")


def test_mcts_node():
    """MCTSNode 기본 기능 테스트"""
    print("\n" + "=" * 60)
    print("테스트 2: MCTSNode 기능")
    print("=" * 60)

    board = AtaxxBoard()
    root = MCTSNode(board)

    # UCB1 테스트
    print("\n미방문 노드 UCB1:", root.ucb1())
    assert root.ucb1() == float('inf'), "미방문 노드는 무한대여야 함"

    # Expansion 테스트
    child = root.expand()
    print(f"자식 노드 생성: {child.move}")
    assert len(root.children) == 1, "자식이 1개여야 함"

    # Rollout 테스트
    print("\n롤아웃 시작...")
    start = time.time()
    result = child.rollout()
    elapsed = time.time() - start
    print(f"롤아웃 결과: {result} (소요 시간: {elapsed*1000:.1f}ms)")

    # Backpropagation 테스트
    child.backpropagate(result)
    print(f"\n역전파 후:")
    print(f"  자식: wins={child.wins}, visits={child.visits}")
    print(f"  루트: wins={root.wins}, visits={root.visits}")

    # UCB1 재계산
    if child.visits > 0:
        ucb1_value = child.ucb1()
        print(f"  자식 UCB1: {ucb1_value:.3f}")

    print("\n✓ MCTSNode 기능 정상")


def test_mcts_search():
    """MCTS 탐색 테스트"""
    print("\n" + "=" * 60)
    print("테스트 3: MCTS 탐색")
    print("=" * 60)

    board = AtaxxBoard()

    # 다양한 시간 제한으로 테스트
    for time_limit in [10, 50, 100]:
        print(f"\n시간 제한 {time_limit}ms로 탐색...")
        start = time.time()
        move = mcts_search(board, time_limit)
        elapsed = time.time() - start

        print(f"  선택된 수: {move}")
        print(f"  실제 소요 시간: {elapsed*1000:.1f}ms")

        assert move is not None, "수가 선택되어야 함"

    print("\n✓ MCTS 탐색 정상")


def test_game_play():
    """게임 플레이 테스트 (MCTS vs 랜덤)"""
    print("\n" + "=" * 60)
    print("테스트 4: 게임 플레이 (MCTS vs 랜덤)")
    print("=" * 60)

    wins = 0
    losses = 0
    draws = 0
    num_games = 5

    for game_num in range(num_games):
        print(f"\n게임 {game_num + 1}/{num_games}:")

        board = AtaxxBoard()
        mcts_color = 1  # MCTS가 FIRST (1번)

        move_count = 0
        max_moves = 100

        while not board.is_terminal() and move_count < max_moves:
            if board.current_player == mcts_color:
                # MCTS 차례
                move = mcts_search(board, time_limit_ms=50)
            else:
                # 랜덤 차례
                moves = board.get_legal_moves()
                move = random.choice(moves)

            if move is None:
                # PASS
                board.current_player = 3 - board.current_player
            else:
                board._apply_move_inplace(move)

            move_count += 1

        # 결과 확인
        count1, count2 = board.count_pieces()
        print(f"  최종 점수: FIRST(MCTS)={count1}, SECOND(랜덤)={count2}")

        if count1 > count2:
            wins += 1
            result = "승리"
        elif count1 < count2:
            losses += 1
            result = "패배"
        else:
            draws += 1
            result = "무승부"

        print(f"  결과: {result}")

    # 통계 출력
    print(f"\n{'=' * 60}")
    print(f"최종 결과 (MCTS 관점):")
    print(f"  승: {wins}/{num_games} ({wins/num_games*100:.1f}%)")
    print(f"  패: {losses}/{num_games} ({losses/num_games*100:.1f}%)")
    print(f"  무: {draws}/{num_games} ({draws/num_games*100:.1f}%)")
    print(f"{'=' * 60}")

    # MCTS가 랜덤보다 강해야 함
    assert wins >= losses, "MCTS가 랜덤보다 약하면 안 됨!"

    print("\n✓ 게임 플레이 정상 (MCTS가 랜덤을 이김)")


def test_terminal_states():
    """터미널 상태 테스트"""
    print("\n" + "=" * 60)
    print("테스트 5: 터미널 상태")
    print("=" * 60)

    # 빈 보드 (한쪽이 돌이 없음)
    board = AtaxxBoard()
    board.board = [[0] * 7 for _ in range(7)]
    board.board[0][0] = 1  # FIRST만 존재
    print("\n한쪽만 돌이 있는 경우:")
    assert board.is_terminal(), "터미널이어야 함"
    print("  ✓ 터미널 확인됨")

    # 꽉 찬 보드
    board = AtaxxBoard()
    board.board = [[1] * 7 for _ in range(7)]
    print("\n보드가 꽉 찬 경우:")
    assert board.is_terminal(), "터미널이어야 함"
    print("  ✓ 터미널 확인됨")

    print("\n✓ 터미널 상태 테스트 통과")


def print_board(board):
    """보드 출력 (디버깅용)"""
    symbols = {0: '.', 1: 'X', 2: 'O'}
    print("  1 2 3 4 5 6 7")
    for i, row in enumerate(board.board):
        print(f"{i+1} {' '.join(symbols[cell] for cell in row)}")
    count1, count2 = board.count_pieces()
    print(f"X(FIRST)={count1}, O(SECOND)={count2}")


def performance_test():
    """성능 테스트 (선택)"""
    print("\n" + "=" * 60)
    print("성능 테스트: 시뮬레이션 횟수 vs 승률")
    print("=" * 60)

    board = AtaxxBoard()
    time_limits = [10, 50, 100, 200, 500]

    print("\n각 시간 제한으로 5번씩 탐색:")
    for time_limit in time_limits:
        iterations_list = []

        for _ in range(5):
            root = MCTSNode(board)
            start_time = time.time()
            end_time = start_time + time_limit / 1000.0

            iterations = 0
            while time.time() < end_time:
                node = root
                while not node.is_terminal() and node.is_fully_expanded():
                    node = node.select_child()
                if not node.is_terminal() and not node.is_fully_expanded():
                    node = node.expand()
                result = node.rollout()
                node.backpropagate(result)
                iterations += 1

            iterations_list.append(iterations)

        avg_iterations = sum(iterations_list) / len(iterations_list)
        print(f"  {time_limit:3d}ms: 평균 {avg_iterations:6.1f} iterations")

    print("\n✓ 성능 테스트 완료")


def main():
    """모든 테스트 실행"""
    print("\n" + "=" * 60)
    print("MCTS 에이전트 테스트 시작")
    print("=" * 60)

    try:
        # 기본 테스트
        test_board_basic()
        test_mcts_node()
        test_mcts_search()
        test_terminal_states()

        # 게임 플레이 테스트
        test_game_play()

        # 성능 테스트 (선택)
        print("\n성능 테스트를 실행하시겠습니까? (y/n): ", end="")
        response = input().strip().lower()
        if response == 'y':
            performance_test()

        # 최종 결과
        print("\n" + "=" * 60)
        print("✓ 모든 테스트 통과!")
        print("=" * 60)
        print("\nMCTS 에이전트가 정상적으로 작동합니다.")
        print("ALPHANO 플랫폼에 제출할 준비가 되었습니다.")

    except AssertionError as e:
        print(f"\n✗ 테스트 실패: {e}")
        return 1
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
