# 백준 7682번: 틱택토
# 난이도: Silver II
# 분류: 게임 이론 / 구현 / 시뮬레이션
#
# 풀이 핵심:
# - X가 먼저 시작하므로 X의 개수 >= O의 개수이어야 함
# - X가 이긴 경우: X 개수 = O 개수 + 1, O는 이기지 않아야 함
# - O가 이긴 경우: X 개수 = O 개수, X는 이기지 않아야 함
# - 무승부 (보드가 꽉 찬 경우): 9개 모두 채워지고 승자가 없어야 함

def check_winner(board):
    """
    보드에서 승자가 있는지 확인
    반환값: 'X', 'O', 또는 None
    """
    lines = []

    # 가로줄 3개
    for i in range(3):
        lines.append(board[i*3:i*3+3])

    # 세로줄 3개
    for i in range(3):
        lines.append(board[i] + board[i+3] + board[i+6])

    # 대각선 2개
    lines.append(board[0] + board[4] + board[8])
    lines.append(board[2] + board[4] + board[6])

    # 각 라인 체크
    winners = set()
    for line in lines:
        if line == 'XXX':
            winners.add('X')
        elif line == 'OOO':
            winners.add('O')

    if 'X' in winners and 'O' in winners:
        return 'both'  # 두 명 다 이긴 경우 (불가능한 상태)
    elif 'X' in winners:
        return 'X'
    elif 'O' in winners:
        return 'O'
    else:
        return None

def is_valid(board):
    """
    주어진 틱택토 보드 상태가 유효한 종료 상태인지 판단
    """
    x_count = board.count('X')
    o_count = board.count('O')

    # 기본 규칙: X가 먼저 시작하므로 X 개수 >= O 개수
    # X와 O의 개수 차이는 0 또는 1이어야 함
    if x_count < o_count or x_count > o_count + 1:
        return False

    winner = check_winner(board)

    # 두 명 다 이긴 경우는 불가능
    if winner == 'both':
        return False

    # X가 이긴 경우
    if winner == 'X':
        # X가 이기면 게임이 끝나므로 X 개수 = O 개수 + 1
        return x_count == o_count + 1

    # O가 이긴 경우
    if winner == 'O':
        # O가 이기면 게임이 끝나므로 X 개수 = O 개수
        return x_count == o_count

    # 승자가 없는 경우
    # 보드가 꽉 차야 함 (9개 모두 채워짐)
    return x_count + o_count == 9

# 메인 로직
while True:
    board = input().strip()
    if board == "end":
        break

    if is_valid(board):
        print("valid")
    else:
        print("invalid")
