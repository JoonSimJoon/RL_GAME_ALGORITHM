# 백준 9657번: 돌 게임 3
# 난이도: Silver III
# 분류: 게임 이론 / DP
#
# 풀이 핵심:
# - N개의 돌, 한 턴에 1개, 3개, 또는 4개 가져갈 수 있음
# - 마지막 돌을 가져가는 사람이 승리
# - 상근이가 먼저 시작
#
# DP 접근:
# - dp[i] = True: i개 남았을 때 현재 차례 플레이어가 승리
# - dp[i] = False: i개 남았을 때 현재 차례 플레이어가 패배
#
# 점화식:
# - dp[i] = True if (dp[i-1] == False or dp[i-3] == False or dp[i-4] == False)
# - 상대방을 패배 상태로 만들 수 있는 선택지가 하나라도 있으면 승리

N = int(input())

# DP 테이블 초기화
dp = [False] * (N + 5)

# 기저 사례
dp[1] = True   # 1개 남음 → 1개 가져가서 승리
dp[2] = False  # 2개 남음 → 1개 가져감 → 상대가 1개 가져가서 승리 → 패배
dp[3] = True   # 3개 남음 → 3개 가져가서 승리
dp[4] = True   # 4개 남음 → 4개 가져가서 승리

# DP 테이블 채우기
for i in range(5, N + 1):
    # 1개, 3개, 또는 4개 가져갔을 때 상대방이 패배 상태가 되면 승리
    if not dp[i-1] or not dp[i-3] or not dp[i-4]:
        dp[i] = True
    else:
        dp[i] = False

# 결과 출력
if dp[N]:
    print("SK")  # 상근 승리
else:
    print("CY")  # 창영 승리
