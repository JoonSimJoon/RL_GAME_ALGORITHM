"""
전체 구현 테스트 스크립트
모든 주요 기능이 정상 작동하는지 확인
"""

import sys
from gridworld import GridWorld
from value_iteration import value_iteration as vi
from policy_iteration import policy_iteration as pi


def test_gridworld():
    """GridWorld 환경 테스트"""
    print("=" * 60)
    print("테스트 1: GridWorld 환경")
    print("=" * 60)

    try:
        env = GridWorld(grid_size=4)
        print("✓ GridWorld 생성 성공")

        # 초기화 테스트
        state = env.reset()
        assert state == (0, 0), "시작 위치가 (0, 0)이 아닙니다"
        print("✓ 환경 초기화 성공")

        # 이동 테스트
        next_state, reward, done = env.step(state, GridWorld.ACTION_RIGHT)
        assert next_state == (0, 1), "오른쪽 이동 실패"
        assert not done, "게임이 너무 빨리 종료됨"
        print("✓ 행동 수행 성공")

        # 목표 도달 테스트
        next_state, reward, done = env.step(env.goal, GridWorld.ACTION_UP)
        assert done, "목표 상태가 종료 상태가 아닙니다"
        assert reward == 1.0, f"목표 보상이 잘못됨: {reward}"
        print("✓ 목표 도달 테스트 성공")

        # 장애물 테스트
        next_state, reward, done = env.step((1, 1), GridWorld.ACTION_UP)
        assert done, "장애물이 종료 상태가 아닙니다"
        assert reward == -1.0, f"장애물 보상이 잘못됨: {reward}"
        print("✓ 장애물 테스트 성공")

        # 벽 테스트
        next_state, reward, done = env.step((0, 0), GridWorld.ACTION_UP)
        assert next_state == (0, 0), "벽에서 제자리 유지 실패"
        print("✓ 벽 처리 테스트 성공")

        print("\n✓ GridWorld 모든 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"\n✗ GridWorld 테스트 실패: {e}\n")
        return False


def test_value_iteration():
    """Value Iteration 테스트"""
    print("=" * 60)
    print("테스트 2: Value Iteration")
    print("=" * 60)

    try:
        env = GridWorld(grid_size=4)
        print("Value Iteration 실행 중...")

        V, policy = vi(env, gamma=0.9, theta=0.001, max_iterations=100)

        print("✓ Value Iteration 실행 성공")

        # 가치 함수 확인
        assert len(V) == 16, "가치 함수 크기가 잘못됨"
        assert env.goal in V, "목표 상태가 가치 함수에 없음"
        print("✓ 가치 함수 생성 성공")

        # 정책 확인
        non_terminal_states = [s for s in env.states if not env.is_terminal(s)]
        assert len(policy) == len(non_terminal_states), "정책 크기가 잘못됨"
        print("✓ 정책 생성 성공")

        # 정책 검증 (목표를 향해야 함)
        # 시작 위치에서는 오른쪽 또는 아래로 이동해야 함
        start_action = policy.get((0, 0))
        assert start_action in [GridWorld.ACTION_RIGHT, GridWorld.ACTION_DOWN], \
            f"시작 위치 정책이 이상함: {start_action}"
        print("✓ 정책 검증 성공")

        # 가치 함수가 목표에 가까울수록 커야 함
        v_start = V[(0, 0)]
        v_goal_neighbor = V[(3, 2)]  # 목표 옆
        assert v_goal_neighbor > v_start, "가치 함수가 올바르지 않음"
        print("✓ 가치 함수 검증 성공")

        print("\n✓ Value Iteration 모든 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"\n✗ Value Iteration 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_policy_iteration():
    """Policy Iteration 테스트"""
    print("=" * 60)
    print("테스트 3: Policy Iteration")
    print("=" * 60)

    try:
        env = GridWorld(grid_size=4)
        print("Policy Iteration 실행 중...")

        policy, V = pi(env, gamma=0.9, theta=0.001, max_iterations=100)

        print("✓ Policy Iteration 실행 성공")

        # 가치 함수 확인
        assert len(V) == 16, "가치 함수 크기가 잘못됨"
        print("✓ 가치 함수 생성 성공")

        # 정책 확인
        non_terminal_states = [s for s in env.states if not env.is_terminal(s)]
        assert len(policy) == len(non_terminal_states), "정책 크기가 잘못됨"
        print("✓ 정책 생성 성공")

        # 정책 검증
        start_action = policy.get((0, 0))
        assert start_action in [GridWorld.ACTION_RIGHT, GridWorld.ACTION_DOWN], \
            f"시작 위치 정책이 이상함: {start_action}"
        print("✓ 정책 검증 성공")

        print("\n✓ Policy Iteration 모든 테스트 통과!\n")
        return True

    except Exception as e:
        print(f"\n✗ Policy Iteration 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_algorithms_comparison():
    """두 알고리즘 결과 비교"""
    print("=" * 60)
    print("테스트 4: Value Iteration vs Policy Iteration 비교")
    print("=" * 60)

    try:
        env = GridWorld(grid_size=4)

        # Value Iteration
        print("\nValue Iteration 실행...")
        V_vi, policy_vi = vi(env, gamma=0.9, theta=0.001, max_iterations=100)

        # Policy Iteration
        print("\nPolicy Iteration 실행...")
        policy_pi, V_pi = pi(env, gamma=0.9, theta=0.001, max_iterations=100)

        # 정책 비교
        policies_match = True
        for state in env.states:
            if env.is_terminal(state):
                continue

            if policy_vi.get(state) != policy_pi.get(state):
                policies_match = False
                break

        if policies_match:
            print("\n✓ 두 알고리즘의 정책이 일치합니다!")
        else:
            print("\n⚠ 두 알고리즘의 정책이 일부 다릅니다 (동점 행동 선택 차이)")

        # 가치 함수 비교
        max_diff = 0
        for state in env.states:
            diff = abs(V_vi.get(state, 0) - V_pi.get(state, 0))
            max_diff = max(max_diff, diff)

        print(f"✓ 가치 함수 최대 차이: {max_diff:.6f}")

        if max_diff < 0.01:
            print("✓ 가치 함수가 거의 일치합니다!")
        else:
            print("⚠ 가치 함수에 약간의 차이가 있습니다")

        print("\n✓ 알고리즘 비교 테스트 완료!\n")
        return True

    except Exception as e:
        print(f"\n✗ 비교 테스트 실패: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_gamma_effect():
    """Gamma 값의 영향 테스트"""
    print("=" * 60)
    print("테스트 5: Gamma 값의 영향")
    print("=" * 60)

    try:
        env = GridWorld(grid_size=4)
        gamma_values = [0.5, 0.9]

        policies = {}
        for gamma in gamma_values:
            print(f"\nγ = {gamma} 테스트 중...")
            V, policy = vi(env, gamma=gamma, theta=0.001, max_iterations=100)
            policies[gamma] = policy
            print(f"✓ γ = {gamma} 완료")

        # Gamma가 다르면 일부 정책이 다를 수 있음
        print("\n✓ Gamma 테스트 완료!")
        print("   낮은 γ는 근시안적, 높은 γ는 장기적 정책을 생성합니다.")

        return True

    except Exception as e:
        print(f"\n✗ Gamma 테스트 실패: {e}\n")
        return False


def test_simulation():
    """최적 정책으로 시뮬레이션"""
    print("=" * 60)
    print("테스트 6: 최적 정책 시뮬레이션")
    print("=" * 60)

    try:
        env = GridWorld(grid_size=4)
        V, policy = vi(env, gamma=0.9, theta=0.001, max_iterations=100)

        print("\n최적 정책으로 에피소드 실행:")

        state = env.reset()
        steps = 0
        max_steps = 50
        total_reward = 0

        print(f"시작: {state}")

        while steps < max_steps:
            if env.is_terminal(state):
                print(f"\n✓ 목표 도달! (총 {steps}걸음)")
                break

            if state not in policy:
                print(f"\n✗ 정책에 없는 상태: {state}")
                return False

            action = policy[state]
            next_state, reward, done = env.step(state, action)
            total_reward += reward

            action_name = GridWorld.ACTION_NAMES[action]
            print(f"  {state} → {action_name} → {next_state} (보상: {reward:.2f})")

            state = next_state
            steps += 1

        if steps >= max_steps:
            print("\n✗ 최대 스텝 초과")
            return False

        print(f"총 보상: {total_reward:.4f}")
        print("\n✓ 시뮬레이션 성공!\n")
        return True

    except Exception as e:
        print(f"\n✗ 시뮬레이션 실패: {e}\n")
        return False


def main():
    """전체 테스트 실행"""
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  Week 5 강화학습 기초 - 전체 구현 테스트".center(58) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    results = []

    # 각 테스트 실행
    results.append(("GridWorld 환경", test_gridworld()))
    results.append(("Value Iteration", test_value_iteration()))
    results.append(("Policy Iteration", test_policy_iteration()))
    results.append(("알고리즘 비교", test_algorithms_comparison()))
    results.append(("Gamma 효과", test_gamma_effect()))
    results.append(("시뮬레이션", test_simulation()))

    # 결과 요약
    print("\n")
    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:30s} : {status}")

    # 전체 결과
    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("=" * 60)
    print(f"\n전체: {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n🎉 모든 테스트 통과! Week 5 구현이 완벽합니다!")
        print("\n다음 단계:")
        print("1. lecture.md로 이론 학습")
        print("2. script.md로 수업 진행")
        print("3. 실습 과제 수행")
        print("4. Week 6 Q-Learning 준비")
        return 0
    else:
        print("\n⚠ 일부 테스트 실패. 코드를 확인해주세요.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
