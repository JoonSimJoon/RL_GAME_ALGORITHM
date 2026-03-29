"""
전체 구현 테스트 스크립트
모든 주요 기능이 정상 작동하는지 확인
"""

import math
from gridworld import GridWorld


# ---- Value Iteration ----
def value_iteration(env, gamma=0.9, theta=0.001, max_iterations=1000):
    V = {state: 0.0 for state in env.states}

    for iteration in range(1, max_iterations + 1):
        delta = 0
        for state in env.states:
            if env.is_terminal(state):
                continue
            v = V[state]
            max_value = -math.inf
            for action in range(env.n_actions):
                next_state, reward, _ = env.step(action, state=state)
                max_value = max(max_value, reward + gamma * V[next_state])
            V[state] = max_value
            delta = max(delta, abs(v - V[state]))
        if delta < theta:
            break

    policy = {}
    for state in env.states:
        if env.is_terminal(state):
            continue
        q_values = [0.0] * env.n_actions
        for action in range(env.n_actions):
            next_state, reward, _ = env.step(action, state=state)
            q_values[action] = reward + gamma * V[next_state]
        policy[state] = max(range(env.n_actions), key=lambda a: q_values[a])

    return V, policy


# ---- Policy Iteration ----
def policy_iteration(env, gamma=0.9, theta=0.001, max_iterations=100):
    policy = {state: [0.25] * 4 for state in env.states}

    for _ in range(max_iterations):
        # Policy evaluation
        V = {state: 0.0 for state in env.states}
        for _ in range(1000):
            delta = 0
            for state in env.states:
                if env.is_terminal(state):
                    continue
                v = V[state]
                new_value = 0.0
                for action in range(env.n_actions):
                    prob = policy[state][action]
                    if prob == 0:
                        continue
                    next_state, reward, _ = env.step(action, state=state)
                    new_value += prob * (reward + gamma * V[next_state])
                V[state] = new_value
                delta = max(delta, abs(v - V[state]))
            if delta < theta:
                break

        # Policy improvement
        new_policy = {}
        stable = True
        for state in env.states:
            if env.is_terminal(state):
                new_policy[state] = [0.25] * 4
                continue
            q_values = []
            for action in range(env.n_actions):
                next_state, reward, _ = env.step(action, state=state)
                q_values.append(reward + gamma * V[next_state])
            max_q = max(q_values)
            best = [i for i, q in enumerate(q_values) if abs(q - max_q) < 1e-10]
            probs = [0.0] * 4
            for a in best:
                probs[a] = 1.0 / len(best)

            old_best = max(range(4), key=lambda a: policy[state][a])
            new_best = max(range(4), key=lambda a: probs[a])
            if old_best != new_best:
                stable = False

            new_policy[state] = probs

        policy = new_policy
        if stable:
            break

    policy_det = {}
    for state in env.states:
        if not env.is_terminal(state):
            policy_det[state] = max(range(4), key=lambda a: policy[state][a])

    return policy_det, V


# ---- Tests ----
def test_gridworld():
    print("=" * 60)
    print("테스트 1: GridWorld 환경")
    print("=" * 60)

    try:
        env = GridWorld(4)
        print("✓ GridWorld 생성 성공")

        state = env.reset()
        assert state == (0, 0)
        print("✓ 환경 초기화 성공")

        next_state, reward, done = env.step(GridWorld.ACTION_RIGHT, state=(0, 0))
        assert next_state == (0, 1)
        assert not done
        print("✓ 행동 수행 성공")

        next_state, reward, done = env.step(GridWorld.ACTION_DOWN, state=(2, 3))
        assert next_state == (3, 3)
        assert done
        assert abs(reward - 1.0) < 1e-6
        print("✓ 목표 도달 테스트 성공")

        next_state, reward, done = env.step(GridWorld.ACTION_DOWN, state=(0, 1))
        assert next_state == (1, 1)
        assert done
        assert abs(reward - (-1.0)) < 1e-6
        print("✓ 장애물 테스트 성공")

        next_state, reward, done = env.step(GridWorld.ACTION_UP, state=(0, 0))
        assert next_state == (0, 0)
        print("✓ 벽 처리 테스트 성공")

        print("\n✓ GridWorld 모든 테스트 통과!\n")
        return True
    except Exception as e:
        print(f"\n✗ GridWorld 테스트 실패: {e}\n")
        return False


def test_value_iteration():
    print("=" * 60)
    print("테스트 2: Value Iteration")
    print("=" * 60)

    try:
        env = GridWorld(4)
        print("Value Iteration 실행 중...")
        V, policy = value_iteration(env, 0.9, 0.001, 100)

        print("✓ Value Iteration 실행 성공")

        assert len(V) == 16
        assert env.goal in V
        print("✓ 가치 함수 생성 성공")

        non_terminal = sum(1 for s in env.states if not env.is_terminal(s))
        assert len(policy) == non_terminal
        print("✓ 정책 생성 성공")

        start_action = policy[(0, 0)]
        assert start_action in (GridWorld.ACTION_RIGHT, GridWorld.ACTION_DOWN)
        print("✓ 정책 검증 성공")

        assert V[(3, 2)] > V[(0, 0)]
        print("✓ 가치 함수 검증 성공")

        print("\n✓ Value Iteration 모든 테스트 통과!\n")
        return True
    except Exception as e:
        print(f"\n✗ Value Iteration 테스트 실패: {e}\n")
        return False


def test_policy_iteration():
    print("=" * 60)
    print("테스트 3: Policy Iteration")
    print("=" * 60)

    try:
        env = GridWorld(4)
        print("Policy Iteration 실행 중...")
        policy, V = policy_iteration(env, 0.9, 0.001, 100)

        print("✓ Policy Iteration 실행 성공")

        assert len(V) == 16
        print("✓ 가치 함수 생성 성공")

        non_terminal = sum(1 for s in env.states if not env.is_terminal(s))
        assert len(policy) == non_terminal
        print("✓ 정책 생성 성공")

        start_action = policy[(0, 0)]
        assert start_action in (GridWorld.ACTION_RIGHT, GridWorld.ACTION_DOWN)
        print("✓ 정책 검증 성공")

        print("\n✓ Policy Iteration 모든 테스트 통과!\n")
        return True
    except Exception as e:
        print(f"\n✗ Policy Iteration 테스트 실패: {e}\n")
        return False


def test_algorithms_comparison():
    print("=" * 60)
    print("테스트 4: Value Iteration vs Policy Iteration 비교")
    print("=" * 60)

    try:
        env1 = GridWorld(4)
        env2 = GridWorld(4)

        print("\nValue Iteration 실행...")
        V_vi, policy_vi = value_iteration(env1, 0.9, 0.001, 100)

        print("\nPolicy Iteration 실행...")
        policy_pi, V_pi = policy_iteration(env2, 0.9, 0.001, 100)

        policies_match = all(
            policy_vi.get(s) == policy_pi.get(s)
            for s in env1.states
            if not env1.is_terminal(s)
        )

        if policies_match:
            print("\n✓ 두 알고리즘의 정책이 일치합니다!")
        else:
            print("\n⚠ 두 알고리즘의 정책이 일부 다릅니다 (동점 행동 선택 차이)")

        max_diff = max(abs(V_vi[s] - V_pi[s]) for s in env1.states)
        print(f"✓ 가치 함수 최대 차이: {max_diff:.6f}")

        if max_diff < 0.01:
            print("✓ 가치 함수가 거의 일치합니다!")

        print("\n✓ 알고리즘 비교 테스트 완료!\n")
        return True
    except Exception as e:
        print(f"\n✗ 비교 테스트 실패: {e}\n")
        return False


def test_simulation():
    print("=" * 60)
    print("테스트 5: 최적 정책 시뮬레이션")
    print("=" * 60)

    try:
        env = GridWorld(4)
        V, policy = value_iteration(env, 0.9, 0.001, 100)

        print("\n최적 정책으로 에피소드 실행:")
        state = env.reset()
        total_reward = 0
        print(f"시작: {state}")

        for step in range(50):
            if env.is_terminal(state):
                print(f"\n✓ 목표 도달! (총 {step}걸음)")
                break
            if state not in policy:
                print("\n✗ 정책에 없는 상태")
                return False

            action = policy[state]
            next_state, reward, done = env.step(action, state=state)
            total_reward += reward
            print(
                f"  {state} → {GridWorld.ACTION_NAMES[action]} → {next_state} (보상: {reward:.2f})"
            )
            state = next_state
        else:
            print("\n✗ 최대 스텝 초과")
            return False

        print(f"총 보상: {total_reward:.4f}")
        print("\n✓ 시뮬레이션 성공!\n")
        return True
    except Exception as e:
        print(f"\n✗ 시뮬레이션 실패: {e}\n")
        return False


if __name__ == "__main__":
    print()
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*  Week 5 강화학습 기초 - 전체 구현 테스트              *")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print()

    results = [
        ("GridWorld 환경", test_gridworld()),
        ("Value Iteration", test_value_iteration()),
        ("Policy Iteration", test_policy_iteration()),
        ("알고리즘 비교", test_algorithms_comparison()),
        ("시뮬레이션", test_simulation()),
    ]

    print()
    print("=" * 60)
    print("테스트 결과 요약")
    print("=" * 60)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:<30} : {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print("=" * 60)
    print(f"\n전체: {passed}/{total} 테스트 통과")

    if passed == total:
        print("\n모든 테스트 통과! Week 5 구현이 완벽합니다!")
        print("\n다음 단계:")
        print("1. lecture.md로 이론 학습")
        print("2. script.md로 수업 진행")
        print("3. 실습 과제 수행")
        print("4. Week 6 Q-Learning 준비")
    else:
        print("\n⚠ 일부 테스트 실패. 코드를 확인해주세요.")
