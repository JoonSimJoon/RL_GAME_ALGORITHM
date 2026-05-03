from __future__ import annotations

from rl_lab.algorithms.policy_based.actor_critic import ActorCriticAlgorithm
from rl_lab.algorithms.policy_based.reinforce_continuous import ReinforceContinuousAlgorithm
from rl_lab.algorithms.policy_based.reinforce_discrete import ReinforceDiscreteAlgorithm
from rl_lab.algorithms.tabular.q_learning import QLearningAlgorithm
from rl_lab.algorithms.tabular.sarsa import SARSAAlgorithm
from rl_lab.algorithms.value_based.dqn import DQNAlgorithm
from rl_lab.envs.specs import ENV_SPECS


def get_env_spec(env_key: str):
    if env_key not in ENV_SPECS:
        raise KeyError(f"Unknown environment: {env_key}")
    return ENV_SPECS[env_key]


def get_algorithm_class(env_spec, algo_key: str):
    if algo_key not in env_spec.compatible_algorithms:
        raise ValueError(f"Algorithm {algo_key} is not compatible with environment {env_spec.key}")

    if algo_key == "q_learning":
        return QLearningAlgorithm
    if algo_key == "sarsa":
        return SARSAAlgorithm
    if algo_key == "dqn":
        return DQNAlgorithm
    if algo_key == "reinforce":
        if env_spec.action_type == "discrete":
            return ReinforceDiscreteAlgorithm
        return ReinforceContinuousAlgorithm
    if algo_key in {"a2c", "actor_critic"}:
        return ActorCriticAlgorithm
    raise KeyError(f"Unknown algorithm: {algo_key}")
