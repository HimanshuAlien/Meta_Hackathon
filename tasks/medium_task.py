"""
tasks/medium_task.py
MEDIUM TASK: Generate a correct, helpful response to the customer.

The agent must respond with text that contains appropriate keywords
matching the scenario's expected good-response vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from env.customer_support_env import CustomerSupportEnv
from env.models import Action, ActionType, Observation
from env.scenarios import get_scenario


@dataclass
class MediumTaskResult:
    scenario_id: str
    steps: int
    cumulative_reward: float
    response_rewards: List[float] = field(default_factory=list)
    score: float = 0.0


MEDIUM_SCENARIOS = [
    "medium_billing_02",
    "medium_technical_02",
]


class MediumTask:
    """
    Difficulty: MEDIUM
    Objective : For each customer turn, produce a response text that
                contains relevant keywords, earning positive rewards.
    Scoring   : score = clamp(cumulative_reward / max_possible, 0, 1)

    The maximum possible reward per scenario is computed from the number
    of turns × REWARD_RELEVANT_RESPONSE (0.4) + REWARD_CORRECT_CLASSIFICATION (0.3).
    """

    name = "medium_generate_response"
    description = "Produce helpful, relevant responses across a multi-turn conversation."

    def __init__(self, scenario_id: str | None = None):
        self.env = CustomerSupportEnv(seed=1)
        self._scenario_id = scenario_id or MEDIUM_SCENARIOS[0]

    def _max_possible_reward(self, scenario_id: str) -> float:
        """Compute a rough upper bound for normalisation."""
        scenario = get_scenario(scenario_id)
        if not scenario:
            return 1.0
        # 1 classification + one respond per turn
        turns = len(scenario.turns)
        return round(
            CustomerSupportEnv.REWARD_CORRECT_CLASSIFICATION
            + turns * CustomerSupportEnv.REWARD_RELEVANT_RESPONSE,
            4,
        )

    def run(self, agent_fn, scenario_id: str | None = None) -> MediumTaskResult:
        sid = scenario_id or self._scenario_id
        if hasattr(agent_fn, "reset"):
            agent_fn.reset()
        obs = self.env.reset(scenario_id=sid)

        done = False
        total_reward = 0.0
        steps = 0
        response_rewards: List[float] = []

        while not done:
            action: Action = agent_fn(obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

            if action.action_type in (ActionType.RESPOND, ActionType.ASK_INFO):
                response_rewards.append(reward)

        max_r = self._max_possible_reward(sid)
        raw_score = total_reward / max_r if max_r > 0 else 0.0
        score = round(max(0.0, min(1.0, raw_score)), 4)

        return MediumTaskResult(
            scenario_id=sid,
            steps=steps,
            cumulative_reward=round(total_reward, 4),
            response_rewards=response_rewards,
            score=score,
        )

    def run_all(self, agent_fn) -> float:
        scores = []
        for sid in MEDIUM_SCENARIOS:
            result = self.run(agent_fn, scenario_id=sid)
            scores.append(result.score)
        return round(sum(scores) / len(scores), 4)
