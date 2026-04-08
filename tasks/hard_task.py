"""
tasks/hard_task.py
HARD TASK: Multi-turn conversation with correct escalation decision.

The agent must:
  1. Classify the ticket correctly.
  2. Respond appropriately across multiple turns.
  3. Decide WHEN (and whether) to escalate.

Grading is multi-dimensional:
  - Classification accuracy  (30 %)
  - Response quality          (40 %)
  - Escalation correctness    (30 %)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from env.customer_support_env import CustomerSupportEnv
from env.models import Action, ActionType, Observation
from env.scenarios import get_scenario

import math

def soft_clip(score: float) -> float:
    """Bulletproof clamp: always returns a value in (0.01, 0.99)."""
    try:
        val = float(score)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(val) or math.isinf(val):
        return 0.5
    return max(0.01, min(0.99, val))


@dataclass
class HardTaskResult:
    scenario_id: str
    steps: int
    cumulative_reward: float
    correctly_classified: bool
    escalated: bool
    should_have_escalated: bool
    escalation_correct: bool
    score: float  # strictly in (0, 1) after soft_clip

    def __post_init__(self):
        self.score = soft_clip(self.score)
        self.cumulative_reward = soft_clip(self.cumulative_reward)


HARD_SCENARIOS = [
    "hard_escalation_01",
    "hard_technical_escalation_01",
]


class HardTask:
    """
    Difficulty: HARD
    Objective : Full multi-turn support workflow with escalation decision.

    Score breakdown:
    - Classification score  : 0.3 × (1 if correct category else 0)
    - Response score        : 0.4 × clamp(response_reward / max_response_reward, 0, 1)
    - Escalation score      : 0.3 × (1 if escalation decision matches should_escalate else 0)

    Total score is always strictly in (0, 1) after soft_clip.
    """

    name = "hard_multi_turn_escalation"
    description = (
        "Conduct a full multi-turn conversation and decide correctly whether to escalate."
    )

    def __init__(self, scenario_id: str | None = None):
        self.env = CustomerSupportEnv(seed=2)
        self._scenario_id = scenario_id or HARD_SCENARIOS[0]

    def run(self, agent_fn, scenario_id: str | None = None) -> HardTaskResult:
        sid = scenario_id or self._scenario_id
        scenario = get_scenario(sid)
        if not scenario:
            raise ValueError(f"Scenario not found: {sid}")

        if hasattr(agent_fn, "reset"):
            agent_fn.reset()
        obs = self.env.reset(scenario_id=sid)
        should_escalate = scenario.should_escalate

        done = False
        steps = 0
        cumulative_reward = 0.0

        classified_correctly = False
        escalated = False
        total_response_reward = 0.0
        num_responses = 0

        while not done:
            action: Action = agent_fn(obs)
            obs, reward, done, info = self.env.step(action)
            steps += 1
            cumulative_reward += reward

            # Track classification
            if action.action_type == ActionType.CLASSIFY_TICKET:
                if action.category == scenario.true_category:
                    classified_correctly = True

            # Track escalation
            if action.action_type == ActionType.ESCALATE:
                escalated = True

            # Track response quality (sum response rewards)
            if action.action_type in (ActionType.RESPOND, ActionType.ASK_INFO):
                total_response_reward += reward
                num_responses += 1

        # ── Grade ────────────────────────────────────────────
        # 1. Classification component (0–0.3)
        classification_score = 0.3 if classified_correctly else 0.0

        # 2. Response quality component (0–0.4)
        max_response_reward = len(scenario.turns) * CustomerSupportEnv.REWARD_RELEVANT_RESPONSE
        if max_response_reward > 0:
            resp_ratio = max(0.0, min(1.0, total_response_reward / max_response_reward))
        else:
            resp_ratio = 0.0
        response_score = round(0.4 * resp_ratio, 4)

        # 3. Escalation decision component (0–0.3)
        escalation_correct = escalated == should_escalate
        escalation_score = 0.3 if escalation_correct else 0.0

        total_score = soft_clip(classification_score + response_score + escalation_score)

        return HardTaskResult(
            scenario_id=sid,
            steps=steps,
            cumulative_reward=soft_clip(cumulative_reward),
            correctly_classified=classified_correctly,
            escalated=escalated,
            should_have_escalated=should_escalate,
            escalation_correct=escalation_correct,
            score=total_score,
        )

    def run_all(self, agent_fn) -> float:
        scores: List[float] = []
        for sid in HARD_SCENARIOS:
            result = self.run(agent_fn, scenario_id=sid)
            scores.append(result.score)
        return soft_clip(sum(scores) / len(scores))
