"""
tasks/easy_task.py
EASY TASK: Classify the customer's issue into the correct ticket category.

The agent must issue exactly one `classify_ticket` action with the correct category.
"""

from __future__ import annotations

from dataclasses import dataclass

from env.customer_support_env import CustomerSupportEnv
from env.models import Action, ActionType, Observation, TicketCategory


@dataclass
class EasyTaskResult:
    predicted_category: str
    true_category: str
    is_correct: bool
    score: float
    steps: int
    reward: float


EASY_SCENARIOS = [
    ("easy_billing_01",   TicketCategory.BILLING),
    ("easy_technical_01", TicketCategory.TECHNICAL),
    ("easy_general_01",   TicketCategory.GENERAL),
]


class EasyTask:
    """
    Difficulty: EASY
    Objective : Given the opening customer message, classify the ticket
                into [billing | technical | general].
    Success   : Graded 1.0 if classification is correct, 0.0 otherwise.
    """

    name = "easy_classify_ticket"
    description = "Classify the customer issue into the correct category."

    def __init__(self, scenario_id: str | None = None):
        self.env = CustomerSupportEnv(seed=0)
        # Default to the first easy scenario; override via run()
        self._scenario_id = scenario_id or EASY_SCENARIOS[0][0]

    def run(self, agent_fn, scenario_id: str | None = None) -> EasyTaskResult:
        """
        Run the task for one scenario.

        Args:
            agent_fn : Callable[[Observation], Action]
            scenario_id: Optional override.

        Returns:
            EasyTaskResult with score [0.0, 1.0].
        """
        sid = scenario_id or self._scenario_id
        # Reset stateful agents between episodes
        if hasattr(agent_fn, "reset"):
            agent_fn.reset()
        obs = self.env.reset(scenario_id=sid)
        true_cat = obs.metadata.get("scenario_id", "")

        # Map scenario_id back to true category
        true_category = next(
            (cat for s_id, cat in EASY_SCENARIOS if s_id == sid),
            TicketCategory.GENERAL,
        )

        done = False
        total_reward = 0.0
        steps = 0
        predicted_category = TicketCategory.GENERAL  # default

        while not done:
            action: Action = agent_fn(obs)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1

            if action.action_type == ActionType.CLASSIFY_TICKET and action.category:
                predicted_category = action.category
                break  # task complete after first classification

        is_correct = predicted_category == true_category
        score = 1.0 if is_correct else 0.0

        return EasyTaskResult(
            predicted_category=str(predicted_category),
            true_category=str(true_category),
            is_correct=is_correct,
            score=score,
            steps=steps,
            reward=round(total_reward, 4),
        )

    def run_all(self, agent_fn) -> float:
        """Run across all easy scenarios; return mean score."""
        scores = []
        for sid, _ in EASY_SCENARIOS:
            result = self.run(agent_fn, scenario_id=sid)
            scores.append(result.score)
        return round(sum(scores) / len(scores), 4)
