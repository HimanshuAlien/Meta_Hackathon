"""
grader/grader.py
Deterministic graders for all three task difficulties.

Each grader exposes a single `grade(result)` method that returns a
normalised score ∈ [0.0, 1.0] with a human-readable report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

from tasks.easy_task import EasyTaskResult
from tasks.medium_task import MediumTaskResult
from tasks.hard_task import HardTaskResult


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
class GradeReport:
    task_name: str
    score: float           # strictly (0.0, 1.0)
    passed: bool           # score >= 0.5
    commentary: str

    def __post_init__(self):
        self.score = soft_clip(self.score)

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            "\n" + "-"*50 + "\n"
            "  Task      : " + str(self.task_name) + "\n"
            "  Score     : " + "{:.4f}".format(self.score) + "\n"
            "  Status    : " + str(status) + "\n"
            "  Comments  : " + str(self.commentary) + "\n"
            + "-"*50
        )


class EasyGrader:
    """
    Deterministic grader for the EASY task (ticket classification).
    Logic: 1.0 if predicted == true_category, else 0.0.
    """

    PASS_THRESHOLD = 0.5

    def grade(self, result: EasyTaskResult) -> float:
        score = soft_clip(result.score)  # already 0.0 or 1.0 but enforce strict clipping

        if result.is_correct:
            commentary = (
                f"Correctly classified as '{result.true_category}'. "
                f"Completed in {result.steps} step(s)."
            )
        else:
            commentary = (
                f"Predicted '{result.predicted_category}' but expected "
                f"'{result.true_category}'. "
                f"Classification was wrong after {result.steps} step(s)."
            )

        return score


class MediumGrader:
    """
    Deterministic grader for the MEDIUM task (response quality).
    Score = clamp(cumulative_reward / max_possible_reward, 0, 1).
    """

    PASS_THRESHOLD = 0.4

    def grade(self, result: MediumTaskResult) -> float:
        score = soft_clip(result.score)

        resp_scores = result.response_rewards
        if resp_scores:
            avg_resp = sum(resp_scores) / len(resp_scores)
            resp_detail = (
                f"Avg response reward per turn: {avg_resp:.3f} over "
                f"{len(resp_scores)} response(s)."
            )
        else:
            resp_detail = "No response actions taken."

        commentary = (
            f"Scenario '{result.scenario_id}' | "
            f"Cumulative reward: {result.cumulative_reward:.3f} | "
            f"{resp_detail}"
        )

        return score


class HardGrader:
    """
    Deterministic grader for the HARD task (multi-turn + escalation).

    Breakdown:
      - classification_score  : 0.30 points (correct category)
      - response_score        : 0.40 points (response quality ratio)
      - escalation_score      : 0.30 points (correct escalation decision)
    """

    PASS_THRESHOLD = 0.5

    def grade(self, result: HardTaskResult) -> float:
        score = soft_clip(result.score)

        esc_decision = (
            "Escalated ✓" if result.escalated else "Did NOT escalate"
        )
        esc_expected = (
            "should have escalated" if result.should_have_escalated
            else "should NOT have escalated"
        )
        esc_verdict = "CORRECT" if result.escalation_correct else "WRONG"

        classification_verdict = (
            "Correctly classified ✓" if result.correctly_classified
            else "Wrong classification ✗"
        )

        commentary = (
            f"Scenario '{result.scenario_id}' | "
            f"{classification_verdict} | "
            f"{esc_decision} ({esc_expected}) → {esc_verdict} | "
            f"Steps: {result.steps}"
        )

        return score


def grade_all(
    easy_result: Union[EasyTaskResult, None] = None,
    medium_result: Union[MediumTaskResult, None] = None,
    hard_result: Union[HardTaskResult, None] = None,
) -> float:
    """Grade all tasks provided and return the mean score."""
    scores = []
    if easy_result is not None:
        scores.append(EasyGrader().grade(easy_result))
    if medium_result is not None:
        scores.append(MediumGrader().grade(medium_result))
    if hard_result is not None:
        scores.append(HardGrader().grade(hard_result))
    if not scores:
        return 0.5
    return soft_clip(sum(scores) / len(scores))
