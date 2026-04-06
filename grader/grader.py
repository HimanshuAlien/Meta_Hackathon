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


@dataclass
class GradeReport:
    task_name: str
    score: float           # 0.0 – 1.0
    passed: bool           # score >= 0.5
    commentary: str

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"\n{'-'*50}\n"
            f"  Task      : {self.task_name}\n"
            f"  Score     : {self.score:.4f} / 1.0000\n"
            f"  Status    : {status}\n"
            f"  Comments  : {self.commentary}\n"
            f"{'-'*50}"
        )


class EasyGrader:
    """
    Deterministic grader for the EASY task (ticket classification).
    Logic: 1.0 if predicted == true_category, else 0.0.
    """

    PASS_THRESHOLD = 0.5

    def grade(self, result: EasyTaskResult) -> GradeReport:
        score = result.score  # already 0.0 or 1.0

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

        return GradeReport(
            task_name="easy_classify_ticket",
            score=score,
            passed=score >= self.PASS_THRESHOLD,
            commentary=commentary,
        )


class MediumGrader:
    """
    Deterministic grader for the MEDIUM task (response quality).
    Score = clamp(cumulative_reward / max_possible_reward, 0, 1).
    """

    PASS_THRESHOLD = 0.4

    def grade(self, result: MediumTaskResult) -> GradeReport:
        score = result.score

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

        return GradeReport(
            task_name="medium_generate_response",
            score=score,
            passed=score >= self.PASS_THRESHOLD,
            commentary=commentary,
        )


class HardGrader:
    """
    Deterministic grader for the HARD task (multi-turn + escalation).

    Breakdown:
      - classification_score  : 0.30 points (correct category)
      - response_score        : 0.40 points (response quality ratio)
      - escalation_score      : 0.30 points (correct escalation decision)
    """

    PASS_THRESHOLD = 0.5

    def grade(self, result: HardTaskResult) -> GradeReport:
        score = result.score

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

        return GradeReport(
            task_name="hard_multi_turn_escalation",
            score=score,
            passed=score >= self.PASS_THRESHOLD,
            commentary=commentary,
        )


def grade_all(
    easy_result: Union[EasyTaskResult, None] = None,
    medium_result: Union[MediumTaskResult, None] = None,
    hard_result: Union[HardTaskResult, None] = None,
) -> float:
    """Grade all tasks provided and return the mean score."""
    scores = []
    if easy_result is not None:
        scores.append(EasyGrader().grade(easy_result).score)
    if medium_result is not None:
        scores.append(MediumGrader().grade(medium_result).score)
    if hard_result is not None:
        scores.append(HardGrader().grade(hard_result).score)
    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)
