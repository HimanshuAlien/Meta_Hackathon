"""
env/models.py
Pydantic models for the AI Customer Support Escalation environment.
Defines Observation, Action, and Reward data structures (OpenEnv-compliant).
"""

from __future__ import annotations

from enum import Enum
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Enum helpers (keeps literals consistent)
# ─────────────────────────────────────────────

class UserEmotion(str, Enum):
    CALM = "calm"
    NEUTRAL = "neutral"
    ANGRY = "angry"


class TicketStatus(str, Enum):
    OPEN = "open"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class TicketCategory(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    GENERAL = "general"
    UNCLASSIFIED = "unclassified"


class ActionType(str, Enum):
    CLASSIFY_TICKET = "classify_ticket"
    RESPOND = "respond"
    ASK_INFO = "ask_info"
    ESCALATE = "escalate"


# ─────────────────────────────────────────────
# Observation
# ─────────────────────────────────────────────

class Observation(BaseModel):
    """Full state visible to the agent at each step."""

    user_message: str = Field(
        description="The most recent message sent by the customer."
    )
    conversation_history: List[str] = Field(
        default_factory=list,
        description="Full turn-by-turn conversation log (alternating user/agent).",
    )
    user_emotion: UserEmotion = Field(
        default=UserEmotion.NEUTRAL,
        description="Inferred emotional state of the customer.",
    )
    ticket_status: TicketStatus = Field(
        default=TicketStatus.OPEN,
        description="Current status of the support ticket.",
    )
    ticket_category: TicketCategory = Field(
        default=TicketCategory.UNCLASSIFIED,
        description="Category assigned to the ticket (if classified).",
    )
    is_vip: bool = Field(
        default=False,
        description="Whether the customer is a VIP or Enterprise client.",
    )
    turns_remaining: int = Field(
        default=99,
        description="Remaining turns before SLA deadline is breached.",
    )
    turn: int = Field(default=0, description="Current turn number within the episode.")
    metadata: Dict = Field(
        default_factory=dict,
        description="Auxiliary info (scenario id, difficulty, etc.).",
    )

    class Config:
        use_enum_values = True


# ─────────────────────────────────────────────
# Action
# ─────────────────────────────────────────────

class Action(BaseModel):
    """Action chosen by the agent for a single step."""

    action_type: ActionType = Field(
        description="Which action the agent is taking."
    )

    # Used when action_type == CLASSIFY_TICKET
    category: Optional[TicketCategory] = Field(
        default=None,
        description="Category to assign (required for classify_ticket).",
    )

    # Used when action_type == RESPOND or ASK_INFO
    text: Optional[str] = Field(
        default=None,
        description="Text of the response or information request.",
    )

    class Config:
        use_enum_values = True

    def to_log(self) -> str:
        """Human-readable one-liner for logging."""
        if self.action_type == ActionType.CLASSIFY_TICKET:
            return f"[classify_ticket] category={self.category}"
        if self.action_type == ActionType.RESPOND:
            return f"[respond] '{self.text}'"
        if self.action_type == ActionType.ASK_INFO:
            return f"[ask_info] '{self.text}'"
        if self.action_type == ActionType.ESCALATE:
            return "[escalate]"
        return f"[{self.action_type}]"


# ─────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """Fine-grained reward signals for a single step."""

    correct_classification: float = Field(default=0.0)   # +0.3
    relevant_response: float = Field(default=0.0)         # +0.4
    successful_resolution: float = Field(default=0.0)     # +0.2
    correct_escalation: float = Field(default=0.0)        # +0.4
    wrong_escalation: float = Field(default=0.0)          # -0.3
    missed_escalation: float = Field(default=0.0)         # -0.4
    delayed_escalation: float = Field(default=0.0)        # -0.2
    early_escalation: float = Field(default=0.0)          # -0.2
    empathy_bonus: float = Field(default=0.0)             # +0.1
    repeated_response: float = Field(default=0.0)         # -0.1
    increased_frustration: float = Field(default=0.0)     # -0.2
    unnecessary_action: float = Field(default=0.0)        # -0.1

    @property
    def total(self) -> float:
        return round(
            self.correct_classification
            + self.relevant_response
            + self.successful_resolution
            + self.correct_escalation
            + self.wrong_escalation
            + self.missed_escalation
            + self.delayed_escalation
            + self.early_escalation
            + self.empathy_bonus
            + self.repeated_response
            + self.increased_frustration
            + self.unnecessary_action,
            4,
        )

    def to_log(self) -> str:
        parts = []
        if self.correct_classification: parts.append(f"class_ok={self.correct_classification:+.2f}")
        if self.relevant_response: parts.append(f"rel_resp={self.relevant_response:+.2f}")
        if self.successful_resolution: parts.append(f"resolved={self.successful_resolution:+.2f}")
        if self.correct_escalation: parts.append(f"esc_ok={self.correct_escalation:+.2f}")
        if self.wrong_escalation: parts.append(f"esc_wrong={self.wrong_escalation:+.2f}")
        if self.missed_escalation: parts.append(f"esc_miss={self.missed_escalation:+.2f}")
        if self.delayed_escalation: parts.append(f"esc_delay={self.delayed_escalation:+.2f}")
        if self.early_escalation: parts.append(f"esc_early={self.early_escalation:+.2f}")
        if self.empathy_bonus: parts.append(f"empathy={self.empathy_bonus:+.2f}")
        if self.repeated_response: parts.append(f"repeat={self.repeated_response:+.2f}")
        if self.increased_frustration: parts.append(f"frust={self.increased_frustration:+.2f}")
        if self.unnecessary_action: parts.append(f"unnec={self.unnecessary_action:+.2f}")
        total_str = f"TOT={self.total:+.2f}"
        return " | ".join(parts + [total_str]) if parts else total_str
