"""
env/scenarios.py
A library of deterministic customer-support conversation scenarios.
Each scenario drives the environment's realistic simulation logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import TicketCategory, UserEmotion


@dataclass
class ScenarioTurn:
    """A single expected turn inside a scenario."""
    user_message: str
    # Keywords that represent a "good" agent response for this turn
    good_keywords: List[str] = field(default_factory=list)
    # Keywords that indicate an angry / frustrated reaction if NOT addressed
    frustration_triggers: List[str] = field(default_factory=list)


@dataclass
class Scenario:
    """
    Full conversation scenario used by the environment.
    Encodes the ground-truth category, correct escalation flag,
    and a sequence of realistic customer utterances.
    """

    scenario_id: str
    true_category: TicketCategory
    difficulty: str                       # easy | medium | hard
    should_escalate: bool
    description: str
    turns: List[ScenarioTurn] = field(default_factory=list)

    # Starting emotion of the customer
    initial_emotion: UserEmotion = UserEmotion.NEUTRAL

    # Maximum turns before the episode auto-terminates
    max_turns: int = 6

    # New fields for VIP and SLA
    is_vip: bool = False
    sla_deadline: int = 99


# ──────────────────────────────────────────────
# Scenario Library
# ──────────────────────────────────────────────

SCENARIOS: List[Scenario] = [

    # ── Easy ──────────────────────────────────
    Scenario(
        scenario_id="easy_billing_01",
        true_category=TicketCategory.BILLING,
        difficulty="easy",
        should_escalate=False,
        description="Customer overcharged on monthly subscription.",
        initial_emotion=UserEmotion.NEUTRAL,
        max_turns=4,
        turns=[
            ScenarioTurn(
                user_message="Hi, I was charged twice on my subscription this month. "
                             "My account number is 78234.",
                good_keywords=["billing", "subscription", "charge", "refund", "sorry"],
                frustration_triggers=["unrelated", "ignore", "hardware"],
            ),
            ScenarioTurn(
                user_message="It says $49.99 was taken twice on March 10th.",
                good_keywords=["refund", "investigate", "processing", "team"],
                frustration_triggers=["cannot", "impossible"],
            ),
            ScenarioTurn(
                user_message="Okay, how long will the refund take?",
                good_keywords=["3", "5", "business", "days", "email", "confirmation"],
                frustration_triggers=["not sure", "don't know"],
            ),
        ],
    ),

    Scenario(
        scenario_id="easy_technical_01",
        true_category=TicketCategory.TECHNICAL,
        difficulty="easy",
        should_escalate=False,
        description="Customer cannot log in to their account.",
        initial_emotion=UserEmotion.NEUTRAL,
        max_turns=4,
        turns=[
            ScenarioTurn(
                user_message="I can't log in to my account. It keeps saying my password is wrong.",
                good_keywords=["reset", "password", "link", "email", "help"],
                frustration_triggers=["billing", "payment"],
            ),
            ScenarioTurn(
                user_message="I tried resetting but the email never arrived.",
                good_keywords=["spam", "junk", "resend", "alternative", "check"],
                frustration_triggers=["just wait", "nothing we can do"],
            ),
        ],
    ),

    Scenario(
        scenario_id="easy_general_01",
        true_category=TicketCategory.GENERAL,
        difficulty="easy",
        should_escalate=False,
        description="Customer asking about office hours and return policy.",
        initial_emotion=UserEmotion.CALM,
        max_turns=3,
        turns=[
            ScenarioTurn(
                user_message="What are your customer support hours? And what is your return policy?",
                good_keywords=["hours", "available", "return", "policy", "days"],
                frustration_triggers=[],
            ),
        ],
    ),

    # ── Medium ────────────────────────────────
    Scenario(
        scenario_id="medium_billing_02",
        true_category=TicketCategory.BILLING,
        difficulty="medium",
        should_escalate=False,
        description="Customer disputing an unknown charge on a cancelled account.",
        initial_emotion=UserEmotion.NEUTRAL,
        max_turns=6,
        turns=[
            ScenarioTurn(
                user_message="I cancelled my account last month but you still charged me $19.99.",
                good_keywords=["apolog", "cancel", "refund", "account", "charge"],
                frustration_triggers=["policy", "non-refundable"],
            ),
            ScenarioTurn(
                user_message="I have the cancellation confirmation email from March 1st.",
                good_keywords=["confirm", "escalat", "refund", "process", "sorry"],
                frustration_triggers=["we cannot", "invalid"],
            ),
            ScenarioTurn(
                user_message="This is unacceptable. I want my money back immediately.",
                good_keywords=["understand", "priority", "refund", "24", "48", "hours"],
                frustration_triggers=["wait", "policy"],
            ),
        ],
    ),

    Scenario(
        scenario_id="medium_technical_02",
        true_category=TicketCategory.TECHNICAL,
        difficulty="medium",
        should_escalate=False,
        description="Customer experiencing frequent app crashes after update.",
        initial_emotion=UserEmotion.NEUTRAL,
        max_turns=6,
        turns=[
            ScenarioTurn(
                user_message="Your app keeps crashing ever since the latest update. "
                             "I'm on iPhone 14 iOS 17.",
                good_keywords=["crash", "update", "reinstall", "clear", "cache", "sorry"],
                frustration_triggers=["billing", "unrelated"],
            ),
            ScenarioTurn(
                user_message="I already reinstalled three times. It still crashes on startup.",
                good_keywords=["engineer", "report", "log", "team", "investigate"],
                frustration_triggers=["just wait", "nothing"],
            ),
            ScenarioTurn(
                user_message="How long will this take to fix?",
                good_keywords=["48", "72", "hours", "patch", "update", "notify"],
                frustration_triggers=["don't know", "not sure"],
            ),
        ],
    ),

    # ── Hard ──────────────────────────────────
    Scenario(
        scenario_id="hard_escalation_01",
        true_category=TicketCategory.BILLING,
        difficulty="hard",
        should_escalate=True,
        description="Angry VIP customer with billing fraud; requires escalation to manager.",
        initial_emotion=UserEmotion.ANGRY,
        max_turns=8,
        is_vip=True,
        sla_deadline=4,
        turns=[
            ScenarioTurn(
                user_message="I have been charged $500 without authorization! "
                             "This is FRAUD. I need a manager NOW.",
                good_keywords=["manager", "escalat", "apolog", "understand", "priority"],
                frustration_triggers=["policy", "can't", "sorry nothing"],
            ),
            ScenarioTurn(
                user_message="I have been a customer for 5 years. This is absolutely ridiculous.",
                good_keywords=["value", "loyal", "important", "escalat", "senior"],
                frustration_triggers=["rules", "policy", "can't help"],
            ),
            ScenarioTurn(
                user_message="What is the manager's name and when can I speak to them?",
                good_keywords=["connect", "within", "hours", "call", "schedule"],
                frustration_triggers=["don't know", "not available"],
            ),
            ScenarioTurn(
                user_message="Fine. But I also want a full refund for this month.",
                good_keywords=["noted", "refund", "process", "priority", "confirm"],
                frustration_triggers=["cannot guarantee", "no"],
            ),
        ],
    ),

    Scenario(
        scenario_id="hard_technical_escalation_01",
        true_category=TicketCategory.TECHNICAL,
        difficulty="hard",
        should_escalate=True,
        description="Enterprise customer with data loss incident needing immediate escalation.",
        initial_emotion=UserEmotion.ANGRY,
        max_turns=8,
        is_vip=True,
        sla_deadline=3,
        turns=[
            ScenarioTurn(
                user_message="We lost 3 months of data after your server migration yesterday. "
                             "We are a paying enterprise client.",
                good_keywords=["escalat", "critical", "urgent", "data", "engineer"],
                frustration_triggers=["policy", "please wait"],
            ),
            ScenarioTurn(
                user_message="We need this resolved in 2 hours or we are cancelling our $50K contract.",
                good_keywords=["senior", "engineer", "immediately", "escalat", "priority"],
                frustration_triggers=["cannot", "impossible", "wait"],
            ),
            ScenarioTurn(
                user_message="Who is your escalation contact? Give me a direct line.",
                good_keywords=["manager", "contact", "direct", "call", "now"],
                frustration_triggers=["not available", "don't have"],
            ),
        ],
    ),
]


def get_scenario(scenario_id: str) -> Optional[Scenario]:
    for s in SCENARIOS:
        if s.scenario_id == scenario_id:
            return s
    return None


def get_scenarios_by_difficulty(difficulty: str) -> List[Scenario]:
    return [s for s in SCENARIOS if s.difficulty == difficulty]
