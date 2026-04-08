"""
env/customer_support_env.py
OpenEnv-compatible environment for AI Customer Support Escalation.

Public API:
    env.reset(scenario_id=None)        -> Observation
    env.step(action: Action)           -> (Observation, float, bool, dict)
    env.state()                        -> Observation
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .models import (
    Action,
    ActionType,
    Observation,
    RewardBreakdown,
    TicketCategory,
    TicketStatus,
    UserEmotion,
)
from .scenarios import SCENARIOS, Scenario, ScenarioTurn, get_scenario


class CustomerSupportEnv:
    """
    OpenEnv-compatible simulation of a customer-support escalation desk.

    The environment maintains:
    - A conversation history
    - The customer's current emotional state
    - The ticket's category and status
    - Dense step-level rewards aligned to the agent's actions
    """

    # Dense reward constants
    REWARD_CORRECT_CLASSIFICATION = +0.3
    REWARD_RELEVANT_RESPONSE = +0.4
    REWARD_SUCCESSFUL_RESOLUTION = +0.2
    REWARD_CORRECT_ESCALATION = +0.4
    REWARD_MISSED_ESCALATION = -0.4
    REWARD_DELAYED_ESCALATION = -0.2
    REWARD_EARLY_ESCALATION = -0.2
    REWARD_EMPATHY_BONUS = +0.1
    REWARD_REPEATED_RESPONSE = -0.05
    REWARD_WRONG_ESCALATION = -0.3
    REWARD_INCREASED_FRUSTRATION = -0.2
    REWARD_UNNECESSARY_ACTION = -0.1

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._scenario: Optional[Scenario] = None
        self._current_turn_idx: int = 0
        self._obs: Optional[Observation] = None
        self._done: bool = False
        self._cumulative_reward: float = 0.0
        self._step_count: int = 0
        self._classified: bool = False
        self._escalated: bool = False
        self._past_responses: set[str] = set()

    # ─────────────────────────────────────────
    # OpenEnv Core API
    # ─────────────────────────────────────────

    def reset(self, scenario_id: Optional[str] = None) -> Observation:
        """
        Start a new episode.  If scenario_id is None, a random scenario
        is chosen from the library.

        Returns:
            Initial Observation.
        """
        if scenario_id:
            scenario = get_scenario(scenario_id)
            if scenario is None:
                raise ValueError("Unknown scenario_id: " + str(scenario_id))
        else:
            scenario = self._rng.choice(SCENARIOS)

        self._scenario = scenario
        self._current_turn_idx = 0
        self._done = False
        self._cumulative_reward = 0.0
        self._step_count = 0
        self._classified = False
        self._escalated = False
        self._past_responses = set()

        first_turn = scenario.turns[0]

        self._obs = Observation(
            user_message=first_turn.user_message,
            conversation_history=["[Customer] " + str(first_turn.user_message)],
            user_emotion=scenario.initial_emotion,
            ticket_status=TicketStatus.OPEN,
            ticket_category=TicketCategory.UNCLASSIFIED,
            is_vip=scenario.is_vip,
            turns_remaining=scenario.sla_deadline,
            turn=0,
            metadata={
                "scenario_id": scenario.scenario_id,
                "difficulty": scenario.difficulty,
                "should_escalate": scenario.should_escalate,
                "description": scenario.description,
            },
        )
        return self._obs

    def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """
        Apply an agent action and advance the environment by one step.

        Args:
            action: An Action pydantic model.

        Returns:
            observation: Updated Observation.
            reward:      Scalar reward for this step.
            done:        True when the episode is terminal.
            info:        Auxiliary info dict (reward breakdown, etc.).
        """
        if self._done:
            raise RuntimeError("Episode is finished. Call reset() to start a new one.")
        if self._scenario is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        self._step_count += 1
        reward_breakdown = RewardBreakdown()

        obs = self._obs  # current state before action
        scenario = self._scenario
        current_scenario_turn: ScenarioTurn = scenario.turns[self._current_turn_idx]

        # ── Process action ──────────────────────────────
        if action.action_type == ActionType.CLASSIFY_TICKET:
            reward_breakdown = self._handle_classify(action, reward_breakdown)

        elif action.action_type == ActionType.RESPOND:
            reward_breakdown = self._handle_respond(
                action, current_scenario_turn, reward_breakdown, obs.user_emotion
            )

        elif action.action_type == ActionType.ASK_INFO:
            reward_breakdown = self._handle_ask_info(
                action, current_scenario_turn, reward_breakdown, obs.user_emotion
            )

        elif action.action_type == ActionType.ESCALATE:
            reward_breakdown = self._handle_escalate(reward_breakdown)

        # ── Append agent action to history ─────────────
        agent_line = "[Agent] " + str(action.to_log())
        conversation_history: List[str] = list(obs.conversation_history) + [agent_line]

        # ── Advance to next customer turn ───────────────
        next_turn_idx = self._current_turn_idx + 1
        new_emotion = self._update_emotion(obs.user_emotion, reward_breakdown)
        new_status = self._update_status(obs.ticket_status, action)

        # Frustration penalty if emotion worsened
        if (
            obs.user_emotion == UserEmotion.NEUTRAL
            and new_emotion == UserEmotion.ANGRY
        ):
            reward_breakdown.increased_frustration += self.REWARD_INCREASED_FRUSTRATION
        elif (
            obs.user_emotion == UserEmotion.CALM
            and new_emotion in (UserEmotion.NEUTRAL, UserEmotion.ANGRY)
        ):
            reward_breakdown.increased_frustration += self.REWARD_INCREASED_FRUSTRATION

        # Determine episode termination
        done = self._check_done(next_turn_idx, new_status, action)

        # Resolution bonus (and penalize missed escalation if applicable)
        if done:
            if new_status == TicketStatus.RESOLVED:
                reward_breakdown.successful_resolution += self.REWARD_SUCCESSFUL_RESOLUTION
            if new_status != TicketStatus.ESCALATED and scenario.should_escalate:
                reward_breakdown.missed_escalation += self.REWARD_MISSED_ESCALATION

        # Build next user message (if not done)
        if not done and next_turn_idx < len(scenario.turns):
            next_user_msg = scenario.turns[next_turn_idx].user_message
            conversation_history.append("[Customer] " + str(next_user_msg))
        else:
            next_user_msg = obs.user_message  # keep last message when done

        raw_reward = reward_breakdown.total
        # Bulletproof clamp: every step reward is strictly in (0.01, 0.99)
        scalar_reward = max(0.01, min(0.99, float(raw_reward)))

        self._cumulative_reward = round(self._cumulative_reward + scalar_reward, 4)
        self._current_turn_idx = next_turn_idx
        self._done = done

        self._obs = Observation(
            user_message=next_user_msg,
            conversation_history=conversation_history,
            user_emotion=new_emotion,
            ticket_status=new_status,
            ticket_category=(
                action.category
                if action.action_type == ActionType.CLASSIFY_TICKET and action.category
                else obs.ticket_category
            ),
            is_vip=scenario.is_vip,
            turns_remaining=max(0, scenario.sla_deadline - self._step_count),
            turn=self._step_count,
            metadata=obs.metadata,
        )

        info = {
            "reward_breakdown": reward_breakdown.model_dump(),
            "reward_breakdown_log": reward_breakdown.to_log(),
            "cumulative_reward": self._cumulative_reward,
            "step": self._step_count,
            "scenario_id": scenario.scenario_id,
        }

        return self._obs, scalar_reward, done, info

    def state(self) -> Observation:
        """Return the current observation without advancing the environment."""
        if self._obs is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")
        return self._obs

    # ─────────────────────────────────────────
    # Internal action handlers
    # ─────────────────────────────────────────

    def _handle_classify(
        self, action: Action, rb: RewardBreakdown
    ) -> RewardBreakdown:
        if self._classified:
            # Already classified — redundant action
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION
            return rb

        if action.category == self._scenario.true_category:
            rb.correct_classification += self.REWARD_CORRECT_CLASSIFICATION
        else:
            # Wrong classification is an unnecessary/harmful action
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION

        self._classified = True
        return rb

    def _handle_respond(
        self,
        action: Action,
        turn: ScenarioTurn,
        rb: RewardBreakdown,
        user_emotion: UserEmotion,
    ) -> RewardBreakdown:
        if action.text is None:
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION
            return rb

        text_lower = action.text.lower()
        
        # Check repeated response
        if text_lower in self._past_responses:
            rb.repeated_response += self.REWARD_REPEATED_RESPONSE
        else:
            self._past_responses.add(text_lower)

        # Empathy bonus
        empathy_keywords = ["apolog", "sorry", "understand", "frustrat"]
        if user_emotion == UserEmotion.ANGRY and any(kw in text_lower for kw in empathy_keywords):
            rb.empathy_bonus += self.REWARD_EMPATHY_BONUS
            if "sorry" in text_lower:
                rb.empathy_bonus += 0.05

        # Check if any good keyword appears in the response
        matched = any(kw.lower() in text_lower for kw in turn.good_keywords)
        # Check if any frustration trigger appears
        triggered = any(kw.lower() in text_lower for kw in turn.frustration_triggers)

        if matched and not triggered:
            rb.relevant_response += self.REWARD_RELEVANT_RESPONSE
        elif triggered:
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION
        
        # Long response bonus
        if len(action.text) > 50:
            rb.relevant_response += 0.05
        # Neutral response (neither matched nor triggered) → no extra reward

        return rb

    def _handle_ask_info(
        self,
        action: Action,
        turn: ScenarioTurn,
        rb: RewardBreakdown,
        user_emotion: UserEmotion,
    ) -> RewardBreakdown:
        if action.text is None:
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION
            return rb

        text_lower = action.text.lower()
        
        if text_lower in self._past_responses:
            rb.repeated_response += self.REWARD_REPEATED_RESPONSE
        else:
            self._past_responses.add(text_lower)
            
        empathy_keywords = ["apolog", "sorry", "understand", "frustrat"]
        if user_emotion == UserEmotion.ANGRY and any(kw in text_lower for kw in empathy_keywords):
            rb.empathy_bonus += self.REWARD_EMPATHY_BONUS
            if "sorry" in text_lower:
                rb.empathy_bonus += 0.05

        matched = any(kw.lower() in text_lower for kw in turn.good_keywords)

        if matched:
            # Useful clarifying question counts as half a relevant response
            rb.relevant_response += self.REWARD_RELEVANT_RESPONSE * 0.5
        # If asking for irrelevant info, small penalty
        elif any(kw.lower() in text_lower for kw in turn.frustration_triggers):
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION

        # Long response bonus
        if len(action.text) > 50:
            rb.relevant_response += 0.05

        return rb

    def _handle_escalate(self, rb: RewardBreakdown) -> RewardBreakdown:
        if self._escalated:
            rb.unnecessary_action += self.REWARD_UNNECESSARY_ACTION
            return rb

        self._escalated = True
        
        if not self._past_responses:
            rb.early_escalation += self.REWARD_EARLY_ESCALATION

        if self._scenario.should_escalate:
            rb.correct_escalation += self.REWARD_CORRECT_ESCALATION
            if self._step_count > self._scenario.sla_deadline:
                rb.delayed_escalation += self.REWARD_DELAYED_ESCALATION
        else:
            rb.wrong_escalation += self.REWARD_WRONG_ESCALATION

        return rb

    # ─────────────────────────────────────────
    # State-transition helpers
    # ─────────────────────────────────────────

    def _update_emotion(
        self, current: UserEmotion, rb: RewardBreakdown
    ) -> UserEmotion:
        if rb.relevant_response > 0:
            # Good response → calm customer down one notch
            if current == UserEmotion.ANGRY:
                return UserEmotion.NEUTRAL
            return UserEmotion.CALM
        if rb.unnecessary_action < 0 or rb.wrong_escalation < 0:
            # Bad action → customer becomes more frustrated
            if current == UserEmotion.CALM:
                return UserEmotion.NEUTRAL
            return UserEmotion.ANGRY
        return current

    def _update_status(
        self, current: TicketStatus, action: Action
    ) -> TicketStatus:
        if action.action_type == ActionType.ESCALATE:
            return TicketStatus.ESCALATED
        # Resolution happens when max turns reached with no escalation
        return current

    def _check_done(
        self,
        next_turn_idx: int,
        new_status: TicketStatus,
        action: Action,
    ) -> bool:
        # Escalation ends episode
        if new_status == TicketStatus.ESCALATED:
            return True

        # Ran out of scenario turns
        if next_turn_idx >= len(self._scenario.turns):
            return True

        # Hit max turn limit
        if self._step_count >= self._scenario.max_turns:
            return True

        return False

    # ─────────────────────────────────────────
    # Convenience helpers
    # ─────────────────────────────────────────

    @property
    def cumulative_reward(self) -> float:
        return self._cumulative_reward

    @property
    def is_done(self) -> bool:
        return self._done
