from __future__ import annotations

import logging
import os
import contextlib
import io
import warnings
import random
import json
import time
from typing import Callable, Optional

from transformers import logging as hf_logging

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

# The validator expects the direct import
from openai import OpenAI

from env.models import Action, ActionType, Observation, TicketCategory

logger = logging.getLogger(__name__)


class HybridAgent:
    """
    OpenEnv-compliant Agent.
    Prioritizes the validator's LiteLLM proxy for all actions.
    """

    def __init__(
        self,
        model_id: str = "mistralai/Mistral-7B-Instruct-v0.2",
        hf_token: Optional[str] = None,
        use_gpu: bool = False,
        force_fallback: bool = False,
    ):
        self._classified = False
        self._has_responded = False
        self._escalated = False
        self._past_responses_sent = set()
        self.model_id = os.environ.get("MODEL_NAME", model_id)
        if self.model_id in ["distilgpt2", "gpt-3.5-turbo"]:
            self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self._openai_client = None
        try:
            self._openai_client = OpenAI(
                base_url=os.environ["API_BASE_URL"],
                api_key=os.environ["API_KEY"]
            )
        except Exception: pass

    def reset(self):
        self._classified = False
        self._has_responded = False
        self._escalated = False
        self._past_responses_sent = set()

    def __call__(self, obs: Observation) -> Action:
        # 1. Classification
        if not self._classified and str(obs.ticket_category) == "unclassified":
            self._classified = True
            category = TicketCategory.GENERAL
            if self._openai_client:
                try:
                    res = self._openai_client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": "Classify: billing, technical, general.\n" + str(obs.user_message)}],
                        max_tokens=10,
                        temperature=0.01
                    )
                    c = utils_extract_content(res).lower()
                    if "billing" in c: category = TicketCategory.BILLING
                    elif "technical" in c: category = TicketCategory.TECHNICAL
                except Exception:
                    m = str(obs.user_message).lower()
                    if "charge" in m or "refund" in m: category = TicketCategory.BILLING
                    elif "bug" in m or "error" in m: category = TicketCategory.TECHNICAL
            return Action(action_type=ActionType.CLASSIFY_TICKET, category=category)

        # 2. Escalation
        if not self._escalated:
            needs_esc = False
            if self._openai_client:
                try:
                    res = self._openai_client.chat.completions.create(
                        model=self.model_id,
                        messages=[{"role": "user", "content": "Escalate? YES/NO: " + str(obs.user_message)}],
                        max_tokens=5,
                        temperature=0.01
                    )
                    if "yes" in utils_extract_content(res).lower(): needs_esc = True
                except Exception:
                    if obs.user_emotion == "angry" or obs.is_vip: needs_esc = True
            if needs_esc:
                self._escalated = True
                return Action(action_type=ActionType.ESCALATE)

        # 3. Respond
        self._has_responded = True
        txt = "I understand the issue and will help you. [Support Response]"
        if self._openai_client:
            try:
                res = self._openai_client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": "Reply to: " + str(obs.user_message)}],
                    max_tokens=60,
                    temperature=0.7
                )
                g = utils_extract_content(res)
                if len(g) > 5: txt = g
            except Exception: pass
        self._past_responses_sent.add(txt.lower())
        return Action(action_type=ActionType.RESPOND, text=txt)

class RuleBasedAgent:
    def __call__(self, obs: Observation) -> Action:
        if str(obs.ticket_category) == "unclassified":
            return Action(action_type=ActionType.CLASSIFY_TICKET, category=TicketCategory.GENERAL)
        return Action(action_type=ActionType.RESPOND, text="Help is on the way.")

def utils_extract_content(res) -> str:
    try:
        if hasattr(res, "choices") and len(res.choices) > 0:
            return res.choices[0].message.content.strip()
    except Exception: pass
    return "Neutral response."

def build_agent(model_id: str = "mistralai/Mistral-7B-Instruct-v0.2", hf_token: Optional[str] = None, **kwargs):
    # Ensure no None or binary logic leaks
    return HybridAgent(model_id=model_id, hf_token=hf_token)


