import os
import sys
import json
import logging
from typing import Callable, Any

# ──────── Bootstrap Path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────── Environment Variables & CLI ──────────────────────────────────────
MODELS = ["distilgpt2", "gpt2", "microsoft/phi-2"] # Examples
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
HF_TOKEN = os.getenv("HF_TOKEN")

if "--hf-token" in sys.argv:
    idx = sys.argv.index("--hf-token")
    if idx + 1 < len(sys.argv):
        HF_TOKEN = sys.argv[idx + 1]

if "--use-hf" in sys.argv:
    # Just a flag to indicate we should definitely try loading the model
    pass

import traceback
from env.customer_support_env import CustomerSupportEnv
from env.models import Action, Observation
# build_agent moved into run_all_tasks for lazy import safety
from grader.grader import EasyGrader, MediumGrader, HardGrader
from tasks.easy_task import EasyTask, EASY_SCENARIOS
from tasks.medium_task import MediumTask, MEDIUM_SCENARIOS
from tasks.hard_task import HardTask, HARD_SCENARIOS

import math

def clamp(v: Any) -> float:
    """Bulletproof clamp: always returns a value strictly in (0.01, 0.99)."""
    try:
        val = float(v)
    except (TypeError, ValueError):
        return 0.5
    if math.isnan(val) or math.isinf(val):
        return 0.5
    return max(0.01, min(0.99, val))

# ──────── Structured Logging via Monkey-Patching ─────────────────────────
current_task_name = ""
current_steps = 0
current_rewards = []

original_reset = CustomerSupportEnv.reset
original_step = CustomerSupportEnv.step

def format_action(a: Any) -> str:
    try:
        atype = a.action_type.value if hasattr(a.action_type, "value") else str(a.action_type)
        if atype == "classify_ticket":
            cat = a.category.value if hasattr(a.category, "value") else str(a.category)
            return f"classify_ticket('{cat}')"
        elif atype in ("respond", "ask_info"):
            txt = str(a.text).replace('\n', ' ').replace('\r', '')
            # Truncate text if too long to avoid huge lines, or keep it. We'll keep it.
            return f"{atype}('{txt}')"
        elif atype == "escalate":
            return "escalate()"
        return f"{atype}()"
    except Exception:
        return "action()"

def patched_reset(self, scenario_id=None):
    global current_steps, current_rewards
    current_steps = 0
    current_rewards = []
    print(f"[START] task={current_task_name} env=customer_support model={MODEL_NAME}", flush=True)
    return original_reset(self, scenario_id=scenario_id)

def patched_step(self, action):
    global current_steps, current_rewards
    obs, reward, done, info = original_step(self, action)
    current_steps += 1
    current_rewards.append(reward)
    
    act_str = format_action(action)
    d_str = "true" if done else "false"
    print(f"[STEP] step={current_steps} action={act_str} reward={reward:.2f} done={d_str} error=null", flush=True)
    return obs, reward, done, info

CustomerSupportEnv.reset = patched_reset
CustomerSupportEnv.step = patched_step

def log_end(success: bool, steps: int, score: float, rewards: list):
    r_str = ",".join([f"{r:.2f}" for r in rewards])
    if not r_str: r_str = "0.00"
    s_str = "true" if success else "false"
    print(f"[END] success={s_str} steps={steps} score={score:.3f} rewards={r_str}", flush=True)


# ──────── Task Runners ────────────────────────────────────────────────────
def run_all_tasks():
    from agent.hf_agent import build_agent
    agent_fn = build_agent(model_id=MODEL_NAME, hf_token=HF_TOKEN)
    env = CustomerSupportEnv(seed=42)
    
    task_scores = {}

    global current_task_name
    
    # 1. Easy Task
    current_task_name = "easy_classify_ticket"
    easy_task = EasyTask()
    easy_grader = EasyGrader()
    easy_scores = []
    for sid, _ in EASY_SCENARIOS:
        result = easy_task.run(agent_fn, scenario_id=sid)
        score = clamp(easy_grader.grade(result))
        easy_scores.append(score)
        success = score >= easy_grader.PASS_THRESHOLD
        log_end(success=success, steps=current_steps, score=score, rewards=current_rewards)
    easy_final = clamp(sum(easy_scores) / len(easy_scores) if easy_scores else 0.5)
    task_scores["Easy Task"] = easy_final
    
    # 2. Medium Task
    current_task_name = "medium_generate_response"
    medium_task = MediumTask()
    medium_grader = MediumGrader()
    medium_scores = []
    for sid in MEDIUM_SCENARIOS:
        result = medium_task.run(agent_fn, scenario_id=sid)
        score = clamp(medium_grader.grade(result))
        medium_scores.append(score)
        success = score >= medium_grader.PASS_THRESHOLD
        log_end(success=success, steps=current_steps, score=score, rewards=current_rewards)
    medium_final = clamp(sum(medium_scores) / len(medium_scores) if medium_scores else 0.5)
    task_scores["Medium Task"] = medium_final
    
    # 3. Hard Task
    current_task_name = "hard_multi_turn_escalation"
    hard_task = HardTask()
    hard_grader = HardGrader()
    hard_scores = []
    for sid in HARD_SCENARIOS:
        result = hard_task.run(agent_fn, scenario_id=sid)
        score = clamp(hard_grader.grade(result))
        hard_scores.append(score)
        success = score >= hard_grader.PASS_THRESHOLD
        log_end(success=success, steps=current_steps, score=score, rewards=current_rewards)
    hard_final = clamp(sum(hard_scores) / len(hard_scores) if hard_scores else 0.5)
    task_scores["Hard Task"] = hard_final

    # ──────── Human-Readable Summary ─────
    overall = clamp(sum(task_scores.values()) / len(task_scores))
    sys.stderr.write("\n" + "="*50 + "\n")
    sys.stderr.write("  EVALUATION SUMMARY\n")
    sys.stderr.write("="*50 + "\n")
    for task, score in task_scores.items():
        sys.stderr.write("  " + str(task).ljust(30) + " : " + "{:.4f}".format(score) + "\n")
    sys.stderr.write("-" * 50 + "\n")
    sys.stderr.write("  OVERALL AGGREGATE SCORE        : " + "{:.4f}".format(overall) + "\n")
    sys.stderr.write("="*50 + "\n\n")

import traceback

def main():
    try:
        run_all_tasks()
        sys.exit(0)
    except Exception as e:
        # Capture full traceback for deep debugging
        err_msg = str(e)
        stack = traceback.format_exc()
        error_output = {
            'error': err_msg,
            'traceback': stack,
            'step': 'main_execution'
        }
        print("[ERROR] " + json.dumps(error_output), flush=True)
        # We still log to stderr for local visibility
        sys.stderr.write("\n!!! CRITICAL ERROR !!!\n" + stack + "\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
