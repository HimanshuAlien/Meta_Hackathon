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

# ──────── Structured Logging ──────────────────────────────────────────────
def log_start(task_id: str):
    print("[START] " + json.dumps({'task_id': task_id}), flush=True)

def log_step(step_idx: int, action: Any, observation: Any, reward: float):
    try:
        a_dict = action.model_dump() if hasattr(action, "model_dump") else action
        o_dict = observation.model_dump() if hasattr(observation, "model_dump") else observation
        
        log_data = {
            'step': step_idx,
            'action': a_dict,
            'observation': o_dict,
            'reward': clamp(reward)
        }
        print("[STEP] " + json.dumps(log_data), flush=True)
    except Exception as e:
        sys.stderr.write("[LOG_ERROR] Failed to log step: " + str(e) + "\n")

def log_end(score: float):
    print("[END] " + json.dumps({'score': clamp(score)}), flush=True)

# ──────── Episode Runner ──────────────────────────────────────────────────
def run_episode_logged(env: CustomerSupportEnv, agent_fn: Callable, scenario_id: str):
    obs = env.reset(scenario_id=scenario_id)
    if hasattr(agent_fn, "reset"):
        agent_fn.reset()
    
    done = False
    step = 0
    cumulative = 0.0
    
    while not done:
        action = agent_fn(obs)
        obs, reward, done, info = env.step(action)
        step += 1
        cumulative += reward
        
        log_step(
            step_idx=step,
            action=action,
            observation=obs,
            reward=reward
        )
    return clamp(cumulative)

# ──────── Task Runners ────────────────────────────────────────────────────
def run_all_tasks():
    from agent.hf_agent import build_agent
    agent_fn = build_agent(model_id=MODEL_NAME, hf_token=HF_TOKEN)
    env = CustomerSupportEnv(seed=42)
    
    task_scores = {}

    # 1. Easy Task
    log_start("easy_classify_ticket")
    easy_task = EasyTask()
    easy_grader = EasyGrader()
    easy_scores = []
    for sid, _ in EASY_SCENARIOS:
        run_episode_logged(env, agent_fn, sid)
        result = easy_task.run(agent_fn, scenario_id=sid)
        easy_scores.append(clamp(easy_grader.grade(result).score))
    easy_final = clamp(sum(easy_scores) / len(easy_scores) if easy_scores else 0.5)
    log_end(easy_final)
    task_scores["Easy Task"] = easy_final
    
    # 2. Medium Task
    log_start("medium_generate_response")
    medium_task = MediumTask()
    medium_grader = MediumGrader()
    medium_scores = []
    for sid in MEDIUM_SCENARIOS:
        run_episode_logged(env, agent_fn, sid)
        result = medium_task.run(agent_fn, scenario_id=sid)
        medium_scores.append(clamp(medium_grader.grade(result).score))
    medium_final = clamp(sum(medium_scores) / len(medium_scores) if medium_scores else 0.5)
    log_end(medium_final)
    task_scores["Medium Task"] = medium_final
    
    # 3. Hard Task
    log_start("hard_multi_turn_escalation")
    hard_task = HardTask()
    hard_grader = HardGrader()
    hard_scores = []
    for sid in HARD_SCENARIOS:
        run_episode_logged(env, agent_fn, sid)
        result = hard_task.run(agent_fn, scenario_id=sid)
        hard_scores.append(clamp(hard_grader.grade(result).score))
    hard_final = clamp(sum(hard_scores) / len(hard_scores) if hard_scores else 0.5)
    log_end(hard_final)
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
