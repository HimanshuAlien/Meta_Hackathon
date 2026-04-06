import os
import sys
import json
import logging
from typing import Callable, Any

# ──────── Bootstrap Path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ──────── Environment Variables ───────────────────────────────────────────
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

from env.customer_support_env import CustomerSupportEnv
from env.models import Action, Observation
from agent.hf_agent import build_agent
from grader.grader import EasyGrader, MediumGrader, HardGrader
from tasks.easy_task import EasyTask, EASY_SCENARIOS
from tasks.medium_task import MediumTask, MEDIUM_SCENARIOS
from tasks.hard_task import HardTask, HARD_SCENARIOS

# ──────── Structured Logging ──────────────────────────────────────────────
def log_start(task_id: str):
    print(f"[START] {json.dumps({'task_id': task_id})}", flush=True)

def log_step(step_idx: int, action: Any, observation: Any, reward: float):
    # Handle both pydantic models and dicts
    a_dict = action.model_dump() if hasattr(action, "model_dump") else action
    o_dict = observation.model_dump() if hasattr(observation, "model_dump") else observation
    
    print(f"[STEP] {json.dumps({
        'step': step_idx,
        'action': a_dict,
        'observation': o_dict,
        'reward': float(reward)
    })}", flush=True)

def log_end(score: float):
    print(f"[END] {json.dumps({'score': float(score)})}", flush=True)

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
    return cumulative

# ──────── Task Runners ────────────────────────────────────────────────────
def run_all_tasks():
    # Performance-tuned agent (Hugging Face + Rule Hybrid)
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
        easy_scores.append(easy_grader.grade(result).score)
    easy_final = sum(easy_scores) / len(easy_scores) if easy_scores else 0.0
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
        medium_scores.append(medium_grader.grade(result).score)
    medium_final = sum(medium_scores) / len(medium_scores) if medium_scores else 0.0
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
        hard_scores.append(hard_grader.grade(result).score)
    hard_final = sum(hard_scores) / len(hard_scores) if hard_scores else 0.0
    log_end(hard_final)
    task_scores["Hard Task"] = hard_final

    # ──────── Human-Readable Summary (Printed to stderr to keep stdout clean) ─────
    overall = sum(task_scores.values()) / len(task_scores)
    sys.stderr.write("\n" + "="*50 + "\n")
    sys.stderr.write("  EVALUATION SUMMARY\n")
    sys.stderr.write("="*50 + "\n")
    for task, score in task_scores.items():
        sys.stderr.write(f"  {task:<30} : {score:.4f}\n")
    sys.stderr.write("-" * 50 + "\n")
    sys.stderr.write(f"  OVERALL AGGREGATE SCORE        : {overall:.4f}\n")
    sys.stderr.write("="*50 + "\n\n")

if __name__ == "__main__":
    run_all_tasks()
