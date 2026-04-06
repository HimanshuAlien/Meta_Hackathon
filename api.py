import os
import sys
from typing import Optional

from fastapi import FastAPI
import uvicorn

# ── path bootstrap (run from project root) ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.customer_support_env import CustomerSupportEnv
from env.models import Action, Observation
from agent.hf_agent import build_agent
from run_agent import run_easy_task, run_medium_task, run_hard_task

app = FastAPI(title="Customer Support Escalation API", description="Automated evaluation endpoints for HF Spaces")

# Global variables for single-instance simulation
env = CustomerSupportEnv(seed=42)
last_episode_score = 0.0


@app.get("/reset")
@app.post("/reset")
def reset_env(scenario_id: Optional[str] = None) -> Observation:
    """Reset the environment, optionally to a specific scenario."""
    obs = env.reset(scenario_id=scenario_id)
    return obs


@app.post("/step")
def step_env(action: Action) -> dict:
    """Apply an action to the environment."""
    global last_episode_score
    obs, reward, done, info = env.step(action)
    
    if done:
        last_episode_score = info.get("cumulative_reward", 0.0)
        
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }


@app.get("/state")
def get_state() -> Observation:
    """Return the current environment state."""
    return env.state()


@app.get("/tasks")
def list_tasks() -> dict:
    """Return available tasks and action schema."""
    return {
        "tasks": ["easy", "medium", "hard"],
        "actions": ["classify_ticket", "respond", "ask_info", "escalate"]
    }


@app.get("/baseline")
@app.post("/baseline")
def run_baseline() -> dict:
    """Run all 3 tasks and return results."""
    # Respect environment variables set in HF Spaces
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    hf_token = os.getenv("HF_TOKEN")
    
    agent_fn = build_agent(model_id=model_name, hf_token=hf_token)
    
    easy_score = run_easy_task(agent_fn, 1)
    medium_score = run_medium_task(agent_fn, 1)
    hard_score = run_hard_task(agent_fn, 1)
    
    overall = round((easy_score + medium_score + hard_score) / 3, 4)
    
    return {
        "easy": easy_score,
        "medium": medium_score,
        "hard": hard_score,
        "overall": overall
    }


@app.get("/grader")
def get_grader_score() -> dict:
    """Return the cumulative score from the most recently completed episode."""
    return {
        "score": last_episode_score
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
