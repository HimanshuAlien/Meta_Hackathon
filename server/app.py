import os
import sys
from typing import Optional

from fastapi import FastAPI
import uvicorn

# ── path bootstrap (run from server/ directory) ──
# We move up ONE step to reach the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.customer_support_env import CustomerSupportEnv
from env.models import Action, Observation
from agent.hf_agent import build_agent
from grader.grader import EasyGrader, MediumGrader, HardGrader
from tasks.easy_task import EasyTask, EASY_SCENARIOS
from tasks.medium_task import MediumTask, MEDIUM_SCENARIOS
from tasks.hard_task import HardTask, HARD_SCENARIOS

from fastapi.responses import HTMLResponse

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Customer Support Escalation API", description="Automated evaluation endpoints for HF Spaces")

# ── CORS Middleware (Essential for cross-origin evaluator dashboards) ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Meta Hackathon — OpenEnv API</title>
            <style>
                body { font-family: 'Inter', sans-serif; background: #0f172a; color: #f8fafc; text-align: center; padding: 50px; }
                .card { background: #1e293b; border-radius: 12px; padding: 40px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.5); display: inline-block; max-width: 600px; }
                h1 { color: #38bdf8; font-size: 2.5rem; }
                .status { background: #10b981; color: white; padding: 5px 15px; border-radius: 9999px; font-weight: bold; font-size: 0.9rem; }
                ul { text-align: left; margin-top: 20px; list-style: none; padding: 0; }
                li { padding: 10px; border-bottom: 1px solid #334155; font-family: monospace; }
                li:last-child { border: none; }
                .footer { margin-top: 30px; color: #64748b; font-size: 0.8rem; }
            </style>
        </head>
        <body>
            <div class="card">
                <h1>🤖 OpenEnv AI Console</h1>
                <p><span class="status">🟢 READY FOR EVALUATION</span></p>
                <p>Welcome to the <strong>Customer Support Escalation</strong> API.</p>
                <ul>
                    <li>GET /reset &nbsp;&nbsp;&nbsp; - Initialize environment</li>
                    <li>POST /step &nbsp;&nbsp;&nbsp; - Execute agent action</li>
                    <li>GET /baseline - Run full evaluation (Easy/Med/Hard)</li>
                    <li>GET /state &nbsp;&nbsp;&nbsp; - View current observation</li>
                </ul>
                <p class="footer">OpenEnv Meta Hackathon 2026 — Developed by HimanshuAlien</p>
            </div>
        </body>
    </html>
    """

# Global variables for single-instance simulation
env_instance = CustomerSupportEnv(seed=42)
last_episode_score = 0.0

@app.get("/reset")
@app.post("/reset")
def reset_env(scenario_id: Optional[str] = None) -> Observation:
    """Reset the environment to a particular scenario."""
    global last_episode_score
    last_episode_score = 0.0
    obs = env_instance.reset(scenario_id=scenario_id)
    return obs

@app.post("/step")
def step_env(action: Action) -> dict:
    """Take a single step in the environment."""
    global last_episode_score
    obs, reward, done, info = env_instance.step(action)
    last_episode_score += reward
    return {
        "observation": obs,
        "reward": float(reward),
        "done": done,
        "info": info
    }

@app.get("/state")
def get_state() -> Observation:
    """Return the current environment observation."""
    return env_instance.state()

@app.get("/baseline")
@app.post("/baseline")
def run_baseline() -> dict:
    """Run all 3 tasks and return results."""
    # Respect environment variables set in HF Spaces
    model_name = os.getenv("MODEL_NAME", "distilgpt2")
    hf_token = os.getenv("HF_TOKEN")
    
    agent_fn = build_agent(model_id=model_name, hf_token=hf_token)
    
    # 1. Easy
    easy_task = EasyTask()
    easy_grader = EasyGrader()
    easy_scores = []
    for sid, _ in EASY_SCENARIOS:
        result = easy_task.run(agent_fn, scenario_id=sid)
        easy_scores.append(easy_grader.grade(result))
    import math
    def clamp(v) -> float:
        try:
            val = float(v)
        except (TypeError, ValueError):
            return 0.5
        if math.isnan(val) or math.isinf(val):
            return 0.5
        return max(0.01, min(0.99, val))

    easy_final = clamp(sum(easy_scores) / len(easy_scores) if easy_scores else 0.5)
    
    # 2. Medium
    medium_task = MediumTask()
    medium_grader = MediumGrader()
    medium_scores = []
    for sid in MEDIUM_SCENARIOS:
        result = medium_task.run(agent_fn, scenario_id=sid)
        medium_scores.append(medium_grader.grade(result))
    medium_final = clamp(sum(medium_scores) / len(medium_scores) if medium_scores else 0.5)
    
    # 3. Hard
    hard_task = HardTask()
    hard_grader = HardGrader()
    hard_scores = []
    for sid in HARD_SCENARIOS:
        result = hard_task.run(agent_fn, scenario_id=sid)
        hard_scores.append(hard_grader.grade(result))
    hard_final = clamp(sum(hard_scores) / len(hard_scores) if hard_scores else 0.5)
    
    overall = clamp((easy_final + medium_final + hard_final) / 3.0)
    
    return {
        "easy": easy_final,
        "medium": medium_final,
        "hard": hard_final,
        "overall": overall
    }

@app.get("/grader")
def get_grader_score() -> dict:
    """Return the cumulative score from the most recently completed episode."""
    import math
    val = float(last_episode_score)
    val = max(0.01, min(0.99, val))
    if math.isnan(val) or math.isinf(val):
        val = 0.5
    return {
        "score": round(val, 4)
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
