# AI Customer Support Escalation — OpenEnv Agentic Environment

An OpenEnv-compliant reinforcement learning and agentic execution environment designed for training and evaluating customer support AI agents. This project was developed for the **OpenEnv Meta Hackathon**.

---

## 🎯 Task Overview

The environment simulates a realistic customer support ticket desk. The AI agent must:
1.  **Classify Tickets**: Assign incoming issues to `billing`, `technical`, or `general` categories.
2.  **Generate Responses**: Provide professional, helpful, and empathetic responses to customers.
3.  **Decide Escalation**: Identify high-priority or complex cases (VIP customers, fraud, data loss) and escalate them to a human manager at the correct time.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Evaluation Baseline
This executes all 3 tasks (Easy, Medium, Hard) and produces structured logs for the OpenEnv grader.
```bash
python inference.py
```

### 3. Start the OpenEnv API Server
Run the local environment server (compatible with Hugging Face Spaces):
```bash
python api.py
```

---

## 📁 Repository Structure

*   **/env**: Core OpenEnv environment logic (reset/step/state endpoints).
*   **/agent**: Hybrid AI Agent (Rules + Hugging Face Transformers).
*   **/tasks**: Difficulty levels (Easy Ticket Classification to Hard Multi-turn Escalation).
*   **/grader**: Automated scoring logic producing scores $\in [0, 1]$.
*   `openenv.yaml`: Official environment manifest.
*   `inference.py`: Standardized entry point for evaluation and logging.
*   `api.py`: FastAPI server for remote interaction.
*   `Dockerfile`: Containerization setup for Hugging Face Spaces.

---

## 📄 OpenEnv Compliance

*   **API Standard**: Implements `step()`, `reset()`, and `state()` endpoints as defined in `api.py`.
*   **Logging**: `inference.py` emits structured `[START]`, `[STEP]`, and `[END]` JSON logs to `stdout`.
*   **Deployment**: Fully Dockerized for seamless hosting on Hugging Face Spaces.

---

## 🛠️ Configuration

The agent uses a **Hugging Face + Rule-Based Hybrid** approach. By default, it uses `distilgpt2` for lightweight inference, making it fully compatible with free-tier hosting.

**Environment Variables**:
- `MODEL_NAME`: Hugging Face model ID (default: `distilgpt2`).
- `HF_TOKEN`: (Optional) Your Hugging Face API key for private models.

---

## ⚖️ License
MIT License. Developed for the OpenEnv Meta Hackathon.
