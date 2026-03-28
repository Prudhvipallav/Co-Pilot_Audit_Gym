---
title: Governance Review
emoji: 🏛️
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
license: mit
short_description: Real-world AI Governance Review Environment
---

# 🏛️ GovernanceReview-Gym v3

**A self-improving RL environment where a small language model (Qwen2.5-0.5B) learns enterprise AI governance review from scratch.**

## Architecture

| Agent | Model | Role |
|---|---|---|
| **Reviewer** | Qwen2.5-0.5B-Instruct (local) | Step-by-step governance review |
| **Problem Maker** | GPT-4o-mini (API) | Adversarial scenario generation targeting agent weaknesses |
| **Judge** | GPT-4o (API) | 3-persona evaluation (Junior/Senior/Principal GRC) |

## Key Features

- **Cold-start memory**: Agent bootstraps from a governance knowledge base and learns from every episode
- **Adversarial training**: Problem maker targets the agent's weakness map (EMA-tracked miss rates per violation)
- **Curriculum controller**: Mastery-gated difficulty escalation (easy → medium → hard → adversarial)
- **8 policy rules**: PII-001 through EVAL-008, each with deterministic checkers
- **3 static tasks** + infinite adversarially generated scenarios

## API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /reset` | Initialize environment |
| `POST /step` | Execute one action |
| `GET /grader` | Get deterministic grader score |
| `POST /generate_task` | Generate new scenario |
| `POST /adversarial_task` | Weakness-targeted scenario |
| `POST /judge` | 3-persona LLM judge |
| `GET /curriculum` | Curriculum + weakness map |

## Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8000 &
python3 demo/app.py
```

## Train (AWS)

See `AWS_TRAINING_GUIDE.md` for full instructions.

```bash
python3 agents/orchestrator.py --mode adversarial --n_episodes 50
python3 training/run_training.py --total_steps 200
```
