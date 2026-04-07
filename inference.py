"""
GovernanceReview-Gym Inference Script
===================================
MANDATORY for OpenEnv Hackathon Submission.
Uses OpenAI Client to run a baseline agent against the environment.
"""

import os
import re
import json
import time
import argparse
from typing import List, Optional, Dict
from openai import OpenAI
import requests

# Required environment variables (per hackathon spec)
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
ENV_URL = os.getenv("GOVERNANCE_ENV_URL", "https://prudhvi06-co-pilot-audit-gym.hf.space")

MAX_STEPS = 25
TEMPERATURE = 0.2
MAX_TOKENS = 300

SYSTEM_PROMPT = """You are an AI governance reviewer. Review enterprise AI copilot launch packets.

AVAILABLE ACTIONS (Submit exactly one as JSON):
1. Inspect artifact: {"action_type": "inspect_artifact", "target": "filename.yaml"}
2. Flag violation: {"action_type": "flag_issue", "issue_code": "PII-001", "severity": "critical", "target": "filename.yaml", "note": "reason"}
3. Request mitigation: {"action_type": "request_mitigation", "issue_code": "PII-001", "note": "required fix"}
4. Set risk: {"action_type": "set_risk", "severity": "high"}
5. Final decision: {"action_type": "approve"} or {"action_type": "reject"} or {"action_type": "escalate"}

POLICY CODES:
- PII-001: PII logging without filter (critical)
- ACCESS-002: Missing access controls (high)
- ESCALATION-003: Incomplete escalation plan (high)
- DOMAIN-004: Prohibited domain (critical)
- RETENTION-005: Excess data retention (medium)
- TRAINING-006: Unauthorized model training (critical)
- AUDIT-007: Missing audit trail (medium)
- EVAL-008: Insufficient evaluation (high)

Rules:
- You MUST inspect artifacts before flagging or deciding.
- Only one action per turn.
- Output ONLY valid JSON."""

def run_inference_episode(task_id: int, client: OpenAI):
    """Run a single episode against the environment."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()["observation"]
    task_name = obs.get("feature_name", f"task_{task_id}")

    print(f"[START] task={task_name}", flush=True)

    history = []
    total_reward = 0.0
    step = 0

    for step in range(1, MAX_STEPS + 1):
        inspected = obs.get("inspected_artifacts", [])
        visible = list(obs.get("visible_artifacts", {}).keys())
        flagged = [f['code'] for f in obs.get("flagged_issues", [])]

        user_prompt = (
            f"Step {step}/{MAX_STEPS}\n"
            f"Visible Artifacts: {visible}\n"
            f"Inspected: {inspected}\n"
            f"Flagged Issues: {flagged}\n"
            f"Current Risk: {obs.get('current_risk')}\n"
            f"Last Message: {obs.get('message', '')[:300]}\n"
            f"What is your next action? Return JSON."
        )

        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content or ""
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            action_json = json_match.group() if json_match else response_text
            action_dict = json.loads(action_json)
        except Exception as e:
            print(f"[STEP] step={step} reward=0.0 action=error error={e}", flush=True)
            break

        resp = requests.post(f"{ENV_URL}/step", json={"action": json.dumps(action_dict)})
        result = resp.json()

        obs = result["observation"]
        reward = result["reward"]
        total_reward += reward

        print(f"[STEP] step={step} reward={reward:.4f} action={action_dict.get('action_type','?')}", flush=True)

        if result["done"]:
            break

    grader_resp = requests.get(f"{ENV_URL}/grader")
    score_data = grader_resp.json()
    score = score_data['scores']['overall']

    print(f"[END] task={task_name} score={score:.4f} steps={step}", flush=True)
    return score

def run_rule_based_episode(task_id: int):
    """Fallback: deterministic rule-based agent when no API key available."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()["observation"]
    task_name = obs.get("feature_name", f"task_{task_id}")

    print(f"[START] task={task_name}", flush=True)

    total_reward = 0.0
    step = 0

    # Step 1: Inspect all artifacts
    for art_name in list(obs.get("visible_artifacts", {}).keys()):
        step += 1
        r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
            "action_type": "inspect_artifact", "target": art_name
        })}).json()
        reward = r["reward"]
        total_reward += reward
        obs = r["observation"]
        print(f"[STEP] step={step} reward={reward:.4f} action=inspect_artifact", flush=True)
        if r["done"]:
            break

    # Step 2: Flag based on observation hints
    msg = obs.get("message", "").lower()
    known_flags = [
        ("PII-001", "critical"), ("ACCESS-002", "high"),
        ("ESCALATION-003", "high"), ("DOMAIN-004", "critical"),
        ("RETENTION-005", "medium"), ("TRAINING-006", "critical"),
        ("AUDIT-007", "medium"), ("EVAL-008", "high"),
    ]
    for code, sev in known_flags:
        if code.lower().replace("-", "") in msg.replace("-", "").replace("_", "").lower() or \
           code.split("-")[0].lower() in msg:
            step += 1
            r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
                "action_type": "flag_issue", "issue_code": code,
                "severity": sev, "target": "product_spec.md",
                "note": f"Auto-detected {code}"
            })}).json()
            reward = r["reward"]
            total_reward += reward
            obs = r["observation"]
            print(f"[STEP] step={step} reward={reward:.4f} action=flag_issue", flush=True)
            if r["done"]:
                break

            step += 1
            r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
                "action_type": "request_mitigation", "issue_code": code,
                "note": f"Remediate {code}"
            })}).json()
            reward = r["reward"]
            total_reward += reward
            obs = r["observation"]
            print(f"[STEP] step={step} reward={reward:.4f} action=request_mitigation", flush=True)
            if r["done"]:
                break

    # Step 3: Set risk and decide
    if not obs.get("done", False):
        step += 1
        r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
            "action_type": "set_risk", "severity": "high"
        })}).json()
        reward = r["reward"]
        total_reward += reward
        obs = r["observation"]
        print(f"[STEP] step={step} reward={reward:.4f} action=set_risk", flush=True)

    if not r.get("done", False):
        step += 1
        r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
            "action_type": "reject"
        })}).json()
        reward = r["reward"]
        total_reward += reward
        print(f"[STEP] step={step} reward={reward:.4f} action=reject", flush=True)

    grader_resp = requests.get(f"{ENV_URL}/grader")
    score_data = grader_resp.json()
    score = score_data['scores']['overall']

    print(f"[END] task={task_name} score={score:.4f} steps={step}", flush=True)
    return score


def main():
    # Check server
    try:
        requests.get(f"{ENV_URL}/health", timeout=5)
    except Exception:
        print(f"❌ Error: Environment not running at {ENV_URL}")
        print("Please start the server first: uvicorn app.main:app --port 8000")
        return

    # Determine agent mode
    use_llm = bool(HF_TOKEN)
    if use_llm:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        print(f"🤖 Using LLM agent: {MODEL_NAME}", flush=True)
    else:
        client = None
        print("⚠️  No HF_TOKEN found. Using rule-based fallback agent.", flush=True)

    scores = []
    for task_id in [1, 2, 3, 4]:
        try:
            if use_llm:
                score = run_inference_episode(task_id, client)
            else:
                score = run_rule_based_episode(task_id)
            scores.append(score)
        except Exception as e:
            print(f"❌ Task {task_id} failed: {e}")
            scores.append(0.0)

    avg_score = sum(scores) / len(scores)
    print(f"\n==========================================")
    print(f"BASELINE SUMMARY: Average Score = {avg_score:.4f}")
    print(f"Agent: {'LLM (' + MODEL_NAME + ')' if use_llm else 'Rule-Based Fallback'}")
    print(f"==========================================")

if __name__ == "__main__":
    main()
