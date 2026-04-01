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

# Required environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
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
    # 1. Reset
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()["observation"]
    
    print(f"\n--- Starting Task {task_id}: {obs['feature_name']} ---")
    
    history = []
    total_reward = 0.0
    
    for step in range(1, MAX_STEPS + 1):
        # Build prompt
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
        
        # 2. Get LLM Action
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
            
            # Clean JSON
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            action_json = json_match.group() if json_match else response_text
            action_dict = json.loads(action_json)
        except Exception as e:
            print(f"Error at step {step}: {e}")
            break

        # 3. Step Environment
        resp = requests.post(f"{ENV_URL}/step", json={"action": json.dumps(action_dict)})
        result = resp.json()
        
        obs = result["observation"]
        reward = result["reward"]
        total_reward += reward
        
        print(f"Step {step}: {action_dict.get('action_type')} -> Reward: {reward:+.2f}")
        
        if result["done"]:
            print("Episode finished.")
            break
            
    # 4. Get Grader Score
    grader_resp = requests.get(f"{ENV_URL}/grader")
    score_data = grader_resp.json()
    print(f"Final Score: {score_data['scores']['overall']:.4f} ({score_data['grade']})")
    return score_data['scores']['overall']

def run_rule_based_episode(task_id: int):
    """Fallback: deterministic rule-based agent when no API key available."""
    resp = requests.post(f"{ENV_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()["observation"]
    print(f"\n--- [Rule-Based] Task {task_id}: {obs['feature_name']} ---")

    total_reward = 0.0

    # Step 1: Inspect all artifacts
    for art_name in list(obs.get("visible_artifacts", {}).keys()):
        r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
            "action_type": "inspect_artifact", "target": art_name
        })}).json()
        total_reward += r["reward"]
        obs = r["observation"]
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
            r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
                "action_type": "flag_issue", "issue_code": code,
                "severity": sev, "target": "product_spec.md",
                "note": f"Auto-detected {code}"
            })}).json()
            total_reward += r["reward"]
            obs = r["observation"]
            if r["done"]:
                break

            # Request mitigation
            r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
                "action_type": "request_mitigation", "issue_code": code,
                "note": f"Remediate {code}"
            })}).json()
            total_reward += r["reward"]
            obs = r["observation"]
            if r["done"]:
                break

    # Step 3: Set risk and decide
    if not obs.get("done", False):
        r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
            "action_type": "set_risk", "severity": "high"
        })}).json()
        total_reward += r["reward"]
        obs = r["observation"]

    if not r.get("done", False):
        r = requests.post(f"{ENV_URL}/step", json={"action": json.dumps({
            "action_type": "reject"
        })}).json()
        total_reward += r["reward"]

    grader_resp = requests.get(f"{ENV_URL}/grader")
    score_data = grader_resp.json()
    print(f"Final Score: {score_data['scores']['overall']:.4f} ({score_data['grade']})")
    return score_data['scores']['overall']


def main():
    # Check server
    try:
        requests.get(f"{ENV_URL}/health", timeout=5)
    except Exception:
        print(f"❌ Error: Environment not running at {ENV_URL}")
        print("Please start the server first: uvicorn app.main:app --port 8000")
        return

    # Determine agent mode
    use_llm = bool(API_KEY)
    if use_llm:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        print(f"🤖 Using LLM agent: {MODEL_NAME}")
    else:
        client = None
        print("⚠️  No API key found (HF_TOKEN/API_KEY). Using rule-based fallback agent.")

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
