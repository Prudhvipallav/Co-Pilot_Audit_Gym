"""
Baseline Agent — GovernanceReview-Gym v3

A simple rule-based OpenAI GPT-4o-mini agent that achieves a baseline
grader score to compare against the Qwen2.5-0.5B trained reviewer.

Usage:
  python3 agents/baseline.py --task_id 1
  python3 agents/baseline.py --task_id all

Reports grader scores to stdout in table format.
"""

import os
import sys
import json
import requests
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("GOVERNANCE_ENV_URL", "http://localhost:8000")

SYSTEM_PROMPT = """You are an AI governance reviewer. Review enterprise AI copilot launch packets.

WORKFLOW — follow in order:
1. inspect_artifact for EACH artifact listed (inspect all of them)
2. flag_issue for any policy violations you find (with correct issue_code + severity)
3. request_mitigation for each flagged issue
4. set_risk based on worst violation severity
5. Final decision: reject (critical violations), escalate (high/medium), approve (none)

Policy codes:
- PII-001: PII logging without filter (severity: critical)
- ACCESS-002: Missing access controls (severity: high)
- ESCALATION-003: Incomplete escalation plan (severity: high)
- DOMAIN-004: Prohibited domain (financial/legal advice) (severity: critical)
- RETENTION-005: Excess data retention (severity: medium)
- TRAINING-006: External provider training on employee data (severity: critical)
- AUDIT-007: Missing audit trail (severity: medium)
- EVAL-008: Insufficient safety evaluation (severity: high)

Return ONLY valid JSON. One action per response. No extra text.
Example: {"action_type": "inspect_artifact", "target": "logging_policy.yaml"}"""


def call_openai(system: str, user: str) -> str:
    """Call OpenAI API for baseline agent decisions."""
    import openai
    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


def run_baseline_episode(task_id: int, verbose: bool = True) -> dict:
    """Run a single baseline episode."""
    # Reset
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    resp.raise_for_status()
    obs = resp.json()["observation"]

    if verbose:
        print(f"\n{'='*50}")
        print(f"Task {task_id}: {obs.get('feature_name')}")
        print(f"{'='*50}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    total_reward = 0.0
    steps = 0

    for step in range(25):
        # Build state summary for GPT
        inspected = obs.get("inspected_artifacts", [])
        artifacts = list(obs.get("visible_artifacts", {}).keys())
        flagged = obs.get("flagged_issues", [])
        mitigated = [m.get("issue_code") for m in obs.get("requested_mitigations", [])]

        remaining = [a for a in artifacts if a not in inspected]
        state_msg = (
            f"Artifacts: {artifacts}\n"
            f"Inspected: {inspected}\n"
            f"Remaining to inspect: {remaining}\n"
            f"Flagged: {[f['code'] for f in flagged]}\n"
            f"Mitigated: {mitigated}\n"
            f"Risk: {obs.get('current_risk')}\n"
            f"Stage: {obs.get('review_stage')}\n"
            f"Last message: {obs.get('message', '')[:200]}\n"
            f"What is your next action? Return JSON."
        )

        messages.append({"role": "user", "content": state_msg})

        try:
            action_text = call_openai(SYSTEM_PROMPT, state_msg)
        except Exception as e:
            if verbose:
                print(f"  [Step {step+1}] API error: {e}")
            break

        # Parse and display
        try:
            action_dict = json.loads(action_text)
        except json.JSONDecodeError:
            # Try to extract JSON
            import re
            m = re.search(r'\{[^{}]*\}', action_text)
            action_dict = json.loads(m.group()) if m else {"action_type": "escalate", "note": "parse error"}
            action_text = json.dumps(action_dict)

        atype = action_dict.get("action_type", "?")
        detail = action_dict.get("target") or action_dict.get("issue_code") or ""

        # Step
        resp = requests.post(f"{BASE_URL}/step", json={"action": action_text})
        resp.raise_for_status()
        result = resp.json()
        obs = result["observation"]
        reward = result["reward"]
        total_reward += reward
        steps += 1

        if verbose:
            print(f"  Step {step+1:2d}: {atype:<25} {detail:<20} → {reward:+.2f}")

        messages.append({"role": "assistant", "content": action_text})

        if result["done"]:
            break

    # Get scores
    grader_resp = requests.get(f"{BASE_URL}/grader")
    grader = grader_resp.json()

    state_resp = requests.get(f"{BASE_URL}/state")
    state = state_resp.json()

    if verbose:
        scores = grader["scores"]
        print(f"\nGrader: {scores['overall']:.4f} | Grade: {grader['grade']}")
        print(f"Caught: {[i['code'] for i in state.get('flagged_issues', [])]}")

    return {
        "task_id": task_id,
        "feature_name": obs.get("feature_name"),
        "grader_score": grader["scores"],
        "grade": grader["grade"],
        "total_reward": total_reward,
        "steps": steps,
        "final_decision": state.get("final_decision"),
        "flagged": [i["code"] for i in state.get("flagged_issues", [])],
    }


def run_all_tasks():
    """Run all 3 tasks and print comparison table."""
    results = []
    for task_id in [1, 2, 3]:
        try:
            r = run_baseline_episode(task_id)
            results.append(r)
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            results.append({"task_id": task_id, "grader_score": {"overall": 0}, "grade": "F", "steps": 0})

    print("\n" + "="*70)
    print(f"{'BASELINE AGENT (GPT-4o-mini) SUMMARY':^70}")
    print("="*70)
    print(f"{'Task':<6} {'Feature':<30} {'Score':>8} {'Grade':>8} {'Steps':>6}")
    print("-"*70)
    for r in results:
        name = r.get("feature_name", "?")[:28]
        score = r.get("grader_score", {}).get("overall", 0)
        print(f"  {r['task_id']:<4} {name:<30} {score:>8.4f} {r.get('grade', '?'):>8} {r.get('steps', 0):>6}")
    avg = sum(r.get("grader_score", {}).get("overall", 0) for r in results) / len(results)
    print("-"*70)
    print(f"{'AVERAGE':^42} {avg:>8.4f}")
    print("="*70)
    print("\nThis is the BASELINE. The trained Qwen2.5-0.5B should match or exceed this.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline GPT-4o-mini agent")
    parser.add_argument("--task_id", default="1", help="Task ID (1, 2, 3, or 'all')")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    try:
        requests.get(f"{BASE_URL}/health").raise_for_status()
    except Exception:
        print(f"❌ Server not running at {BASE_URL}")
        print("Run: python3 -m uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("❌ Set OPENAI_API_KEY environment variable")
        sys.exit(1)

    if args.task_id == "all":
        run_all_tasks()
    else:
        run_baseline_episode(int(args.task_id), verbose=not args.quiet)
