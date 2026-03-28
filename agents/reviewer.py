"""
Reviewer Agent v2 — Qwen2.5-0.5B-Instruct running locally.

Fixes from v1:
  - Stuck-loop detection: if same action repeats 2x consecutively → fallback planner
  - Better JSON validation: extracted JSON must have valid action_type + valid target
  - Improved system prompt with few-shot examples for Qwen 0.5B
  - max_steps capped at 15 (not 50) — inspection + flag + mitigate + decide = ~12 steps max
"""

import os
import sys
import json
import re
import requests
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.model_config import get_reviewer_client
from memory.knowledge_base import get_memory_context, save_episode_result

BASE_URL = os.getenv("GOVERNANCE_ENV_URL", "http://localhost:8000")

MAX_STEPS = 15  # Was 50 — way too many for a 9-artifact review

VALID_ACTIONS = {
    "inspect_artifact", "flag_issue", "request_mitigation",
    "set_risk", "approve", "reject", "escalate",
    "ask_question", "finalize_review",
}

# ─────────────────────────────────────────────────────────────────
# System prompt with few-shot examples (critical for 0.5B models)
# ─────────────────────────────────────────────────────────────────

REVIEWER_SYSTEM_TEMPLATE = """{memory_context}

=== YOUR TASK ===
You are an AI governance reviewer. Review enterprise AI copilot launch packets.

WORKFLOW (follow this order):
1. inspect_artifact for each artifact (9 total)
2. flag_issue for any violations found
3. request_mitigation for each flagged issue
4. set_risk based on worst violation severity
5. Final decision: reject / escalate / approve

Return ONLY valid JSON. One action per response. No extra text.

=== EXAMPLES ===
Step 1 (inspect first artifact):
{{"action_type": "inspect_artifact", "target": "product_spec.md"}}

Step 2 (inspect next):
{{"action_type": "inspect_artifact", "target": "logging_policy.yaml"}}

Step 5 (found a violation):
{{"action_type": "flag_issue", "issue_code": "PII-001", "severity": "critical", "target": "logging_policy.yaml", "note": "raw_prompt_capture enabled with pii_filter disabled"}}

Step 6 (request fix):
{{"action_type": "request_mitigation", "issue_code": "PII-001", "note": "Enable PII filter before launch"}}

Step 7 (set risk):
{{"action_type": "set_risk", "severity": "critical"}}

Step 8 (final decision):
{{"action_type": "reject", "note": "Critical PII violation must be fixed"}}
"""


def _extract_json(text: str) -> Optional[dict]:
    """Robustly extract JSON from Qwen output, with validation."""
    text = text.strip()

    # Remove markdown fences if present
    if "```" in text:
        parts = text.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            try:
                candidate = json.loads(part)
                if _validate_action(candidate):
                    return candidate
            except (json.JSONDecodeError, TypeError):
                continue

    # Try direct parse
    try:
        candidate = json.loads(text)
        if _validate_action(candidate):
            return candidate
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting first JSON object
    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
    if json_match:
        try:
            candidate = json.loads(json_match.group())
            if _validate_action(candidate):
                return candidate
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def _validate_action(action: dict) -> bool:
    """Validate that extracted JSON is actually a governance action."""
    if not isinstance(action, dict):
        return False
    action_type = action.get("action_type", "")
    if action_type not in VALID_ACTIONS:
        return False
    # inspect_artifact MUST have a target that looks like a filename
    if action_type == "inspect_artifact":
        target = action.get("target", "")
        if not target or "." not in target:
            return False  # "PII-001" is NOT a valid artifact name
    # flag_issue must have issue_code
    if action_type == "flag_issue":
        if not action.get("issue_code"):
            return False
    return True


def _plan_actions(obs: dict, inspected: list, flagged: list, mitigated: list) -> Optional[dict]:
    """
    Rule-based fallback planner. Follows the correct workflow order.
    """
    artifacts = list(obs.get("visible_artifacts", {}).keys())
    risk = obs.get("current_risk")

    # 1. Inspect uninspected artifacts
    for art in artifacts:
        if art not in inspected:
            return {"action_type": "inspect_artifact", "target": art}

    # 2. Flag issues (use policy checker on available content)
    if not flagged:
        # Run policy checks on the full artifacts we've seen
        try:
            from app.policies import run_all_checks
            full_artifacts = obs.get("visible_artifacts", {})
            results = run_all_checks(full_artifacts)
            for code, (passed, msg) in results.items():
                if not passed:
                    from app.policies import POLICY_RULE_MAP
                    rule = POLICY_RULE_MAP.get(code)
                    return {
                        "action_type": "flag_issue",
                        "issue_code": code,
                        "severity": rule.severity if rule else "high",
                        "target": rule.artifact_hint if rule else "",
                        "note": msg[:120],
                    }
        except Exception:
            pass

    # 3. Request mitigations for flagged issues
    for issue in flagged:
        code = issue.get("code", "")
        if code and code not in mitigated:
            return {
                "action_type": "request_mitigation",
                "issue_code": code,
                "note": f"Remediate {code} before launch"
            }

    # 4. Set risk if flagged issues exist
    if flagged and not risk:
        severities = [i.get("severity", "medium") for i in flagged]
        if "critical" in severities:
            return {"action_type": "set_risk", "severity": "critical"}
        elif "high" in severities:
            return {"action_type": "set_risk", "severity": "high"}
        else:
            return {"action_type": "set_risk", "severity": "medium"}

    # 5. Final decision
    if flagged:
        risk_level = risk or "medium"
        decision = "reject" if risk_level in ("critical",) else "escalate"
        return {"action_type": decision, "note": f"Policy violations detected: {[i.get('code') for i in flagged]}"}

    return {"action_type": "approve", "note": "No policy violations found in artifacts"}


def run_reviewer_episode(
    task_id: int = 1,
    use_generated: bool = False,
    save_memory: bool = True,
    verbose: bool = True,
) -> dict:
    """Run a complete Qwen2.5 reviewer agent episode."""
    client = get_reviewer_client()
    memory_context = get_memory_context(n_episodes=5)
    system_prompt = REVIEWER_SYSTEM_TEMPLATE.format(memory_context=memory_context)

    if verbose:
        print(f"[Reviewer] Model: {client.model_id} (local)")
        print(f"[Reviewer] Memory: {len(memory_context):,} chars injected into context")

    # Reset environment
    resp = requests.post(
        f"{BASE_URL}/reset",
        json={"task_id": task_id, "use_generated_task": use_generated}
    )
    resp.raise_for_status()
    obs = resp.json()["observation"]

    if verbose:
        print(f"[Reviewer] Task: {obs.get('feature_name')} | Stage: {obs.get('review_stage')}")

    total_reward = 0.0
    steps_log = []
    fallback_count = 0
    last_action_key = None
    repeat_count = 0

    for step_num in range(MAX_STEPS):
        inspected = obs.get("inspected_artifacts", [])
        flagged = obs.get("flagged_issues", [])
        mitigated = [m.get("issue_code") for m in obs.get("requested_mitigations", [])]

        # Compact observation
        compact_obs = {
            "step": step_num + 1,
            "stage": obs.get("review_stage"),
            "artifacts": list(obs.get("visible_artifacts", {}).keys()),
            "inspected": inspected,
            "flagged": [{"code": i.get("code"), "sev": i.get("severity")} for i in flagged],
            "mitigations": mitigated,
            "risk": obs.get("current_risk"),
            "msg": obs.get("message", "")[-150:],
        }

        # Tell the model what to do next explicitly
        remaining_artifacts = [a for a in compact_obs["artifacts"] if a not in inspected]
        if remaining_artifacts:
            hint = f"\nNext: inspect '{remaining_artifacts[0]}'. Return JSON."
        elif not flagged:
            hint = "\nAll inspected. Flag any violations found, or approve. Return JSON."
        elif not all(i.get("code", "") in mitigated for i in flagged):
            hint = "\nRequest mitigations for flagged issues. Return JSON."
        else:
            hint = "\nSet risk and make final decision (reject/escalate/approve). Return JSON."

        user_msg = f"State:\n{json.dumps(compact_obs)}{hint}"

        action_dict = None
        try:
            raw = client.complete(system_prompt, user_msg)
            action_dict = _extract_json(raw)
        except Exception as e:
            if verbose:
                print(f"  [Step {step_num + 1}] Model error: {e}")

        # ── Stuck-loop detection ──────────────────────────────────
        if action_dict:
            action_key = json.dumps(action_dict, sort_keys=True)
            if action_key == last_action_key:
                repeat_count += 1
                if repeat_count >= 2:
                    if verbose:
                        print(f"  [Step {step_num + 1}] 🔄 Stuck loop detected (same action {repeat_count}x) → fallback")
                    action_dict = None  # Force fallback
                    repeat_count = 0
            else:
                repeat_count = 0
            last_action_key = action_key

        # Fallback planner
        if action_dict is None:
            fallback_count += 1
            action_dict = _plan_actions(obs, inspected, flagged, mitigated)
            last_action_key = json.dumps(action_dict, sort_keys=True)
            if verbose:
                print(f"  [Step {step_num + 1}] ⚠️ Fallback planner (#{fallback_count})")

        action_json = json.dumps(action_dict)

        # Execute
        resp = requests.post(f"{BASE_URL}/step", json={"action": action_json})
        resp.raise_for_status()
        result = resp.json()
        obs = result["observation"]
        reward = result["reward"]
        total_reward += reward

        steps_log.append({
            "step": step_num + 1,
            "action": action_dict,
            "reward": reward,
            "message": obs.get("message", "")[:150],
        })

        if verbose:
            atype = action_dict.get("action_type", "?")
            detail = action_dict.get("target") or action_dict.get("issue_code") or ""
            print(f"  Step {step_num + 1:2d}: {atype:<25} {detail:<20} → {reward:+.2f}")

        if result["done"]:
            break

    # Get grader score
    grader_resp = requests.get(f"{BASE_URL}/grader")
    grader = grader_resp.json()
    scores = grader["scores"]

    # Get final state
    state_resp = requests.get(f"{BASE_URL}/state")
    state = state_resp.json()

    result_data = {
        "task_id": obs.get("task_id"),
        "feature_name": obs.get("feature_name"),
        "feature_summary": obs.get("feature_summary"),
        "total_reward": total_reward,
        "steps": len(steps_log),
        "fallback_steps": fallback_count,
        "grader_score": scores,
        "grade": grader["grade"],
        "final_decision": state.get("final_decision"),
        "flagged_issues": state.get("flagged_issues", []),
        "requested_mitigations": state.get("requested_mitigations", []),
        "final_risk": state.get("current_risk"),
        "reviewer_model": client.model_id,
        "steps_log": steps_log,
    }

    if verbose:
        print(f"\n[Reviewer] ✅ Done in {len(steps_log)} steps ({fallback_count} fallbacks)")
        print(f"[Reviewer] Grader: {scores['overall']:.4f} | Grade: {grader['grade']}")
        print(f"[Reviewer] Decision: {state.get('final_decision')} | "
              f"Flagged: {[i['code'] for i in state.get('flagged_issues', [])]}")

    return result_data


def run_and_save(task_id: int = 1, use_generated: bool = False) -> dict:
    """Run episode + save to memory."""
    result = run_reviewer_episode(task_id=task_id, use_generated=use_generated)

    flagged_codes = [i.get("code", "") for i in result.get("flagged_issues", [])]
    try:
        from app.tasks import load_task
        task = load_task(task_id)
        gt = task["ground_truth_violations"]
        caught = [c for c in gt if c in flagged_codes]
        missed = [c for c in gt if c not in flagged_codes]
        fp = [c for c in flagged_codes if c not in gt]
    except Exception:
        caught, missed, fp = flagged_codes, [], []

    save_episode_result(
        task_id=str(result["task_id"]),
        grader_score=result["grader_score"],
        caught=caught, missed=missed, false_positives=fp,
        final_decision=result["final_decision"],
        expected_decision=None,
    )
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=1)
    parser.add_argument("--use_generated", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("Qwen2.5-0.5B Reviewer Agent (v2)")
    print("=" * 60)

    try:
        requests.get(f"{BASE_URL}/health").raise_for_status()
    except Exception:
        print(f"❌ Server not running at {BASE_URL}")
        print("Run: python3 -m uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    result = run_and_save(task_id=args.task_id, use_generated=args.use_generated)
    print(f"\nFinal score: {result['grader_score']['overall']:.4f} | {result['grade']}")
