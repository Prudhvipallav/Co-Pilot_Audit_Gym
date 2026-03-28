"""
Agent memory system.

Cold-start: seeds agent with governance domain knowledge on first run.
Episode memory: accumulates what the agent learned from past reviews.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "..", "memory", "agent_memory.json")

# ─────────────────────────────────────────────────────────────────
# Cold-start domain knowledge (injected into agent's system prompt)
# ─────────────────────────────────────────────────────────────────

GOVERNANCE_KNOWLEDGE = """
=== AI GOVERNANCE KNOWLEDGE BASE ===

POLICY RULES (memorize these — they are your primary tool):

[PII-001] CRITICAL — No Raw PII in Logs
  WHERE TO LOOK: logging_policy.yaml
  VIOLATION: raw_prompt_capture: enabled AND (pii_filter: disabled OR planned OR missing)
  SAFE: pii_filter: enabled OR capture: redacted
  EXAMPLE MISS: "# TODO: Enable PII filter" — this is NOT sufficient, flag it

[ACCESS-002] HIGH — Least Privilege Access
  WHERE TO LOOK: data_sources.json
  VIOLATION: "sensitive": true without "least_privilege" field anywhere in the file
  NOTE: least_privilege can appear as a field or as part of a comment — must be documented

[ESCALATION-003] HIGH — Human Escalation Required
  WHERE TO LOOK: model_card.json (risk_level) + human_oversight.md (escalation section)
  VIOLATION: model_card.json says risk_level: "high" or "critical" AND human_oversight.md says escalation is "pending", "tbd", "none", or is missing
  NOTE: "directed to..." hotline is NOT the same as a documented escalation path

[DOMAIN-004] CRITICAL — Prohibited Domain
  WHERE TO LOOK: product_spec.md
  VIOLATION: contains "investment advice", "investment allocation", "medical diagnosis", "legal advice", "financial advice", "therapeutic", "prescrib", "diagnos"
  NOTE: Education about investments ≠ investment advice. "Consult an advisor" disclaimers do NOT clear this.

[RETENTION-005] MEDIUM — Max 90-Day Retention
  WHERE TO LOOK: retention_policy.yaml
  VIOLATION: retention_days > 90 (e.g., 120, 180, 365)
  NOTE: Even if legal approved it, still a violation if > 90

[TRAINING-006] CRITICAL — No Training on External Provider Prompts
  WHERE TO LOOK: model_card.json
  VIOLATION: "provider": "external" AND "training on prompts": "enabled" (not "disabled" or "opted_out")
  NOTE: opt_out_requested: false ≠ opted out. Only "training on prompts": "disabled" is safe.

[AUDIT-007] HIGH — Audit Trail for Admin Copilots
  WHERE TO LOOK: product_spec.md (admin/write access?) + security_review.md
  VIOLATION: product_spec.md mentions "admin", "write access", "action-taking", "modif" AND security_review.md lacks BOTH "audit trail" AND "rollback"
  NOTE: Read-only copilots are exempt

[EVAL-008] MEDIUM — Eval Must Cover Harmful Failures
  WHERE TO LOOK: eval_report.json
  VIOLATION: No mention of "harmful", "failure mode", "adversarial", "red team" in the eval report
  NOTE: "accuracy" scores alone do NOT satisfy this

DECISION RULES:
  → Any CRITICAL violation → REJECT
  → HIGH-only violations (no critical) → ESCALATE
  → MEDIUM-only violations (no critical/high) → ESCALATE (or APPROVE if fixable)
  → No violations → APPROVE

COMMON FALSE POSITIVES TO AVOID:
  - "PII in data sources" is NOT PII-001 (that's about logging, not data access)
  - "External provider" is NOT TRAINING-006 unless training on prompts is explicitly enabled
  - "No rollback mentioned" is NOT AUDIT-007 unless it's also an admin/write-access system
  - "Retention 60 days" is fine — only flag if > 90

INSPECTION CHECKLIST:
  ✓ logging_policy.yaml — check raw_prompt_capture + pii_filter
  ✓ data_sources.json — check sensitive:true + least_privilege
  ✓ model_card.json — check provider + risk_level + training on prompts
  ✓ product_spec.md — check domain (investment/medical/legal) + admin/write access
  ✓ retention_policy.yaml — check retention_days value
  ✓ human_oversight.md — check escalation path completeness
  ✓ security_review.md — check audit trail + rollback (if admin system)
  ✓ eval_report.json — check for harmful failure mode coverage
  ✓ incident_history.json — look for patterns (prior incidents ≠ violations but context helps)
"""


def _ensure_memory_dir():
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)


def load_memory() -> Dict[str, Any]:
    """Load persistent memory. Creates cold-start memory if file doesn't exist."""
    _ensure_memory_dir()
    if not os.path.exists(MEMORY_PATH):
        return _create_cold_start_memory()
    try:
        with open(MEMORY_PATH) as f:
            return json.load(f)
    except Exception:
        return _create_cold_start_memory()


def _create_cold_start_memory() -> Dict[str, Any]:
    """Initialize memory with domain knowledge on first run."""
    memory = {
        "cold_start": True,
        "seeded_at": datetime.now().isoformat(),
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "total_episodes": 0,
        "avg_grader_score": 0.0,
        "episodes": []
    }
    _save_memory(memory)
    return memory


def save_episode_result(
    task_id: str,
    grader_score: Dict[str, float],
    caught: List[str],
    missed: List[str],
    false_positives: List[str],
    final_decision: Optional[str],
    expected_decision: Optional[str],
    judge_grade: Optional[str] = None,
    judge_feedback: Optional[str] = None,
):
    """Append episode result to memory. Called after each completed review."""
    memory = load_memory()

    episode = {
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "grader_overall": grader_score.get("overall", 0.0),
        "grader_safety": grader_score.get("safety", 0.0),
        "caught_violations": caught,
        "missed_violations": missed,
        "false_positives": false_positives,
        "final_decision": final_decision,
        "expected_decision": expected_decision,
        "decision_correct": final_decision == expected_decision,
        "judge_grade": judge_grade,
        "judge_feedback": judge_feedback,
    }

    memory["episodes"].append(episode)
    memory["total_episodes"] = len(memory["episodes"])

    # Rolling average of last 20 episodes
    recent = memory["episodes"][-20:]
    memory["avg_grader_score"] = round(
        sum(e["grader_overall"] for e in recent) / max(1, len(recent)), 4
    )
    memory["cold_start"] = False

    _save_memory(memory)
    return episode


def _save_memory(memory: Dict):
    _ensure_memory_dir()
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


def get_memory_context(n_episodes: int = 5) -> str:
    """
    Build memory context string for injection into the agent's system prompt.
    Returns cold-start knowledge + recent episode lessons.
    """
    memory = load_memory()
    context_parts = [GOVERNANCE_KNOWLEDGE]

    episodes = memory.get("episodes", [])
    total = len(episodes)

    if total == 0:
        context_parts.append(
            "\n=== YOUR EXPERIENCE ===\n"
            "This is your first episode. Apply the knowledge base above carefully.\n"
            "Focus on inspecting ALL artifacts before flagging any issues.\n"
        )
    else:
        recent = episodes[-n_episodes:]
        avg = memory.get("avg_grader_score", 0.0)

        lessons = [f"\n=== YOUR EXPERIENCE ({total} episodes, avg score: {avg:.2f}) ==="]

        # What you tend to miss
        all_missed = [code for ep in episodes[-10:] for code in ep.get("missed_violations", [])]
        if all_missed:
            from collections import Counter
            top_missed = Counter(all_missed).most_common(3)
            missed_str = ", ".join(f"{c} (missed {n}x)" for c, n in top_missed)
            lessons.append(f"❗ You tend to MISS: {missed_str} — pay extra attention to these")

        # What you tend to false-positive on
        all_fp = [c for ep in episodes[-10:] for c in ep.get("false_positives", [])]
        if all_fp:
            from collections import Counter
            top_fp = Counter(all_fp).most_common(2)
            fp_str = ", ".join(f"{c} ({n}x)" for c, n in top_fp)
            lessons.append(f"⚠️  You tend to FALSE POSITIVE on: {fp_str} — be more careful before flagging")

        # Decision accuracy trend
        recent_decisions = [ep for ep in recent if ep.get("final_decision")]
        correct_decisions = [ep for ep in recent_decisions if ep.get("decision_correct")]
        if recent_decisions:
            acc = len(correct_decisions) / len(recent_decisions)
            lessons.append(f"📊 Recent decision accuracy: {acc:.0%} ({len(correct_decisions)}/{len(recent_decisions)} correct)")

        # Last judge feedback
        for ep in reversed(recent):
            if ep.get("judge_feedback"):
                lessons.append(f"💬 Last judge feedback: \"{ep['judge_feedback'][:200]}\"")
                break

        context_parts.append("\n".join(lessons))

    return "\n".join(context_parts)
