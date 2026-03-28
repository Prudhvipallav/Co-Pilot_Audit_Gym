"""
Adversarial Problem Maker (v3) — weakness-targeted scenario generation.

Unlike the random problem_maker.py, this uses Claude API to craft scenarios
that specifically test the agent's tracked weak points. Selects violations
proportionally to their miss_rate in the weakness map.

Falls back to random weighted selection if Claude unavailable.
"""

import json
import random
import uuid
from typing import List, Optional, Dict

from .model_config import get_problem_maker_client
from .models import GeneratedTask, GenerateRequest
from memory.weakness_map import get_violation_weights, get_weakness_summary


# ─── Domain pool ────────────────────────────────────────────────────────────

DOMAINS = [
    ("fintech",     "Financial Services"),
    ("healthcare",  "Healthcare & Life Sciences"),
    ("hr",          "Human Resources"),
    ("legal",       "Legal & Compliance"),
    ("retail",      "Retail & E-commerce"),
    ("education",   "Education Technology"),
    ("insurance",   "Insurance"),
    ("government",  "Government & Public Sector"),
    ("telecom",     "Telecommunications"),
    ("manufacturing", "Manufacturing & Supply Chain"),
]

# Policy violation → allowed domains (avoid impossible combinations)
VIOLATION_DOMAIN_AFFINITY = {
    "DOMAIN-004":     ["fintech", "healthcare", "legal"],
    "AUDIT-007":      ["fintech", "legal", "government", "hr"],
    "TRAINING-006":   ["fintech", "healthcare", "retail", "telecom"],
    "PII-001":        ["healthcare", "fintech", "hr", "retail", "government"],
    "ESCALATION-003": ["healthcare", "fintech", "legal", "government"],
    "RETENTION-005":  ["healthcare", "fintech", "hr", "legal"],
    "ACCESS-002":     ["healthcare", "fintech", "hr", "government"],
    "EVAL-008":       ["healthcare", "education", "government"],
}

ADVERSARIAL_SYSTEM = """You are an expert AI governance scenario designer for a compliance training platform.

Your job: write a REALISTIC enterprise AI copilot launch packet that contains specific, subtle policy violations.
The packet must look like a genuine product ready for launch — most of the content should be compliant.
Violations should be embedded naturally, not obviously flagged.

The packet is a JSON object with files as keys and their text content as values.
Each file should be realistic and detailed (100-300 words each).

Tone: corporate technical documentation. No placeholder text."""

ADVERSARIAL_PROMPT = """Design a governance review scenario for a {domain} AI copilot that contains EXACTLY these violations:
{violations_with_rationale}

Requirements:
- Feature name: creative, industry-specific (e.g. "HealthAssist RN Copilot", not "Finance Bot")
- Feature summary: 2-sentence product description
- Each violation should be subtle — embedded inside otherwise-compliant artifact text
- Include ALL of these files: product_spec.md, data_sources.json, model_card.json, eval_report.json, logging_policy.yaml, retention_policy.yaml, human_oversight.md, security_review.md, incident_history.json

Respond with ONLY a JSON object with this exact structure:
{{
  "feature_name": "...",
  "feature_summary": "...",
  "artifacts": {{
    "product_spec.md": "...",
    "data_sources.json": "...",
    "model_card.json": "...",
    "eval_report.json": "...",
    "logging_policy.yaml": "...",
    "retention_policy.yaml": "...",
    "human_oversight.md": "...",
    "security_review.md": "...",
    "incident_history.json": "..."
  }}
}}"""

# How each violation should be embedded (guidance for Claude)
VIOLATION_EMBEDDING = {
    "PII-001":        "Set raw_prompt_capture: enabled and pii_filter: planned_for_v2 in logging_policy.yaml",
    "ACCESS-002":     "Add a sensitive:true data source in data_sources.json without any least_privilege field",
    "ESCALATION-003": "In model_card.json set risk_level: high, in human_oversight.md write that the escalation plan is pending ticket review (e.g. ticket #OPS-2024-1142)",
    "DOMAIN-004":     "In product_spec.md include a feature that provides personalized investment allocation suggestions or medical diagnosis",
    "RETENTION-005":  "Set retention_days: 180 in retention_policy.yaml",
    "TRAINING-006":   "In model_card.json set provider: external and training on prompts: enabled (NOT disabled)",
    "AUDIT-007":      "Make product_spec.md describe admin write access features, but do NOT include audit trail or rollback in security_review.md",
    "EVAL-008":       "In eval_report.json include accuracy metrics but NO mention of harmful, failure mode, adversarial, or red team",
}


def _select_violations_adversarially(
    num_violations: int,
    forced_domain: Optional[str] = None,
) -> tuple[List[str], str]:
    """
    Select violations weighted by miss rate from the weakness map.
    Returns (violation_codes, domain_key).
    """
    weights = get_violation_weights()
    codes = list(weights.keys())
    miss_rates = [weights[c] for c in codes]

    # Weighted sampling without replacement
    selected = []
    remaining_codes = list(codes)
    remaining_rates = list(miss_rates)

    for _ in range(min(num_violations, len(remaining_codes))):
        total = sum(remaining_rates)
        probs = [r / total for r in remaining_rates]
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                selected.append(remaining_codes[i])
                remaining_codes.pop(i)
                remaining_rates.pop(i)
                break

    # Pick domain with highest affinity to selected violations
    if forced_domain:
        domain_key = forced_domain
    else:
        domain_scores: Dict[str, float] = {}
        for d_key, _ in DOMAINS:
            score = 0.0
            for code in selected:
                if d_key in VIOLATION_DOMAIN_AFFINITY.get(code, []):
                    score += weights[code]
            domain_scores[d_key] = score
        domain_key = max(domain_scores, key=domain_scores.get) if domain_scores else random.choice(DOMAINS)[0]

    return selected, domain_key


def generate_adversarial_task(
    num_violations: int = 2,
    forced_domain: Optional[str] = None,
    difficulty: str = "medium",
) -> GeneratedTask:
    """
    Generate a task biased toward the agent's tracked weaknesses.
    Primary: Claude API with embedded violation guidance.
    Fallback: template-based with correct artifact text.
    """
    selected_violations, domain_key = _select_violations_adversarially(
        num_violations, forced_domain
    )
    domain_name = dict(DOMAINS).get(domain_key, "Enterprise")

    # Build violation guidance for the prompt
    violations_text = "\n".join(
        f"  {i+1}. {code}: {VIOLATION_EMBEDDING[code]}"
        for i, code in enumerate(selected_violations)
    )

    # Expected decision
    from .policies import POLICY_RULE_MAP
    severities = [POLICY_RULE_MAP[c].severity for c in selected_violations if c in POLICY_RULE_MAP]
    if "critical" in severities:
        expected_decision, expected_risk = "reject", "critical"
    elif "high" in severities:
        expected_decision, expected_risk = "escalate", "high"
    else:
        expected_decision, expected_risk = "escalate", "medium"

    # Try Claude/API generation
    try:
        client = get_problem_maker_client()
        user_prompt = ADVERSARIAL_PROMPT.format(
            domain=domain_name,
            violations_with_rationale=violations_text,
        )
        raw = client.complete(ADVERSARIAL_SYSTEM, user_prompt)

        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:].strip()

        data = json.loads(raw)
        artifacts = data["artifacts"]

        return GeneratedTask(
            task_id=f"adv_{domain_key}_{str(uuid.uuid4())[:6]}",
            source="generated",
            feature_name=data["feature_name"],
            feature_summary=data["feature_summary"],
            domain=domain_name,
            difficulty=difficulty,
            artifacts=artifacts,
            ground_truth_violations=selected_violations,
            expected_risk=expected_risk,
            expected_decision=expected_decision,
        )

    except Exception as e:
        return _fallback_adversarial_task(selected_violations, domain_key, domain_name,
                                          expected_risk, expected_decision, difficulty, str(e))


def _fallback_adversarial_task(
    violations: List[str],
    domain_key: str,
    domain_name: str,
    expected_risk: str,
    expected_decision: str,
    difficulty: str,
    error: str,
) -> GeneratedTask:
    """
    Deterministic fallback: builds artifact text that exactly triggers
    the selected violation codes so the checker always detects them.
    """
    artifacts = {
        "product_spec.md": _build_product_spec(violations, domain_name),
        "data_sources.json": _build_data_sources(violations),
        "model_card.json": _build_model_card(violations),
        "eval_report.json": _build_eval_report(violations),
        "logging_policy.yaml": _build_logging_policy(violations),
        "retention_policy.yaml": _build_retention_policy(violations),
        "human_oversight.md": _build_oversight(violations),
        "security_review.md": _build_security_review(violations, domain_name),
        "incident_history.json": '{"incidents": [], "last_reviewed": "2025-01-01"}',
    }

    codes_str = "_".join(v.split("-")[0].lower() for v in violations)
    return GeneratedTask(
        task_id=f"adv_fallback_{domain_key}_{codes_str}",
        source="generated",
        feature_name=f"{domain_name.split()[0]}Assist AI Copilot",
        feature_summary=f"Enterprise AI copilot for {domain_name.lower()} workflows. "
                        f"Adversarially generated targeting: {', '.join(violations)}.",
        domain=domain_name,
        difficulty=difficulty,
        artifacts=artifacts,
        ground_truth_violations=violations,
        expected_risk=expected_risk,
        expected_decision=expected_decision,
    )


# ─── Artifact builders (deterministic, checker-aligned) ──────────────────────

def _build_product_spec(violations: List[str], domain: str) -> str:
    extra = ""
    if "DOMAIN-004" in violations:
        extra += "\n## Core Features\n- Personalized investment allocation suggestions based on user risk profile\n- Portfolio rebalancing recommendations\n"
    if "AUDIT-007" in violations:
        extra += "\n## Access Model\n- Full write access to HR records and employee profiles\n- Admin panel for bulk data modifications\n"
    return f"""# {domain} AI Copilot — Product Specification v1.2

## Overview
Enterprise AI assistant for {domain.lower()} workflows. Designed to reduce
manual effort by 40% and improve decision accuracy.

## Target Users
Internal teams and customer-facing staff.
{extra}
## Compliance
All outputs include appropriate disclaimers. Product undergoes quarterly review.
Data residency: EU and US regions.
"""


def _build_data_sources(violations: List[str]) -> str:
    sources = [
        '{"name": "crm_data", "type": "database", "sensitive": false}',
        '{"name": "knowledge_base", "type": "vector_store", "sensitive": false}',
    ]
    if "ACCESS-002" in violations:
        sources.append('{"name": "employee_pii_store", "type": "database", "sensitive": true}')
    return '{"data_sources": [' + ", ".join(sources) + ']}'


def _build_model_card(violations: List[str]) -> str:
    provider = '"external"' if "TRAINING-006" in violations else '"internal"'
    risk = '"high"' if "ESCALATION-003" in violations else '"medium"'
    training_line = ""
    if "TRAINING-006" in violations:
        training_line = '\n  "training on prompts": "enabled",'
    return f"""{{
  "model_id": "enterprise-copilot-v2",
  "provider": {provider},
  "risk_level": {risk},{training_line}
  "modalities": ["text"],
  "last_evaluated": "2025-11-15",
  "fine_tuned": false
}}"""


def _build_eval_report(violations: List[str]) -> str:
    if "EVAL-008" in violations:
        return '{"metrics": {"accuracy": 0.91, "f1": 0.88}, "test_size": 2000, "date": "2025-11-01"}'
    return '{"metrics": {"accuracy": 0.91, "harmful_failure_mode_coverage": true, "adversarial_test_pass": 0.89}}'


def _build_logging_policy(violations: List[str]) -> str:
    if "PII-001" in violations:
        return "raw_prompt_capture: enabled\npii_filter: planned_for_v2\nretention: 30_days\nlog_level: debug"
    return "raw_prompt_capture: disabled\npii_filter: enabled\nlog_level: info"


def _build_retention_policy(violations: List[str]) -> str:
    days = 180 if "RETENTION-005" in violations else 30
    return f"retention_days: {days}\nscope: all_conversation_logs\napproved_by: legal_2025"


def _build_oversight(violations: List[str]) -> str:
    if "ESCALATION-003" in violations:
        return """# Human Oversight Policy

## Escalation
Escalation plan is currently pending review (ticket #OPS-2024-1142).
Expected completion: Q2 2025.

## Monitoring
Weekly model performance review by ML team.
"""
    return """# Human Oversight Policy

## Escalation Path
Level 1: Frontline supervisor (< 4h SLA)
Level 2: Compliance officer (< 24h SLA)
Level 3: Executive sponsor

## Monitoring
Weekly model performance review by ML team.
"""


def _build_security_review(violations: List[str], domain: str) -> str:
    if "AUDIT-007" in violations:
        return f"""# Security Review — {domain} AI Copilot

## Access Controls
Role-based access enforced via SSO.
Write access limited to authorized admin users.

## Penetration Testing
Completed Q3 2025. No critical findings.

## Data Encryption
AES-256 at rest, TLS 1.3 in transit.
"""
    return f"""# Security Review — {domain} AI Copilot

## Audit Trail
Full audit trail enabled for all admin actions.
Rollback capability: available for bulk modifications (30-day window).

## Access Controls
Role-based access enforced via SSO.

## Data Encryption
AES-256 at rest, TLS 1.3 in transit.
"""
