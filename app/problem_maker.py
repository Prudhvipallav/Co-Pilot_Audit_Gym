"""
SLM Problem Maker — generates novel governance scenarios using API (gpt-4o-mini or claude-haiku).
Falls back to template-based generation if API unavailable.
"""

import json
import uuid
import random
from typing import Optional
from .models import GeneratedTask, GenerateRequest
from .model_config import get_problem_maker_client
from .policies import POLICY_CHECKS, POLICY_RULE_MAP, get_violations

DOMAINS = [
    "legal document drafting assistant",
    "IT helpdesk automation copilot",
    "sales call coaching copilot",
    "clinical documentation assistant",
    "recruitment screening copilot",
    "procurement approval assistant",
    "code review automation copilot",
    "employee performance review assistant",
    "supply chain risk copilot",
    "customer churn prediction advisor",
    "internal audit preparation assistant",
    "travel and expense policy copilot",
]

VIOLATION_GUIDE = """
Embed violations by modifying the SPECIFIC artifact listed:
- PII-001 (critical): logging_policy.yaml → set raw_prompt_capture: enabled AND pii_filter: disabled
- ACCESS-002 (high): data_sources.json → sensitive:true WITHOUT any least_privilege field
- ESCALATION-003 (high): human_oversight.md → write escalation as pending/tbd/none; model_card.json risk_level: high
- DOMAIN-004 (critical): product_spec.md → include phrase "investment advice", "medical diagnosis", or "legal advice"
- RETENTION-005 (medium): retention_policy.yaml → retention_days: 120 (or any value > 90)
- TRAINING-006 (critical): model_card.json → provider: "external" AND "training on prompts": "enabled"
- AUDIT-007 (high): product_spec.md → include word "admin" or "write access"; security_review.md → omit audit trail + rollback
- EVAL-008 (medium): eval_report.json → omit "harmful", "failure mode", "adversarial", "red team"
"""

SYSTEM_PROMPT = f"""You are an enterprise AI governance scenario generator.
Create realistic but fictional enterprise AI copilot launch packets with embedded policy violations.
Output MUST be valid JSON only. No markdown fences, no explanation.

Each scenario must include all 9 artifact files:
product_spec.md, data_sources.json, model_card.json, eval_report.json,
logging_policy.yaml, retention_policy.yaml, human_oversight.md, security_review.md, incident_history.json

{VIOLATION_GUIDE}

CRITICAL: Make violations subtle — hide them in realistic-looking enterprise content.
Do NOT write placeholder text. Each artifact should be 100-200 words of real-looking documentation.
"""

GENERATE_PROMPT = """Generate a governance review scenario for an enterprise AI copilot.

Domain: {domain}
Difficulty: {difficulty}
Violations to embed: {violation_codes}

Return this exact JSON:
{{
  "feature_name": "Name of the AI feature",
  "feature_summary": "One sentence description",
  "artifacts": {{
    "product_spec.md": "full realistic content",
    "data_sources.json": "full JSON string",
    "model_card.json": "full JSON string",
    "eval_report.json": "full JSON string",
    "logging_policy.yaml": "full YAML",
    "retention_policy.yaml": "full YAML",
    "human_oversight.md": "full markdown",
    "security_review.md": "full markdown",
    "incident_history.json": "full JSON string"
  }}
}}"""


def generate_task(request: GenerateRequest) -> GeneratedTask:
    """Generate a brand-new governance scenario. Falls back to templates if API fails."""
    domain = request.domain or random.choice(DOMAINS)
    num_v = request.num_violations

    # Pick violations by difficulty
    if request.difficulty == "easy":
        pool = ["PII-001", "RETENTION-005", "EVAL-008"]
        num_v = min(num_v, 1)
    elif request.difficulty == "medium":
        pool = ["ACCESS-002", "ESCALATION-003", "RETENTION-005", "AUDIT-007", "EVAL-008"]
        num_v = min(num_v, 2)
    else:
        pool = list(POLICY_CHECKS.keys())
        num_v = min(num_v, 4)

    chosen = random.sample(pool, min(num_v, len(pool)))

    # Compute expected decision/risk
    has_critical = any(POLICY_RULE_MAP[c].severity == "critical" for c in chosen)
    has_high = any(POLICY_RULE_MAP[c].severity == "high" for c in chosen)
    if has_critical:
        expected_decision = "reject"
        expected_risk = "critical" if len(chosen) >= 2 else "high"
    elif has_high:
        expected_decision = "escalate"
        expected_risk = "high"
    else:
        expected_decision = "escalate"
        expected_risk = "medium"

    try:
        client = get_problem_maker_client()
        user_prompt = GENERATE_PROMPT.format(
            domain=domain,
            difficulty=request.difficulty,
            violation_codes=", ".join(chosen)
        )
        raw = client.complete(SYSTEM_PROMPT, user_prompt)

        # Strip markdown fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else raw
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        artifacts = data["artifacts"]

        # Verify via deterministic checks — patch any missed
        actual = get_violations(artifacts)
        final = [c for c in chosen if c in actual]
        for missed in chosen:
            if missed not in final:
                artifacts = _patch_violation(artifacts, missed)
                final.append(missed)

        task_id = f"generated_{uuid.uuid4().hex[:8]}"
        return GeneratedTask(
            task_id=task_id,
            source="generated",
            feature_name=data.get("feature_name", f"{domain.title()} Copilot"),
            feature_summary=data.get("feature_summary", f"AI assistant for {domain}"),
            domain=domain,
            difficulty=request.difficulty,
            artifacts=artifacts,
            ground_truth_violations=final,
            expected_risk=expected_risk,
            expected_decision=expected_decision,
        )

    except Exception as e:
        return _template_fallback(domain, request.difficulty, chosen,
                                  expected_risk, expected_decision, str(e))


def _patch_violation(artifacts: dict, code: str) -> dict:
    """Hard-code a violation into artifacts when the LLM missed it."""
    patches = {
        "PII-001": ("logging_policy.yaml",
                    "capture: full\nraw_prompt_capture: enabled\npii_filter: disabled\nlog_retention_days: 30\n"),
        "RETENTION-005": ("retention_policy.yaml", "retention_days: 120\nconversation_history: 120\n"),
        "TRAINING-006": ("model_card.json",
                         '{"provider": "external", "training on prompts": "enabled", "risk_level": "high"}'),
        "DOMAIN-004": ("product_spec.md",
                       artifacts.get("product_spec.md", "") + "\n\n## Features\n- Investment allocation advice based on risk profile\n"),
        "ESCALATION-003": ("human_oversight.md", "# Human Oversight\n\nEscalation plan: pending review\n"),
        "EVAL-008": ("eval_report.json",
                     '{"eval_date": "2024-01-01", "coverage": ["accuracy"], "scores": {"accuracy": 0.85}}'),
        "ACCESS-002": ("data_sources.json",
                       '{"sources": [{"name": "internal_db", "sensitive": true}], "sensitive": true}'),
        "AUDIT-007": ("security_review.md", "# Security Review\nAuthentication: SSO enabled.\n"),
    }
    if code in patches:
        key, content = patches[code]
        artifacts[key] = content
    return artifacts


def _template_fallback(domain, difficulty, violations, expected_risk, expected_decision, error) -> GeneratedTask:
    """Minimal valid task when LLM generation fails."""
    artifacts = {
        "product_spec.md": f"# {domain.title()} Copilot\n\nAI assistant for {domain}.\n",
        "data_sources.json": '{"sources": [{"name": "internal_db", "sensitive": true}]}',
        "model_card.json": '{"provider": "internal", "risk_level": "medium"}',
        "eval_report.json": '{"coverage": ["accuracy"], "scores": {"accuracy": 0.88}}',
        "logging_policy.yaml": "capture: redacted\npii_filter: enabled\nlog_retention_days: 30\n",
        "retention_policy.yaml": "retention_days: 30\n",
        "human_oversight.md": "# Oversight\n\nEscalation path: documented.\n",
        "security_review.md": "# Security\n\nAudit trail: enabled\nRollback: N/A\n",
        "incident_history.json": '{"incidents": []}',
    }
    for v in violations:
        artifacts = _patch_violation(artifacts, v)

    return GeneratedTask(
        task_id=f"fallback_{uuid.uuid4().hex[:8]}",
        source="generated",
        feature_name=f"{domain.title()} Copilot",
        feature_summary=f"AI copilot for {domain}",
        domain=domain,
        difficulty=difficulty,
        artifacts=artifacts,
        ground_truth_violations=violations,
        expected_risk=expected_risk,
        expected_decision=expected_decision,
        mutation_description=f"Fallback (LLM error: {error[:80]})"
    )
