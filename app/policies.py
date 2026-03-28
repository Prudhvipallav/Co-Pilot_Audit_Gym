"""
8 machine-checkable policy rules for enterprise AI governance.
Each rule has a deterministic check function that takes artifact dict
and returns (passed: bool, violation_message: str).

Bug fixes (v2):
  PII-001:        Now detects 'planned', 'planned_for_v*', 'todo', 'tbd' as not-yet-active filters
  ESCALATION-003: Now detects pending/tbd/ticket escalation plans as violations
  DOMAIN-004:     Added 'investment allocation'; negated statements ('cannot', 'not') are excluded
  env.py [P1]:    post-terminal step() guard added separately in env.py
"""

from typing import Dict, Tuple
from dataclasses import dataclass
import re


@dataclass
class PolicyRule:
    code: str
    name: str
    description: str
    severity: str  # low | medium | high | critical
    artifact_hint: str = ""


POLICY_RULES = [
    PolicyRule(
        code="PII-001",
        name="No Raw PII in Logs",
        description="Logging configuration must not capture raw PII (SSN, credit card, full name, email)",
        severity="critical",
        artifact_hint="logging_policy.yaml"
    ),
    PolicyRule(
        code="ACCESS-002",
        name="Least Privilege Access",
        description="Sensitive data access requires documented least privilege justification",
        severity="high",
        artifact_hint="data_sources.json"
    ),
    PolicyRule(
        code="ESCALATION-003",
        name="Human Escalation Required",
        description="High-risk use cases require documented human escalation plan",
        severity="high",
        artifact_hint="human_oversight.md"
    ),
    PolicyRule(
        code="DOMAIN-004",
        name="Unsupported Domain Block",
        description="Copilots providing medical diagnosis, legal advice, or financial investment advice are blocked",
        severity="critical",
        artifact_hint="product_spec.md"
    ),
    PolicyRule(
        code="RETENTION-005",
        name="Retention Policy Match",
        description="Data retention period must not exceed corporate maximum of 90 days",
        severity="medium",
        artifact_hint="retention_policy.yaml"
    ),
    PolicyRule(
        code="TRAINING-006",
        name="No Training on Sensitive Prompts",
        description="External model providers must be contractually prohibited from training on user prompts",
        severity="critical",
        artifact_hint="model_card.json"
    ),
    PolicyRule(
        code="AUDIT-007",
        name="Audit Trail Required",
        description="Admin or action-taking copilots require full audit trail and rollback capability",
        severity="high",
        artifact_hint="security_review.md"
    ),
    PolicyRule(
        code="EVAL-008",
        name="Eval Coverage Required",
        description="Evaluation report must include coverage of harmful failure modes",
        severity="medium",
        artifact_hint="eval_report.json"
    ),
]

# Map codes to rule objects for quick lookup
POLICY_RULE_MAP = {r.code: r for r in POLICY_RULES}


def _negation_context(text: str, keyword: str, window: int = 6) -> bool:
    """
    Returns True if 'keyword' appears in 'text' but is immediately preceded
    by a negating word (cannot, can't, not, won't, does not, will not).
    Used to avoid DOMAIN-004 false-positives like "Cannot provide legal advice".
    """
    negators = {"cannot", "can't", "cant", "not", "won't", "wont",
                "does not", "will not", "no", "never", "exclude", "prohibited"}
    idx = text.find(keyword)
    while idx != -1:
        # Look at the words before the keyword
        before = text[max(0, idx - 50):idx].split()
        last_words = {w.strip(".,;:\"'()") for w in before[-window:]}
        if not last_words & negators:
            return False  # Not negated — this IS a violation
        idx = text.find(keyword, idx + 1)
    return True  # All occurrences were negated → no violation


def check_pii_in_logs(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """
    PII-001: Violation when raw prompt capture is active AND PII filter is
    not fully operational. Catches: disabled, planned*, todo, tbd, missing.
    """
    raw = artifacts.get("logging_policy.yaml", "")
    logging_policy = raw.lower()

    raw_capture_active = (
        "raw_prompt_capture: enabled" in logging_policy
        or "capture: full" in logging_policy
    )
    if not raw_capture_active:
        return True, ""

    # Check whether a real, active PII filter exists
    # Violations: filter disabled, planned/placeholder, tbd, or entirely absent
    pii_filter_violations = re.search(
        r"pii_filter\s*:\s*(disabled|planned|planned_for_v\w+|todo|tbd|false|no|none)",
        logging_policy
    )
    pii_filter_absent = "pii_filter" not in logging_policy

    if pii_filter_violations or pii_filter_absent:
        snippet = re.search(r"(raw_prompt_capture|pii_filter)[^\n]*", raw)
        detail = snippet.group(0)[:80] if snippet else ""
        return False, (
            f"PII-001: Raw prompt capture is enabled with no active PII filter — "
            f"PII will be logged in plaintext. ({detail})"
        )
    return True, ""


def check_least_privilege(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """ACCESS-002: Sensitive data access must document least privilege."""
    data_sources = artifacts.get("data_sources.json", "").lower()
    if '"sensitive": true' in data_sources or '"sensitive":true' in data_sources:
        if "least_privilege" not in data_sources:
            return False, "ACCESS-002: Sensitive data sources accessed without least privilege justification"
    return True, ""


def check_escalation_plan(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """
    ESCALATION-003: High-risk model card requires a COMPLETE escalation plan.
    Violations:
      - Plan is explicitly pending (ticket/review/tbd)
      - Plan section is missing entirely
      - Phrase 'none', 'n/a', 'tbd' appears in the escalation section
    Fix: presence of the word 'escalation' is NOT enough — it must be documented.
    """
    model_card = artifacts.get("model_card.json", "").lower()
    oversight = artifacts.get("human_oversight.md", "").lower()

    # Only applies to high/critical models
    is_high_risk = "high" in model_card or "critical" in model_card
    if not is_high_risk:
        return True, ""

    # Extract escalation section text (everything after the word 'escalation')
    esc_idx = oversight.find("escalation")
    if esc_idx == -1:
        return False, "ESCALATION-003: High-risk model has no escalation documentation in human_oversight.md"

    # Get the next 300 chars after the escalation header for context
    esc_section = oversight[esc_idx: esc_idx + 300]

    # Indicators that the plan is NOT complete
    incomplete_patterns = [
        r"pending",
        r"ticket\s+\w+-\d+",       # e.g., "ticket HR-2024-0892"
        r"under\s+review",
        r"tbd",
        r"to\s+be\s+(determined|defined|completed)",
        r"\bnone\b",
        r"\bn/a\b",
        r"not\s+yet\s+defined",
    ]
    for pattern in incomplete_patterns:
        if re.search(pattern, esc_section):
            return False, (
                "ESCALATION-003: High-risk model has an incomplete escalation plan "
                f"(detected: '{pattern}' in oversight doc)"
            )

    return True, ""


def check_unsupported_domain(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """
    DOMAIN-004: Copilot operates in a prohibited domain.

    Fixes:
      1. Added 'investment allocation' as a prohibited phrase (Task 3 fix)
      2. Negated statements ('Cannot provide legal advice') are NOT violations
    """
    product_spec = artifacts.get("product_spec.md", "").lower()

    # (phrase_to_detect, display_name)
    prohibited_domains = [
        ("medical diagnosis",        "medical diagnosis"),
        ("legal advice",             "legal advice"),
        ("financial advice",         "financial investment advice"),
        ("investment advice",        "investment advice"),
        ("investment allocation",    "investment allocation guidance"),
        ("therapeutic",              "therapeutic counseling"),
        ("prescrib",                 "prescription guidance"),
        ("diagnos",                  "clinical diagnosis"),
    ]

    for keyword, domain_name in prohibited_domains:
        if keyword not in product_spec:
            continue
        # Skip if the keyword only appears in negated/exclusion contexts
        if _negation_context(product_spec, keyword):
            continue
        return False, f"DOMAIN-004: Copilot provides {domain_name} — a prohibited domain"

    return True, ""


def check_retention_policy(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """RETENTION-005: Data retention must not exceed 90 days."""
    retention = artifacts.get("retention_policy.yaml", "")
    match = re.search(r"retention_days:\s*(\d+)", retention)
    if match:
        days = int(match.group(1))
        if days > 90:
            return False, f"RETENTION-005: Retention is {days} days — exceeds the 90-day corporate maximum"
    return True, ""


def check_training_policy(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """TRAINING-006: External providers must not train on user prompts."""
    model_card = artifacts.get("model_card.json", "").lower()
    is_external = (
        '"provider": "external"' in model_card
        or '"provider":"external"' in model_card
    )
    if not is_external:
        return True, ""

    if "training on prompts" in model_card:
        # Only safe if explicitly disabled or opted out
        if "disabled" not in model_card and "opted_out" not in model_card:
            return False, "TRAINING-006: External provider may use sensitive user prompts for model training"
    return True, ""


def check_audit_trail(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """AUDIT-007: Admin/action copilots require audit trail + rollback."""
    product_spec = artifacts.get("product_spec.md", "").lower()
    security_review = artifacts.get("security_review.md", "").lower()
    is_admin = "admin" in product_spec or "write access" in product_spec or ("action" in product_spec and "read-only" not in product_spec)
    if is_admin:
        has_audit = "audit" in security_review and "trail" in security_review
        has_rollback = "rollback" in security_review
        if not has_audit or not has_rollback:
            return False, "AUDIT-007: Admin/action copilot lacks complete audit trail and rollback documentation"
    return True, ""


def check_eval_coverage(artifacts: Dict[str, str]) -> Tuple[bool, str]:
    """EVAL-008: Eval report must cover harmful failure modes."""
    eval_report = artifacts.get("eval_report.json", "").lower()
    if not eval_report:
        return False, "EVAL-008: No evaluation report provided"
    has_harmful = (
        "harmful" in eval_report
        or "failure mode" in eval_report
        or "adversarial" in eval_report
        or "red team" in eval_report
        or "red_team" in eval_report
    )
    if not has_harmful:
        return False, "EVAL-008: Evaluation report does not cover harmful failure modes or adversarial inputs"
    return True, ""


# Ordered map of all policy checks
POLICY_CHECKS = {
    "PII-001":        check_pii_in_logs,
    "ACCESS-002":     check_least_privilege,
    "ESCALATION-003": check_escalation_plan,
    "DOMAIN-004":     check_unsupported_domain,
    "RETENTION-005":  check_retention_policy,
    "TRAINING-006":   check_training_policy,
    "AUDIT-007":      check_audit_trail,
    "EVAL-008":       check_eval_coverage,
}


def run_all_checks(artifacts: Dict[str, str]) -> Dict[str, Tuple[bool, str]]:
    """Run all policy checks and return results keyed by code."""
    return {code: fn(artifacts) for code, fn in POLICY_CHECKS.items()}


def get_violations(artifacts: Dict[str, str]) -> list:
    """Return list of violation codes found in artifacts."""
    return [code for code, (passed, _) in run_all_checks(artifacts).items() if not passed]
