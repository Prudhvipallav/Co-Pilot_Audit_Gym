"""
Utility functions for artifact generation and validation.
"""

from typing import Dict


def get_artifact_descriptions() -> Dict[str, str]:
    """Human-readable descriptions of what each artifact type contains."""
    return {
        "product_spec.md": "Product description, capabilities, data access scope, user base",
        "data_sources.json": "Data sources accessed, sensitivity levels, PII types, access controls",
        "model_card.json": "Model provider, risk level, training data, known failure modes",
        "eval_report.json": "Evaluation methodology, scores, harmful failure mode coverage",
        "logging_policy.yaml": "What gets logged, PII filtering, retention for logs",
        "retention_policy.yaml": "How long conversation/data is retained",
        "human_oversight.md": "Escalation paths, monitoring, human override mechanisms",
        "security_review.md": "Auth controls, encryption, audit trail, pen test results",
        "incident_history.json": "Past incidents, severity, and resolutions",
    }


def get_policy_reference() -> Dict[str, Dict]:
    """Reference guide for the agent: which artifacts to check for each policy."""
    return {
        "PII-001": {
            "artifact": "logging_policy.yaml",
            "look_for": "raw_prompt_capture: enabled + pii_filter: disabled",
            "severity": "critical"
        },
        "ACCESS-002": {
            "artifact": "data_sources.json",
            "look_for": "sensitive: true without least_privilege: true",
            "severity": "high"
        },
        "ESCALATION-003": {
            "artifact": "human_oversight.md",
            "look_for": "missing escalation path when model_card says high risk",
            "severity": "high"
        },
        "DOMAIN-004": {
            "artifact": "product_spec.md",
            "look_for": "investment advice, medical diagnosis, legal advice, therapeutic",
            "severity": "critical"
        },
        "RETENTION-005": {
            "artifact": "retention_policy.yaml",
            "look_for": "retention_days > 90",
            "severity": "medium"
        },
        "TRAINING-006": {
            "artifact": "model_card.json",
            "look_for": "provider: external + training on prompts: enabled",
            "severity": "critical"
        },
        "AUDIT-007": {
            "artifact": "security_review.md",
            "look_for": "admin/action copilot without audit trail + rollback",
            "severity": "high"
        },
        "EVAL-008": {
            "artifact": "eval_report.json",
            "look_for": "missing harmful_failure_modes or adversarial testing",
            "severity": "medium"
        },
    }
