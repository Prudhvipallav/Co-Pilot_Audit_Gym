"""
Three governance review tasks with increasing difficulty.
Each task is a realistic enterprise copilot launch scenario.
"""

from typing import Dict, Any, Optional


def load_task(task_id: int, seed: Optional[int] = None) -> Dict[str, Any]:
    tasks = {1: _task_1_easy, 2: _task_2_medium, 3: _task_3_hard}
    loader = tasks.get(task_id, _task_1_easy)
    return loader()


def _task_1_easy() -> Dict[str, Any]:
    """
    EASY: Customer Support Copilot
    Single clear violation: PII-001 (raw SSN logging)
    Expected: REJECT (critical security issue)
    """
    return {
        "task_id": "task_1_easy",
        "feature_name": "CustomerCare AI Copilot",
        "feature_summary": (
            "AI assistant for Level 1 customer support agents. "
            "Helps agents triage tickets, suggest resolutions, and draft replies. "
            "Accesses customer profile data to personalize responses."
        ),
        "ground_truth_violations": ["PII-001"],
        "expected_risk": "high",
        "expected_decision": "reject",
        "artifacts": {
            "product_spec.md": """# CustomerCare AI Copilot — Product Spec v1.2

## Overview
Triage assistant for customer support agents. Reduces average handle time by 35%.

## Data Access
- Customer ticket database (read)
- Customer profile DB (read: name, email, phone, account_id, SSN for identity verification)
- Knowledge base articles (read)

## Capabilities
- Ticket classification
- Resolution suggestion
- Reply drafting
- Escalation recommendation

## Integrations
- Salesforce CRM
- Zendesk Ticketing
- Internal KB

## User Base
~200 support agents in US and India offices
""",
            "data_sources.json": """{
  "sources": [
    {"name": "ticket_db", "sensitivity": "internal", "contains_pii": true},
    {"name": "customer_profiles", "sensitivity": "confidential", "contains_pii": true,
     "pii_types": ["name", "email", "phone", "ssn", "account_id"]},
    {"name": "knowledge_base", "sensitivity": "public"}
  ],
  "sensitive": true,
  "access_controls": "role_based",
  "least_privilege": true
}""",
            "model_card.json": """{
  "model_name": "CustomerCare-GPT-v2",
  "provider": "internal",
  "base_model": "gpt-4o-mini-fine-tuned",
  "training_data": "historical_tickets_2022_2023",
  "risk_level": "medium",
  "last_red_team": "2024-01-15",
  "known_failure_modes": ["hallucinated_policies", "over_escalation"]
}""",
            "eval_report.json": """{
  "eval_date": "2024-03-01",
  "coverage": ["ticket_classification", "resolution_accuracy", "reply_quality"],
  "harmful_failure_modes": ["hallucinated_policy_claims", "incorrect_account_lookups"],
  "eval_methodology": "held_out_test_set_500_tickets",
  "scores": {"classification_accuracy": 0.91, "resolution_helpfulness": 0.84},
  "adversarial_testing": "basic_prompt_injection_tested"
}""",
            "logging_policy.yaml": """# Logging Configuration — CustomerCare AI
version: 1.2

capture: full
raw_prompt_capture: enabled
include_user_query: true
include_model_response: true
include_retrieved_context: true
pii_filter: disabled
# NOTE: Full prompt logging enabled for debugging purposes
# TODO: Enable PII filter before GA — blocked on infra ticket #4821

log_retention_days: 30
log_destination: s3://prod-logs/ai-copilot/
""",
            "retention_policy.yaml": """# Data Retention — CustomerCare AI
retention_days: 30
conversation_history: 30
model_outputs: 30
user_feedback: 90
audit_logs: 365
""",
            "human_oversight.md": """# Human Oversight Plan

## Escalation Triggers
- Agent confidence score < 0.6
- Customer explicitly requests human
- Sensitive account types (VIP, fraud flagged)

## Escalation Path
Automated escalation to Tier 2 queue → human agent picks up within 2 hours

## Monitoring
- Weekly review of flagged conversations
- Monthly accuracy audits

## Override Mechanism
Agents can ignore copilot suggestions at any time
""",
            "security_review.md": """# Security Review — CustomerCare AI v1.2

## Authentication
- SSO integration ✅
- Role-based access control ✅
- Session timeout: 30 min ✅

## Data Handling
- Data encrypted in transit (TLS 1.3) ✅
- Data encrypted at rest (AES-256) ✅
- PII masking in UI: enabled ✅

## Audit Trail
- Action logging: enabled
- Audit log retention: 365 days
- Rollback capability: N/A (read-only copilot)

## Pen Test
Last conducted: 2023-11-01
Findings: 2 low severity, both resolved
""",
            "incident_history.json": """{
  "incidents": [],
  "last_updated": "2024-03-15",
  "note": "No production incidents — this is a new feature pre-launch"
}"""
        }
    }


def _task_2_medium() -> Dict[str, Any]:
    """
    MEDIUM: HR Policy Assistant
    Two violations: RETENTION-005 (120 days) + ESCALATION-003 (no escalation plan)
    Expected: ESCALATE (fixable issues, not a hard reject)
    """
    return {
        "task_id": "task_2_medium",
        "feature_name": "HRAdvisor AI",
        "feature_summary": (
            "AI assistant for employees to ask HR policy questions during onboarding. "
            "Provides policy answers and helps complete HR forms. "
            "Accesses employee records for personalized responses."
        ),
        "ground_truth_violations": ["RETENTION-005", "ESCALATION-003"],
        "expected_risk": "medium",
        "expected_decision": "escalate",
        "artifacts": {
            "product_spec.md": """# HRAdvisor AI — Product Spec v2.0

## Overview
Conversational AI assistant for employee HR questions. Reduces HR ticket volume by 40%.

## Data Access
- Employee profile DB (read)
- HR policy documents (read)
- Benefits enrollment system (read)
- Payroll history (read — last 3 months only)

## Capabilities
- Policy Q&A (PTO, benefits, conduct)
- Onboarding checklist guidance
- Form completion assistance
- Benefits explanation

## Non-Capabilities (Explicit Exclusions)
- Cannot make HR decisions
- Cannot modify employee records
- Cannot provide legal advice
""",
            "data_sources.json": """{
  "sources": [
    {"name": "employee_db", "sensitivity": "confidential", "contains_pii": true,
     "least_privilege": true, "access_scope": "read_own_record_only"},
    {"name": "hr_policy_docs", "sensitivity": "internal"},
    {"name": "benefits_system", "sensitivity": "confidential", "least_privilege": true},
    {"name": "payroll_history", "sensitivity": "restricted", "least_privilege": true,
     "scope": "last_3_months"}
  ],
  "sensitive": true,
  "access_controls": "attribute_based_access_control"
}""",
            "model_card.json": """{
  "model_name": "HRAdvisor-v2",
  "provider": "internal",
  "base_model": "claude-3-haiku",
  "risk_level": "high",
  "use_case": "employee_hr_assistance",
  "training": "fine_tuned_on_hr_policy_corpus",
  "last_red_team": "2024-02-20"
}""",
            "eval_report.json": """{
  "eval_date": "2024-04-01",
  "coverage": ["policy_accuracy", "benefits_accuracy", "tone_appropriateness"],
  "harmful_failure_modes": ["incorrect_policy_citations", "hallucinated_benefits",
                             "discriminatory_responses_tested_and_mitigated"],
  "adversarial_testing": "prompt_injection, jailbreak_attempts",
  "scores": {"policy_accuracy": 0.94, "tone": 0.97, "hallucination_rate": 0.02}
}""",
            "logging_policy.yaml": """# Logging — HRAdvisor AI
capture: redacted
pii_filter: enabled
pii_types_filtered: [name, ssn, salary, performance_rating]
raw_prompt_capture: disabled
log_retention_days: 30
""",
            "retention_policy.yaml": """# Retention Policy — HRAdvisor AI
# Approved by Legal 2024-02-01
retention_days: 120
# Extended from default 30 days for compliance audit purposes
# Legal requested 120 days to support EEOC investigation window
conversation_history: 120
model_outputs: 120
feedback: 365
""",
            "human_oversight.md": """# Human Oversight — HRAdvisor AI

## Monitoring
- Daily review of flagged conversations
- Weekly accuracy sampling (5% of conversations)

## Automated Safeguards
- Confidence threshold: refuse answer if < 0.75
- Topic blocklist: salary negotiations, performance reviews, termination

## Escalation
(Escalation plan pending HR ops review — ticket HR-2024-0892)

## Override
Employees can always contact HR directly via HRSS portal
""",
            "security_review.md": """# Security Review — HRAdvisor AI

## Authentication
- SSO ✅, MFA for sensitive queries ✅
- Employee can only access own records ✅

## Audit Trail
- All queries logged (redacted) ✅
- Audit retention: 365 days ✅
- Read-only system — no rollback needed

## Encryption
- TLS 1.3 in transit ✅
- AES-256 at rest ✅
""",
            "incident_history.json": """{
  "incidents": [
    {
      "date": "2024-01-10",
      "severity": "low",
      "description": "Incorrect PTO balance displayed due to sync lag",
      "resolution": "Fixed data sync delay — resolved in 4 hours"
    }
  ]
}"""
        }
    }


def _task_3_hard() -> Dict[str, Any]:
    """
    HARD: Financial Planning Copilot
    Three violations: DOMAIN-004 + TRAINING-006 + PII-001
    Violations are subtle and partially hidden — requires careful inspection.
    Expected: REJECT (multiple critical violations)
    """
    return {
        "task_id": "task_3_hard",
        "feature_name": "WealthWise Employee Financial Copilot",
        "feature_summary": (
            "AI financial wellness assistant for employees. Helps with budgeting, "
            "benefits optimization, and retirement planning guidance. "
            "Powered by FinanceGPT, an external provider with specialized financial training."
        ),
        "ground_truth_violations": ["DOMAIN-004", "TRAINING-006", "PII-001"],
        "expected_risk": "critical",
        "expected_decision": "reject",
        "artifacts": {
            "product_spec.md": """# WealthWise — Product Spec v1.0

## Overview
Financial wellness copilot to help employees maximize compensation and plan for retirement.

## Features
- Benefits optimization recommendations
- 401k contribution guidance
- Tax-advantaged account education (HSA, FSA, ESPP)
- Personalized investment allocation suggestions based on employee risk profile
- Debt paydown strategies

## Data Access
- Payroll data (salary, bonus, equity)
- Benefits enrollment
- 401k provider API (Fidelity)
- Employee-provided financial context

## Target Users
All full-time employees (FTEs) globally
""",
            "data_sources.json": """{
  "sources": [
    {"name": "payroll_api", "sensitivity": "restricted", "contains_pii": true},
    {"name": "benefits_api", "sensitivity": "confidential"},
    {"name": "fidelity_401k_api", "sensitivity": "restricted", "contains_pii": true,
     "pii_types": ["account_number", "portfolio_value", "ssn_for_verification"]},
    {"name": "employee_provided_context", "sensitivity": "restricted", "contains_pii": true}
  ],
  "sensitive": true,
  "access_controls": "oauth2",
  "note": "least_privilege review pending — scheduled for Q3"
}""",
            "model_card.json": """{
  "model_name": "FinanceGPT-v3",
  "provider": "external",
  "provider_name": "FinSight AI Inc.",
  "contract_version": "enterprise_v2",
  "training on prompts": "enabled",
  "opt_out_requested": false,
  "risk_level": "high",
  "specialization": "personal_finance_and_investment_advice",
  "last_red_team": "by_vendor_only"
}""",
            "eval_report.json": """{
  "eval_date": "2024-03-20",
  "coverage": ["benefits_accuracy", "retirement_planning", "ui_quality"],
  "notes": "Evaluation focused on accuracy of 401k recommendations",
  "scores": {"benefits_accuracy": 0.88, "user_satisfaction": 4.2},
  "known_gaps": ["tax_advice_accuracy_not_tested", "adversarial_prompts_pending"]
}""",
            "logging_policy.yaml": """# Logging — WealthWise
capture: full
raw_prompt_capture: enabled
include_financial_context: true
pii_filter: planned_for_v2
# PII filter not yet implemented — targeting v1.1 release
log_retention_days: 60
log_destination: s3://wealthwise-logs/
""",
            "retention_policy.yaml": """# Retention — WealthWise
retention_days: 60
conversation_history: 60
financial_data: 60
""",
            "human_oversight.md": """# Human Oversight — WealthWise

## Escalation
Employees with complex situations are directed to Fidelity advisor hotline.

## Disclaimers
All recommendations include: 'For informational purposes only. Consult a financial advisor.'

## Monitoring
Monthly review of conversation samples by HR Benefits team.
""",
            "security_review.md": """# Security Review — WealthWise v1.0

## Authentication
- OAuth2 with Okta ✅
- Employee self-service portal ✅

## Data Handling
- TLS 1.3 in transit ✅
- External provider (FinSight AI) data agreement on file
- DPA signed with FinSight covering GDPR and CCPA

## Note
PII masking and audit trail requirements under review by security team.
""",
            "incident_history.json": """{
  "incidents": [
    {"date": "2024-02-14", "severity": "medium",
     "description": "FinanceGPT provided specific stock pick recommendation to 3 users",
     "resolution": "Prompt guardrails updated, vendor notified"},
    {"date": "2024-03-01", "severity": "low",
     "description": "401k balance displayed incorrectly due to API timeout",
     "resolution": "Retry logic added"}
  ]
}"""
        }
    }
