"""
Task Mutator — injects new violations into existing tasks via LLM-based artifact rewriting.
Uses the problem maker API client (gpt-4o-mini or claude-haiku).
"""

import json
import uuid
import random
from .models import GeneratedTask, MutationRequest
from .tasks import load_task
from .policies import POLICY_CHECKS, POLICY_RULE_MAP, get_violations
from .model_config import get_problem_maker_client

MUTATOR_SYSTEM = """You are an enterprise document editor.
You receive an artifact file and must subtly modify it to introduce a specific policy violation.
The modification must look realistic — like an honest mistake a developer or PM might make.
Return ONLY the modified file content. No explanation. No markdown fences. No comments like "# VIOLATION"."""

MUTATOR_PROMPT = """Modify this artifact to introduce violation {code}: {description}

Original artifact ({filename}):
{content}

Rules:
- Keep 80% of original content intact
- Embed the violation naturally — it should look like an oversight
- The violation must be detectable by careful reading
- Do NOT add comments indicating you added a violation

Return only the modified file content."""


def mutate_task(request: MutationRequest) -> GeneratedTask:
    """Load a base task and inject new violations into its artifacts."""
    base = load_task(request.base_task_id)
    artifacts = dict(base["artifacts"])
    existing = set(base["ground_truth_violations"])

    # Pick new violations not already present
    available = [c for c in POLICY_CHECKS if c not in existing]
    if request.difficulty_boost:
        priority = [c for c in available if POLICY_RULE_MAP[c].severity in ("critical", "high")]
        pool = priority if priority else available
    else:
        pool = available

    num_to_add = min(request.num_extra_violations, len(pool))
    new_violations = random.sample(pool, num_to_add)
    mutation_notes = []

    for code in new_violations:
        rule = POLICY_RULE_MAP[code]
        target = rule.artifact_hint
        try:
            client = get_problem_maker_client()
            original = artifacts.get(target, "")
            if not original:
                raise ValueError(f"Artifact {target} not found")

            mutated = client.complete(
                MUTATOR_SYSTEM,
                MUTATOR_PROMPT.format(
                    code=code,
                    description=rule.description,
                    filename=target,
                    content=original[:1500]
                )
            )
            artifacts[target] = mutated.strip()
            mutation_notes.append(f"LLM mutated {target} for {code}")

        except Exception as e:
            from .problem_maker import _patch_violation
            artifacts = _patch_violation(artifacts, code)
            mutation_notes.append(f"Template patched {target} for {code} (error: {str(e)[:60]})")

    final = list(existing) + new_violations
    has_critical = any(POLICY_RULE_MAP[c].severity == "critical" for c in final)
    has_high = any(POLICY_RULE_MAP[c].severity == "high" for c in final)

    if has_critical:
        expected_decision = "reject"
        expected_risk = "critical" if len(final) >= 3 else "high"
    elif has_high:
        expected_decision = "escalate"
        expected_risk = "high"
    else:
        expected_decision = "escalate"
        expected_risk = "medium"

    return GeneratedTask(
        task_id=f"mutated_{uuid.uuid4().hex[:8]}",
        source="mutated",
        feature_name=base["feature_name"] + " [Mutated]",
        feature_summary=base["feature_summary"],
        domain="enterprise",
        difficulty="hard",
        artifacts=artifacts,
        ground_truth_violations=final,
        expected_risk=expected_risk,
        expected_decision=expected_decision,
        mutation_description=(
            f"Base: task_{request.base_task_id}. "
            f"Added: {new_violations}. "
            f"Notes: {'; '.join(mutation_notes)}"
        )
    )
