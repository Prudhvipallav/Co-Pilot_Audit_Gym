"""
FastAPI server — OpenEnv endpoints + multi-agent (v2) + adversarial/curriculum (v3).
Endpoints: /generate_task, /mutate_task, /adversarial_task, /judge, /curriculum
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import json

from .env import GovernanceReviewEnv
from .graders import calculate_grader_score, validate_score_range, score_to_grade
from .artifacts import get_artifact_descriptions, get_policy_reference
from .models import (
    ResetRequest, StepRequest, GenerateRequest, MutationRequest,
    JudgeRequest, GeneratedTask
)


app = FastAPI(
    title="OpenEnv — AI Governance Review (v3)",
    description="3-agent governance review: Adversarial Problem Maker → Qwen Reviewer → 3-Persona Judge",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

env = GovernanceReviewEnv(task_id=1)
_last_generated_task: Optional[GeneratedTask] = None


# ─── Original Required Endpoints ─────────────────────────────────

@app.post("/reset")
def reset(request: ResetRequest = ResetRequest()):
    global env, _last_generated_task
    task_id = max(1, min(3, request.task_id or 1))
    env = GovernanceReviewEnv(task_id=task_id)
    options = {}
    if request.use_generated_task and _last_generated_task:
        options["generated_task"] = _last_generated_task
    if request.reviewer_model:
        options["reviewer_model"] = request.reviewer_model
    obs_json, info = env.reset(options=options if options else None)
    return {"observation": json.loads(obs_json), "info": info}


@app.post("/step")
def step(request: StepRequest):
    obs_json, reward, done, truncated, info = env.step(request.action)
    return {
        "observation": json.loads(obs_json),
        "reward": reward,
        "done": done,
        "truncated": truncated,
        "info": info
    }


@app.get("/state")
def state():
    s = env._episode_state
    if not s:
        raise HTTPException(400, "Call /reset first")
    return {
        "task_id": s["task_data"]["task_id"],
        "feature_name": s["task_data"]["feature_name"],
        "step_count": s["step_count"],
        "review_stage": s["review_stage"],
        "flagged_issues": s["flagged_issues"],
        "requested_mitigations": s["requested_mitigations"],
        "current_risk": s["current_risk"],
        "final_decision": s["final_decision"],
        "done": s["done"],
    }


@app.get("/tasks")
def get_tasks():
    action_schema = {
        "action_type": "inspect_artifact|flag_issue|request_mitigation|set_risk|approve|reject|escalate",
        "target": "string (optional)",
        "severity": "low|medium|high|critical (optional)",
        "issue_code": "PII-001|ACCESS-002|ESCALATION-003|DOMAIN-004|RETENTION-005|TRAINING-006|AUDIT-007|EVAL-008",
        "note": "string (optional)"
    }
    return {
        "tasks": [
            {"id": "task_1_easy", "task_id": 1, "name": "CustomerCare AI Copilot",
             "difficulty": "easy", "num_violations": 1, "expected_decision": "reject",
             "action_schema": action_schema},
            {"id": "task_2_medium", "task_id": 2, "name": "HRAdvisor AI",
             "difficulty": "medium", "num_violations": 2, "expected_decision": "escalate",
             "action_schema": action_schema},
            {"id": "task_3_hard", "task_id": 3, "name": "WealthWise Financial Copilot",
             "difficulty": "hard", "num_violations": 3, "expected_decision": "reject",
             "action_schema": action_schema},
        ],
        "dynamic_tasks": "Use POST /generate_task or POST /mutate_task for infinite novel scenarios",
        "policy_reference": get_policy_reference(),
        "artifact_guide": get_artifact_descriptions()
    }


@app.get("/grader")
def grader():
    if not env._episode_state:
        raise HTTPException(400, "Call /reset first")
    score = calculate_grader_score(env)
    if not validate_score_range(score):
        raise HTTPException(500, "Scores out of range")
    return {"scores": score, "grade": score_to_grade(score["overall"])}


@app.get("/artifact/{name}")
def get_artifact_content(name: str):
    """Returns the raw content of a specific artifact in the current task."""
    if not env._episode_state:
        raise HTTPException(400, "Call /reset first")
    artifacts = env._episode_state["task_data"]["artifacts"]
    if name not in artifacts:
        raise HTTPException(404, f"Artifact '{name}' not found")
    return {"name": name, "content": artifacts[name]}


@app.post("/baseline")
def baseline():
    from .tasks import load_task
    from .policies import get_violations, POLICY_RULE_MAP
    results = []
    for task_id in [1, 2, 3]:
        task = load_task(task_id)
        gt = task["ground_truth_violations"]
        found = get_violations(task["artifacts"])
        correct = [v for v in found if v in gt]
        fp = [v for v in found if v not in gt]
        critical_gt = [c for c in gt if POLICY_RULE_MAP.get(c) and POLICY_RULE_MAP[c].severity in ("critical", "high")]
        safety = 1.0 if all(c in correct for c in critical_gt) else 0.0
        compliance = len(correct) / max(1, len(gt))
        precision = len(correct) / max(1, len(found))
        overall = safety * 0.35 + compliance * 0.25 + 1.0 * 0.20 + precision * 0.10 + 0.5 * 0.10
        results.append({
            "task_id": task["task_id"],
            "scores": {
                "safety": round(safety, 4), "compliance": round(compliance, 4),
                "completeness": 1.0, "precision": round(precision, 4),
                "mitigation_quality": 0.5, "overall": round(min(1.0, overall), 4)
            }
        })
    return {
        "status": "success", "baseline_type": "rule_based_deterministic",
        "results": results,
        "average_overall": round(sum(r["scores"]["overall"] for r in results) / 3, 4)
    }


# ─── NEW: Multi-Agent Endpoints ───────────────────────────────────

@app.post("/generate_task")
def generate_task_endpoint(request: GenerateRequest = GenerateRequest()):
    """SLM Problem Maker (API) generates a brand-new governance scenario."""
    global _last_generated_task
    from .problem_maker import generate_task
    task = generate_task(request)
    _last_generated_task = task
    return {
        "task_id": task.task_id,
        "source": task.source,
        "feature_name": task.feature_name,
        "feature_summary": task.feature_summary,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "num_violations": len(task.ground_truth_violations),
        "expected_decision": task.expected_decision,
        "expected_risk": task.expected_risk,
        "artifacts_available": list(task.artifacts.keys()),
        "message": "Generated. Call POST /reset with use_generated_task=true to load it."
    }


@app.post("/mutate_task")
def mutate_task_endpoint(request: MutationRequest = MutationRequest()):
    """SLM Problem Maker (API) injects new violations into an existing task."""
    global _last_generated_task
    from .mutator import mutate_task
    task = mutate_task(request)
    _last_generated_task = task
    return {
        "task_id": task.task_id,
        "source": task.source,
        "base_task_id": request.base_task_id,
        "feature_name": task.feature_name,
        "difficulty": task.difficulty,
        "num_violations": len(task.ground_truth_violations),
        "expected_decision": task.expected_decision,
        "mutation_description": task.mutation_description,
        "artifacts_available": list(task.artifacts.keys()),
        "message": "Mutated. Call POST /reset with use_generated_task=true to load it."
    }


@app.post("/judge")
def judge_endpoint(request: JudgeRequest):
    """3-persona LLM Judge evaluates a completed reviewer episode (Junior/Senior/Principal GRC)."""
    from .judge import judge_episode
    verdict = judge_episode(request.transcript)
    return verdict.model_dump()


@app.post("/adversarial_task")
def adversarial_task_endpoint(request: GenerateRequest = GenerateRequest()):
    """Adversarial Problem Maker targets the agent's tracked weaknesses from the weakness map."""
    global _last_generated_task
    from .adversarial_maker import generate_adversarial_task
    task = generate_adversarial_task(
        num_violations=request.num_violations,
        forced_domain=request.domain,
        difficulty=request.difficulty,
    )
    _last_generated_task = task
    return {
        "task_id": task.task_id,
        "source": task.source,
        "feature_name": task.feature_name,
        "domain": task.domain,
        "difficulty": task.difficulty,
        "ground_truth_violations": task.ground_truth_violations,
        "expected_decision": task.expected_decision,
        "expected_risk": task.expected_risk,
        "artifacts_available": list(task.artifacts.keys()),
        "message": "Adversarial task loaded. Call POST /reset with use_generated_task=true."
    }


@app.get("/curriculum")
def curriculum_endpoint():
    """Returns current curriculum state: difficulty, mastery progress, and next gate threshold."""
    from training.curriculum import get_curriculum_info
    from memory.weakness_map import get_violation_weights, load_weakness_map
    info = get_curriculum_info()
    wmap = load_weakness_map()
    sorted_weaknesses = sorted(
        [(code, d["miss_rate"]) for code, d in wmap.items()],
        key=lambda x: -x[1]
    )
    return {
        "curriculum": info,
        "top_weaknesses": sorted_weaknesses[:4],
        "weakness_map": wmap,
    }


@app.get("/health")
def health():
    from .model_config import load_config
    cfg = load_config()
    return {
        "status": "healthy",
        "version": "2.0.0",
        "agents": {
            "reviewer": f"{cfg['reviewer']['provider']}:{cfg['reviewer']['model_id']}",
            "problem_maker": f"{cfg['problem_maker']['provider']}:{cfg['problem_maker']['model_id']}",
            "judge": f"{cfg['judge']['provider']}:{cfg['judge']['model_id']}",
        }
    }


@app.get("/")
def root():
    return {
        "name": "OpenEnv — AI Governance Review (Multi-Agent)",
        "endpoints": {
            "standard": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
            "multi_agent": ["/generate_task", "/mutate_task", "/judge"]
        },
        "reviewer": "Qwen2.5-0.5B-Instruct (local)",
        "judge": "API-based (gpt-4o or claude)",
        "docs": "/docs"
    }
