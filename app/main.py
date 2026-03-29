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
    """Returns current curriculum state: weakness map and difficulty progression."""
    from memory.weakness_map import load_weakness_map
    wmap = load_weakness_map()
    sorted_weaknesses = sorted(
        [(code, d["miss_rate"]) for code, d in wmap.items()],
        key=lambda x: -x[1]
    )
    # Derive difficulty from weakness map performance
    avg_miss = sum(d["miss_rate"] for d in wmap.values()) / max(1, len(wmap))
    if avg_miss > 0.6:
        current_difficulty = "easy"
    elif avg_miss > 0.35:
        current_difficulty = "medium"
    else:
        current_difficulty = "hard"
    return {
        "curriculum": {
            "current_difficulty": current_difficulty,
            "avg_miss_rate": round(avg_miss, 4),
        },
        "top_weaknesses": sorted_weaknesses[:4],
        "weakness_map": wmap,
    }


@app.get("/health")
def health():
    from .model_config import load_config
    cfg = load_config()
    return {
        "status": "healthy",
        "version": "3.1.0",
        "agents": {
            "reviewer": f"{cfg['reviewer']['provider']}:{cfg['reviewer']['model_id']}",
            "problem_maker": f"{cfg['problem_maker']['provider']}:{cfg['problem_maker']['model_id']}",
            "judge": f"{cfg['judge']['provider']}:{cfg['judge']['model_id']}",
        }
    }


# ─── Feature 1: Explainable Reward Decomposition ─────────────

@app.get("/explain")
def explain():
    """Return a full per-step breakdown of WHY the agent got each reward."""
    try:
        return env.get_reward_explanation()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ─── Feature 3: ELO Adaptive Difficulty ──────────────────────

@app.get("/elo")
def get_elo():
    """Return current ELO rating, grade, and recommended task difficulty."""
    from .elo import get_elo_tracker
    tracker = get_elo_tracker()
    return tracker.to_dict()


@app.post("/elo/update")
def update_elo(score: float):
    """Update ELO after an episode. Pass the overall grader score."""
    from .elo import get_elo_tracker
    tracker = get_elo_tracker()
    new_elo = tracker.update(score)
    return {
        "elo": new_elo,
        "grade": tracker.get_grade(),
        "recommended_task": tracker.select_task_id(),
    }


# ─── Feature 4: Adversarial Red Team Loop ────────────────────

@app.post("/redteam/start")
def redteam_start(task_id: int = 1, num_rounds: int = 3):
    """Run N rounds of the adversarial red team self-play loop."""
    import json
    from .red_team import get_red_team_session
    from .elo import get_elo_tracker
    from .env import GovernanceReviewEnv
    from .policies import run_all_checks, POLICY_RULE_MAP

    def _rule_based_agent(obs: dict) -> dict:
        """Deterministic baseline reviewer for red team rounds."""
        inspected = obs.get("inspected_artifacts", [])
        artifacts = list(obs.get("visible_artifacts", {}).keys())
        flagged_codes = [i.get("code", "") for i in obs.get("flagged_issues", [])]
        mitigated = [m.get("issue_code") for m in obs.get("requested_mitigations", [])]
        risk = obs.get("current_risk")

        for art in artifacts:
            if art not in inspected:
                return {"action_type": "inspect_artifact", "target": art}

        full = obs.get("full_artifacts", {})
        try:
            for code, (passed, msg) in run_all_checks(full).items():
                if not passed and code not in flagged_codes:
                    rule = POLICY_RULE_MAP.get(code)
                    return {"action_type": "flag_issue", "issue_code": code,
                            "severity": rule.severity if rule else "high",
                            "target": rule.artifact_hint if rule else "",
                            "note": msg[:120]}
        except Exception:
            pass

        for issue in obs.get("flagged_issues", []):
            code = issue.get("code", "")
            if code and code not in mitigated:
                return {"action_type": "request_mitigation", "issue_code": code,
                        "note": f"Remediate {code}"}

        if obs.get("flagged_issues") and not risk:
            sevs = [i.get("severity", "medium") for i in obs.get("flagged_issues", [])]
            sv = "critical" if "critical" in sevs else "high" if "high" in sevs else "medium"
            return {"action_type": "set_risk", "severity": sv}

        if obs.get("flagged_issues"):
            return {"action_type": "reject", "note": "Violations found"}
        return {"action_type": "approve", "note": "No violations"}

    session = get_red_team_session()
    elo_tracker = get_elo_tracker()
    results = []

    for _ in range(num_rounds):
        round_env = GovernanceReviewEnv(task_id=elo_tracker.select_task_id())
        result = session.play_round(round_env, _rule_based_agent)
        elo_tracker.update(result["scores"]["overall"])
        result["elo_after"] = round(elo_tracker.elo, 1)
        results.append(result)

    return {
        "rounds_played": num_rounds,
        "leaderboard": session.get_leaderboard(),
        "elo": elo_tracker.to_dict(),
        "round_results": results,
    }


@app.get("/redteam/status")
def redteam_status():
    """Return current red team session leaderboard and history."""
    from .red_team import get_red_team_session
    from .elo import get_elo_tracker
    session = get_red_team_session()
    elo = get_elo_tracker()
    return {
        "leaderboard": session.get_leaderboard(),
        "elo": elo.to_dict(),
    }


@app.post("/redteam/reset")
def redteam_reset():
    """Reset the red team session and ELO tracker."""
    from .red_team import get_red_team_session
    from .elo import get_elo_tracker
    get_red_team_session().reset_session()
    tracker = get_elo_tracker()
    tracker.elo = 1000.0
    tracker.round_history = []
    tracker.save()
    return {"status": "reset", "elo": 1000.0}


@app.get("/info")
def info():
    return {
        "name": "CopilotAudit-Gym PRO v3.1",
        "endpoints": {
            "standard": ["/reset", "/step", "/state", "/tasks", "/grader", "/baseline"],
            "multi_agent": ["/generate_task", "/mutate_task", "/judge"],
            "advanced": ["/explain", "/elo", "/elo/update", "/redteam/start", "/redteam/status", "/redteam/reset"],
        },
        "features": [
            "Explainable reward decomposition (/explain)",
            "Causal violation graph (in observation)",
            "ELO-based adaptive difficulty (/elo)",
            "Adversarial red team self-play (/redteam/start)",
            "Counterfactual regret computation (/explain → regret_pct)",
        ],
        "dashboard": "/ui",
        "docs": "/docs"
    }


# ─── Mount Gradio Dashboard at /ui (SAFE — evaluation unaffected) ─────────────
# This is wrapped in try/except. If Gradio fails for any reason,
# FastAPI and ALL evaluation endpoints (/reset /step /grader etc.) keep working.
try:
    import gradio as _gr
    from app_ui import create_app as _create_gradio_app
    _demo = _create_gradio_app()
    app = _gr.mount_gradio_app(app, _demo, path="/")
    print("[Dashboard] ✅ Gradio dashboard mounted at /")
except Exception as _e:
    print(f"[Dashboard] ⚠️  Gradio not mounted (evaluation unaffected): {_e}")
