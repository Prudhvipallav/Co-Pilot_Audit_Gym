"""
Orchestrator v3 — full SRE-Gym-style 3-agent pipeline.

Modes:
  static     — use one of the 3 built-in tasks
  generate   — random API-generated task (problem_maker)
  mutate     — inject violations into existing task (mutator)
  adversarial — weakness-targeted generation (adversarial_maker) + curriculum

The adversarial mode is the default for sustained training.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_URL = os.getenv("GOVERNANCE_ENV_URL", "http://localhost:8000")


def orchestrate(
    mode: str = "adversarial",
    task_id: int = 1,
    difficulty: str = None,       # None → use curriculum controller
    domain: str = None,
    base_task: int = 1,
    num_violations: int = None,   # None → auto from difficulty
) -> dict:

    # ─── Curriculum: auto-select difficulty ───────────────────────
    if difficulty is None:
        try:
            from training.curriculum import get_current_difficulty, get_curriculum_info
            difficulty = get_current_difficulty()
            info = get_curriculum_info()
            print(f"  📈 [Curriculum] Difficulty: {difficulty} | "
                  f"Recent avg: {info.get('recent_avg_score', '—')} | "
                  f"Next gate: {info.get('next_threshold', '—')}")
        except Exception:
            difficulty = "medium"

    if num_violations is None:
        num_violations = {"easy": 1, "medium": 2, "hard": 3, "adversarial": 3}.get(difficulty, 2)

    print("=" * 70)
    print("🏛️  GovernanceReview-Gym v3  ·  Self-Improving 3-Agent Pipeline")
    print(f"   Reviewer: Qwen2.5-0.5B (local) | Judge: 3-persona API | Mode: {mode}")
    print(f"   Difficulty: {difficulty} | Violations target: {num_violations}")
    print("=" * 70)

    # ─── Health check ────────────────────────────────────────────
    try:
        requests.get(f"{BASE_URL}/health").raise_for_status()
    except Exception as e:
        print(f"❌ Server not running at {BASE_URL}: {e}")
        print("Run: py -3.11 -m uvicorn app.main:app --reload --port 8000")
        sys.exit(1)

    # ─── Step 1: Problem Maker ────────────────────────────────────
    use_generated = False
    generated_meta = None

    if mode == "adversarial":
        print("\n🎯 [Agent 1: Adversarial Maker] Targeting agent weaknesses...")
        from memory.weakness_map import get_weakness_summary, get_violation_weights
        print(get_weakness_summary())

        resp = requests.post(f"{BASE_URL}/adversarial_task", json={
            "num_violations": num_violations,
            "difficulty": difficulty,
            "domain": domain,
        })
        resp.raise_for_status()
        generated_meta = resp.json()
        print(f"   ✅ {generated_meta['feature_name']}")
        print(f"   Target violations: {generated_meta.get('ground_truth_violations', [])}")
        use_generated = True

    elif mode == "generate":
        print("\n📝 [Agent 1: Problem Maker] Generating new scenario...")
        resp = requests.post(f"{BASE_URL}/generate_task", json={
            "difficulty": difficulty, "domain": domain, "num_violations": num_violations
        })
        resp.raise_for_status()
        generated_meta = resp.json()
        print(f"   ✅ {generated_meta['feature_name']}")
        use_generated = True

    elif mode == "mutate":
        print(f"\n🔀 [Agent 1: Mutator] Mutating task {base_task}...")
        resp = requests.post(f"{BASE_URL}/mutate_task", json={
            "base_task_id": base_task,
            "num_extra_violations": min(num_violations, 3),
            "difficulty_boost": difficulty in ("hard", "adversarial"),
        })
        resp.raise_for_status()
        generated_meta = resp.json()
        print(f"   ✅ {generated_meta['feature_name']}")
        use_generated = True

    else:  # static
        print(f"\n📋 [Static Task] Using pre-built task {task_id}")

    # Reset environment
    resp = requests.post(f"{BASE_URL}/reset", json={
        "task_id": task_id,
        "use_generated_task": use_generated,
        "reviewer_model": "Qwen/Qwen2.5-0.5B-Instruct",
    })
    resp.raise_for_status()
    feature = resp.json()["observation"]["feature_name"]
    print(f"   ✅ Loaded: {feature}")

    # ─── Step 2: Reviewer (Qwen2.5-0.5B, local) ──────────────────
    print(f"\n🔍 [Agent 2: Reviewer (Qwen2.5-0.5B)] Starting review...")
    from agents.reviewer import run_reviewer_episode
    episode = run_reviewer_episode(
        task_id=task_id,
        use_generated=False,   # Already loaded via reset
        save_memory=False,
        verbose=True,
    )
    print(f"\n   Grade: {episode['grade']} | Grader: {episode['grader_score']['overall']:.4f} | "
          f"Steps: {episode['steps']} ({episode['fallback_steps']} fallbacks)")

    # ─── Step 3: 3-Persona Judge ──────────────────────────────────
    print(f"\n⚖️  [Agent 3: Judge (3-persona)] Evaluating...")

    # Build transcript
    transcript = {
        "task_id": str(episode["task_id"] or task_id),
        "feature_name": episode["feature_name"] or feature,
        "feature_summary": episode.get("feature_summary", ""),
        "artifacts": {},
        "ground_truth_violations": [],
        "expected_risk": "unknown",
        "expected_decision": "unknown",
        "steps": episode["steps_log"],
        "final_flagged_issues": episode["flagged_issues"],
        "final_mitigations": episode["requested_mitigations"],
        "final_risk": episode["final_risk"],
        "final_decision": episode["final_decision"],
        "total_reward": episode["total_reward"],
        "total_steps": episode["steps"],
        "reviewer_model": episode["reviewer_model"],
    }

    # Enrich with ground truth
    if mode == "static":
        try:
            from app.tasks import load_task
            task_data = load_task(task_id)
            transcript["ground_truth_violations"] = task_data["ground_truth_violations"]
            transcript["expected_risk"] = task_data["expected_risk"]
            transcript["expected_decision"] = task_data["expected_decision"]
        except Exception:
            pass
    elif generated_meta:
        transcript["ground_truth_violations"] = generated_meta.get("ground_truth_violations",
                                                                    generated_meta.get("violations_target", []))
        transcript["expected_decision"] = generated_meta.get("expected_decision", "reject")

    verdict = {"overall_score": None, "grade": "?", "overall_feedback": "Not run"}
    try:
        j = requests.post(f"{BASE_URL}/judge", json={"transcript": transcript}, timeout=60)
        if j.status_code == 200:
            verdict = j.json()
            print(f"   Junior (detection):    {verdict['detection_accuracy']['score']:.2f}")
            print(f"   Senior (reasoning):    {verdict['reasoning_quality']['score']:.2f}")
            print(f"   Principal (decision):  {verdict['decision_appropriateness']['score']:.2f}")
            print(f"   Overall: {verdict['overall_score']:.2f} ({verdict['grade']})")
            print(f"   {verdict.get('overall_feedback', '')[:160]}")
    except Exception as e:
        print(f"   ⚠️ Judge fallback: {e}")

    # ─── Step 4: Update Weakness Map + Memory ─────────────────────
    from memory.knowledge_base import save_episode_result
    from memory.weakness_map import update_weakness_map

    gt = transcript.get("ground_truth_violations", [])
    flagged_codes = [i.get("code", "") for i in episode["flagged_issues"]]
    caught = [c for c in gt if c in flagged_codes]
    missed = [c for c in gt if c not in flagged_codes]
    fp = [c for c in flagged_codes if c not in gt]

    # Update EMA weakness map
    update_weakness_map(gt, caught, missed)
    from memory.weakness_map import get_weakness_summary
    print(f"\n{get_weakness_summary()}")

    # Save to memory
    save_episode_result(
        task_id=str(transcript["task_id"]),
        grader_score=episode["grader_score"],
        caught=caught,
        missed=missed,
        false_positives=fp,
        final_decision=episode["final_decision"],
        expected_decision=transcript.get("expected_decision"),
        judge_grade=verdict.get("grade"),
        judge_feedback=verdict.get("overall_feedback"),
    )

    # ─── Summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 FINAL RESULTS")
    print(f"   Grader (deterministic): {episode['grader_score']['overall']:.4f} ({episode['grade']})")
    if verdict.get("overall_score") is not None:
        print(f"   Judge  (3-persona):     {verdict['overall_score']:.4f} ({verdict['grade']})")
    print(f"   Caught: {caught or ['none']}   Missed: {missed or ['none']}   FP: {fp or ['none']}")
    print("=" * 70)

    results = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "difficulty": difficulty,
        "generated_task": generated_meta,
        "episode": {k: v for k, v in episode.items() if k != "steps_log"},
        "judge_verdict": verdict,
        "weakness_update": {"caught": caught, "missed": missed, "fp": fp},
    }

    fname = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(fname, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results → {fname}")
    return results


# ─── CLI ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GovernanceReview-Gym v3 Orchestrator")
    parser.add_argument("--mode", choices=["static", "generate", "mutate", "adversarial"],
                        default="adversarial",
                        help="adversarial (default): weakness-targeted | static: built-in tasks")
    parser.add_argument("--task_id",   type=int, default=1)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "adversarial"],
                        default=None, help="Override curriculum controller")
    parser.add_argument("--domain",    type=str, default=None)
    parser.add_argument("--base_task", type=int, default=1)
    parser.add_argument("--n_episodes", type=int, default=1,
                        help="Run N consecutive episodes (curriculum auto-advances)")
    args = parser.parse_args()

    for episode_num in range(args.n_episodes):
        if args.n_episodes > 1:
            print(f"\n{'─'*70}")
            print(f"  🔁 Episode {episode_num + 1} / {args.n_episodes}")
            print(f"{'─'*70}\n")
        orchestrate(
            mode=args.mode,
            task_id=args.task_id,
            difficulty=args.difficulty,
            domain=args.domain,
            base_task=args.base_task,
        )
