"""
GovernanceReviewEnv: Core gymnasium environment for AI governance review.

Agent receives enterprise copilot launch packets and must:
1. Inspect artifacts (product spec, data sources, eval report, etc.)
2. Flag policy violations with correct codes and severity
3. Request mitigations for each violation
4. Set the overall risk level
5. Make a final decision: approve / reject / escalate

Reward shaping provides dense signals throughout the episode.
"""

import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Any, Optional, Tuple
import json

from .models import (
    GovernanceAction, GovernanceObservation, ActionType,
    Severity, RiskLevel
)
from .policies import POLICY_CHECKS, POLICY_RULE_MAP, get_causal_hints
from .tasks import load_task


class GovernanceReviewEnv(gym.Env):
    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        task_id: int = 1,
        render_mode: Optional[str] = None,
        max_steps: int = 50
    ):
        super().__init__()
        self.task_id = task_id
        self.render_mode = render_mode
        self.max_steps = max_steps

        # Action space: JSON-encoded GovernanceAction
        self.action_space = spaces.Text(max_length=1000)
        # Observation space: JSON-encoded GovernanceObservation
        self.observation_space = spaces.Text(max_length=10000)

        # Episode state (initialized in reset)
        self._episode_state = {}

        # Mastery tracking for curriculum escalation
        self.episode_history: List[float] = []
        self.mastery_threshold = 0.72
        self._reviewer_model: str = "unknown"

    def reset(self, seed=None, options=None) -> Tuple[str, Dict]:
        super().reset(seed=seed)

        task_options = options or {}

        # Support generated tasks passed via options
        if "generated_task" in task_options:
            t = task_options["generated_task"]
            # Ensure t is a GeneratedTask model or dict
            if hasattr(t, "model_dump"):
                t_data = t.model_dump()
            else:
                t_data = t
                
            task_data = {
                "task_id": t_data["task_id"],
                "feature_name": t_data["feature_name"],
                "feature_summary": t_data["feature_summary"],
                "artifacts": t_data["artifacts"],
                "ground_truth_violations": t_data["ground_truth_violations"],
                "expected_risk": t_data["expected_risk"],
                "expected_decision": t_data["expected_decision"],
            }
        else:
            effective_task_id = task_options.get("task_id", self._get_effective_task_id())
            task_data = load_task(effective_task_id, seed)

        self._reviewer_model = task_options.get("reviewer_model", "unknown")

        self._episode_state = {
            "task_data": task_data,
            "artifacts": task_data["artifacts"],
            "ground_truth_violations": task_data["ground_truth_violations"],
            "expected_risk": task_data["expected_risk"],
            "expected_decision": task_data["expected_decision"],
            # Agent progress
            "step_count": 0,
            "done": False,
            "review_stage": "inspection",
            "inspected_artifacts": [],
            "flagged_issues": [],
            "requested_mitigations": [],
            "current_risk": None,
            "final_decision": None,
            # Tracking
            "false_positives": [],
            "missed_issues": [],
            "reward_total": 0.0,
            "episode_steps": [],    # For transcript tracking
            "reward_log": [],       # Explainable reward decomposition
            "causal_hints": [],     # Causal violation graph hints
        }

        obs = self._build_observation("Welcome! Inspect artifacts to begin the governance review.")
        return obs.model_dump_json(), {}

    def step(self, action_text: str) -> Tuple[str, float, bool, bool, Dict]:
        s = self._episode_state

        # ── P1 fix: reject actions after terminal state ───────────
        if s.get("done"):
            obs = self._build_observation(
                "⚠️ Episode already finished. Call /reset to start a new episode."
            )
            return obs.model_dump_json(), 0.0, True, False, {"error": "post_terminal_action"}

        s["step_count"] += 1
        reward = 0.0
        message = ""
        info = {}

        # --- Parse action ---
        try:
            action_dict = json.loads(action_text)
            action = GovernanceAction(**action_dict)
        except Exception as e:
            obs = self._build_observation(f"⚠️ Invalid action format: {e}. Use JSON matching GovernanceAction schema.")
            return obs.model_dump_json(), -0.5, False, False, {"error": str(e)}

        # --- Execute action ---
        atype = action.action_type

        if atype == ActionType.INSPECT_ARTIFACT:
            reward, message = self._handle_inspect(action)

        elif atype == ActionType.FLAG_ISSUE:
            reward, message = self._handle_flag(action)

        elif atype == ActionType.REQUEST_MITIGATION:
            reward, message = self._handle_mitigation(action)

        elif atype == ActionType.SET_RISK:
            reward, message = self._handle_set_risk(action)

        elif atype in (ActionType.APPROVE, ActionType.REJECT, ActionType.ESCALATE):
            reward, message = self._handle_final_decision(action)
            s["done"] = True

        elif atype == ActionType.FINALIZE_REVIEW:
            # Only valid after approve/reject/escalate
            if s["final_decision"]:
                reward, message = self._compute_final_reward()
                s["done"] = True
            else:
                reward = -0.2
                message = "⚠️ Must call approve/reject/escalate before finalizing."

        # --- Time tax (prevents stalling) ---
        if not s["done"]:
            reward -= 0.05

        # --- Timeout ---
        if s["step_count"] >= self.max_steps and not s["done"]:
            s["done"] = True
            reward -= 3.0
            message = "⏰ Episode timed out — review incomplete."
            info["timeout"] = True

        s["reward_total"] += reward

        # Record reward with explanation
        s["reward_log"].append({
            "step": s["step_count"],
            "action_type": action_dict.get("action_type", "unknown") if isinstance(action_dict, dict) else "unknown",
            "detail": action_dict.get("target") or action_dict.get("issue_code") or "",
            "reward": round(reward, 3),
            "reason": message,
        })

        # Update causal hints when new flags are added
        flagged_codes = [i["code"] for i in s["flagged_issues"]]
        s["causal_hints"] = get_causal_hints(flagged_codes)

        # Record step for transcript
        try:
            action_dict = json.loads(action_text)
        except Exception:
            action_dict = {"action_type": "unknown", "raw": action_text}
        s["episode_steps"].append({
            "step": s["step_count"],
            "action": action_dict,
            "reward": reward,
            "message": message[:200]
        })

        obs = self._build_observation(message)
        return obs.model_dump_json(), reward, s["done"], False, info

    # ─────────────────────────────────────────────────────────────────
    # Action Handlers
    # ─────────────────────────────────────────────────────────────────

    def _handle_inspect(self, action: GovernanceAction) -> Tuple[float, str]:
        s = self._episode_state
        target = action.target or ""
        artifacts = s["artifacts"]

        if target not in artifacts:
            available = ", ".join(artifacts.keys())
            return -0.1, f"❌ Artifact '{target}' not found. Available: {available}"

        if target in s["inspected_artifacts"]:
            return 0.0, f"ℹ️ Already inspected '{target}'. Check observation for full content."

        s["inspected_artifacts"].append(target)

        # Enhanced Reward for inspection progress (cumulative signals)
        num_artifacts = len(artifacts)
        num_inspected = len(s["inspected_artifacts"])
        
        # Base reward for the act of inspection
        reward = 0.2
        
        # Bonus for completing first and last artifacts
        if num_inspected == 1:
            reward += 0.2 # Early momentum
        if num_inspected == num_artifacts:
            reward += 0.5 # Completion bonus
            
        return reward, f"✅ Inspected '{target}'. ({num_inspected}/{num_artifacts} artifacts complete)"

    def _handle_flag(self, action: GovernanceAction) -> Tuple[float, str]:
        s = self._episode_state
        code = action.issue_code or ""
        gt_violations = s["ground_truth_violations"]

        if not code:
            return -0.1, "⚠️ Must provide issue_code when flagging an issue."

        # Duplicate check
        already_flagged = [i["code"] for i in s["flagged_issues"]]
        if code in already_flagged:
            return 0.0, f"ℹ️ Issue {code} already flagged."

        if code in gt_violations:
            # Correct flag
            rule = POLICY_RULE_MAP.get(code)
            expected_severity = rule.severity if rule else "medium"
            severity_match = action.severity and action.severity.value == expected_severity

            s["flagged_issues"].append({
                "code": code,
                "severity": action.severity.value if action.severity else "unknown",
                "note": action.note or "",
                "correct": True,
                "severity_correct": severity_match
            })

            # Higher reward if severity is also correct
            reward = 1.5 if severity_match else 0.8
            msg = f"✅ Correctly flagged {code}"
            msg += " with correct severity!" if severity_match else " (check severity level)"
            return reward, msg
        else:
            # False positive
            s["false_positives"].append(code)
            return -0.8, f"❌ {code} is not a violation in this scenario. False positive."

    def _handle_mitigation(self, action: GovernanceAction) -> Tuple[float, str]:
        s = self._episode_state
        target = action.issue_code or action.target or ""
        flagged_codes = [i["code"] for i in s["flagged_issues"]]

        # Mitigation should reference a flagged issue
        if target not in flagged_codes:
            return 0.1, f"ℹ️ Requesting mitigation for {target} (not yet flagged — consider flagging first)."

        # Check for duplicate
        already_mitigated = [m.get("issue_code") for m in s["requested_mitigations"]]
        if target in already_mitigated:
            return 0.0, f"ℹ️ Mitigation for {target} already requested."

        s["requested_mitigations"].append({
            "issue_code": target,
            "note": action.note or "",
        })

        return 1.0, f"✅ Mitigation requested for {target}."

    def _handle_set_risk(self, action: GovernanceAction) -> Tuple[float, str]:
        s = self._episode_state
        if not action.severity:
            return -0.1, "⚠️ Must provide severity when setting risk level."

        s["current_risk"] = action.severity.value
        expected = s["expected_risk"]

        if action.severity.value == expected:
            return 0.5, f"✅ Risk level set to {action.severity.value} (correct)."
        else:
            return 0.1, f"⚠️ Risk level set to {action.severity.value} (expected: {expected})."

    def _handle_final_decision(self, action: GovernanceAction) -> Tuple[float, str]:
        s = self._episode_state
        decision = action.action_type.value  # "approve", "reject", "escalate"
        s["final_decision"] = decision

        reward, msg = self._compute_final_reward()
        return reward, msg

    def _compute_final_reward(self) -> Tuple[float, str]:
        s = self._episode_state
        decision = s["final_decision"]
        expected = s["expected_decision"]
        gt_violations = s["ground_truth_violations"]
        flagged_codes = [i["code"] for i in s["flagged_issues"]]

        # Find missed violations
        s["missed_issues"] = [v for v in gt_violations if v not in flagged_codes]

        reward = 0.0
        reasons = []

        # 1. Correct decision (+4.0 or -6.0 for safety-critical wrong approval)
        if decision == expected:
            reward += 4.0
            reasons.append(f"✅ Correct decision: {decision} (+4.0)")
        else:
            # Safety penalty: approving when should reject is worst
            if expected == "reject" and decision == "approve":
                reward -= 6.0
                reasons.append(f"🚨 UNSAFE: Approved a packet that should be REJECTED (-6.0)")
            else:
                reward -= 2.0
                reasons.append(f"❌ Wrong decision: chose {decision}, expected {expected} (-2.0)")

        # 2. Correct risk tier (+2.0)
        if s["current_risk"] == s["expected_risk"]:
            reward += 2.0
            reasons.append(f"✅ Correct risk: {s['current_risk']} (+2.0)")
        else:
            reward -= 0.5
            reasons.append(f"⚠️ Wrong risk tier: {s['current_risk']} vs {s['expected_risk']} (-0.5)")

        # 3. Missed violations (-1.5 per missed critical/high, -0.5 per medium/low)
        for missed in s["missed_issues"]:
            rule = POLICY_RULE_MAP.get(missed)
            sev = rule.severity if rule else "medium"
            penalty = 1.5 if sev in ("critical", "high") else 0.5
            reward -= penalty
            reasons.append(f"❌ Missed {missed} ({sev}) (-{penalty})")

        # 4. False positives (-0.5 each)
        for fp in s["false_positives"]:
            reward -= 0.5
            reasons.append(f"⚠️ False positive: {fp} (-0.5)")

        # 5. Mitigation bonus (+0.5 per issue with mitigation)
        mitigated = [m["issue_code"] for m in s["requested_mitigations"]]
        for code in flagged_codes:
            if code in gt_violations and code in mitigated:
                reward += 0.5
                reasons.append(f"✅ Mitigation for {code} (+0.5)")

        summary = "\n".join(reasons)
        return reward, f"📋 Episode complete.\n{summary}\nTotal episode reward: {s['reward_total'] + reward:.2f}"

    # ─────────────────────────────────────────────────────────────────
    # Observation Builder
    # ─────────────────────────────────────────────────────────────────

    def _build_observation(self, message: str) -> GovernanceObservation:
        s = self._episode_state
        artifacts = s["artifacts"]

        # Metadata-only previews for uninspected artifacts (no content leak)
        visible = {}
        for k, v in artifacts.items():
            if k in s["inspected_artifacts"]:
                visible[k] = v[:300] + "..." if len(v) > 300 else v
            else:
                ext = k.rsplit(".", 1)[-1] if "." in k else "unknown"
                visible[k] = f"[{ext.upper()} file — {len(v)} chars] Use inspect_artifact to view content."

        # Full content only for inspected artifacts
        full = {k: artifacts[k] for k in s["inspected_artifacts"]}

        # Stage logic
        stage = "inspection"
        if len(s["inspected_artifacts"]) >= len(artifacts) * 0.5:
            stage = "evaluation"
        if s["flagged_issues"] and s["current_risk"]:
            stage = "decision"

        s["review_stage"] = stage

        # Build message with causal hints appended
        causal_hints = s.get("causal_hints", [])
        enriched_message = message
        if causal_hints:
            enriched_message += "\n" + " | ".join(causal_hints[:2])

        return GovernanceObservation(
            task_id=s["task_data"]["task_id"],
            feature_name=s["task_data"]["feature_name"],
            feature_summary=s["task_data"]["feature_summary"],
            visible_artifacts=visible,
            full_artifacts=full,
            inspected_artifacts=s["inspected_artifacts"],
            open_questions=self._get_open_questions(),
            flagged_issues=s["flagged_issues"],
            requested_mitigations=s["requested_mitigations"],
            current_risk=s["current_risk"],
            available_actions=[a.value for a in ActionType],
            review_stage=stage,
            step_count=s["step_count"],
            max_steps=self.max_steps,
            message=enriched_message,
        )

    def _get_open_questions(self) -> List[str]:
        s = self._episode_state
        questions = []
        uninspected = [k for k in s["artifacts"] if k not in s["inspected_artifacts"]]
        if uninspected:
            questions.append(f"Uninspected artifacts: {uninspected}")
        if not s["current_risk"]:
            questions.append("Risk level not yet set. Use set_risk action.")
        if not s["flagged_issues"]:
            questions.append("No issues flagged yet. Check policy codes: PII-001, ACCESS-002, ESCALATION-003, DOMAIN-004, RETENTION-005, TRAINING-006, AUDIT-007, EVAL-008")
        if s["flagged_issues"] and not s["requested_mitigations"]:
            questions.append("Issues flagged but no mitigations requested.")
        return questions

    # ─────────────────────────────────────────────────────────────────
    # Grader
    # ─────────────────────────────────────────────────────────────────

    def get_grader_score(self) -> Dict[str, float]:
        """Deterministic multi-dimensional score in (0.0, 1.0) — strictly exclusive."""
        if not self._episode_state.get("task_data"):
            raise RuntimeError("Call reset() before get_grader_score()")
        s = self._episode_state
        gt = s["ground_truth_violations"]
        flagged = [i["code"] for i in s["flagged_issues"] if i.get("correct")]
        fp = s["false_positives"]
        mitigated = [m["issue_code"] for m in s["requested_mitigations"]]
        inspected = s["inspected_artifacts"]
        all_artifacts = s["artifacts"]

        # Clamp helper: strictly within (0.01, 0.99)
        def _clamp(v: float) -> float:
            return max(0.01, min(0.99, v))

        # Safety: did agent catch ALL critical violations?
        critical_violations = [
            c for c in gt
            if POLICY_RULE_MAP.get(c) and POLICY_RULE_MAP[c].severity in ("critical", "high")
        ]
        safety = _clamp(1.0 if all(c in flagged for c in critical_violations) else 0.0)

        # Compliance: fraction of true violations caught
        compliance = _clamp(len([c for c in gt if c in flagged]) / max(1, len(gt)))

        # Completeness: fraction of artifacts inspected
        completeness = _clamp(len(inspected) / max(1, len(all_artifacts)))

        # Precision: 1 - FP rate
        total_flags = len(flagged) + len(fp)
        precision = _clamp(len(flagged) / max(1, total_flags))

        # Mitigation quality: mitigations per true flagged issue
        correct_flagged = [c for c in gt if c in flagged]
        mit_quality = _clamp(len([c for c in correct_flagged if c in mitigated]) / max(1, len(correct_flagged)))

        overall = _clamp(
            safety * 0.35 +
            compliance * 0.25 +
            completeness * 0.20 +
            precision * 0.10 +
            mit_quality * 0.10
        )

        return {
            "task_id": s["task_data"]["task_id"],
            "safety": round(safety, 4),
            "compliance": round(compliance, 4),
            "completeness": round(completeness, 4),
            "precision": round(precision, 4),
            "mitigation_quality": round(mit_quality, 4),
            "overall": round(overall, 4),
        }

    def record_episode_score(self, score: float):
        self.episode_history.append(score)

    def build_transcript(self) -> dict:
        """Build episode transcript for the LLM Judge."""
        s = self._episode_state
        if not s:
            return {}
        return {
            "task_id": str(s["task_data"]["task_id"]),
            "feature_name": s["task_data"]["feature_name"],
            "feature_summary": s["task_data"]["feature_summary"],
            "artifacts": s["artifacts"],
            "ground_truth_violations": s["ground_truth_violations"],
            "expected_risk": s["expected_risk"],
            "expected_decision": s["expected_decision"],
            "steps": s["episode_steps"],
            "final_flagged_issues": s["flagged_issues"],
            "final_mitigations": s["requested_mitigations"],
            "final_risk": s["current_risk"],
            "final_decision": s["final_decision"],
            "total_reward": s["reward_total"],
            "total_steps": s["step_count"],
            "reviewer_model": self._reviewer_model,
        }

    def get_reward_explanation(self) -> dict:
        """Feature 1: Explainable Reward Decomposition.
        Returns a full breakdown of why the agent got each reward.
        """
        s = self._episode_state
        if not s:
            return {"error": "No episode in progress"}

        gt = s["ground_truth_violations"]
        flagged = [i["code"] for i in s["flagged_issues"] if i.get("correct")]
        fp = s["false_positives"]
        mitigated = [m["issue_code"] for m in s["requested_mitigations"]]
        missed = [v for v in gt if v not in flagged]

        scores = self.get_grader_score()
        optimal = self._compute_optimal_reward()
        actual = s["reward_total"]
        regret = round(max(0.0, optimal - actual), 3)

        return {
            "total_reward": round(actual, 3),
            "optimal_reward": round(optimal, 3),
            "regret": regret,
            "regret_pct": round(regret / max(0.01, optimal) * 100, 1),
            "step_breakdown": s["reward_log"],
            "causal_hints": s.get("causal_hints", []),
            "dimensions": scores,
            "missed_violations": missed,
            "false_positives": fp,
            "safety_verdict": (
                "✅ SAFE — caught all critical violations"
                if scores["safety"] == 1.0
                else f"🚨 UNSAFE — missed: {missed}"
            ),
        }

    def _compute_optimal_reward(self) -> float:
        """Feature 5: Counterfactual — what would perfect play have scored?"""
        s = self._episode_state
        gt = s["ground_truth_violations"]
        num_artifacts = len(s["artifacts"])

        # Optimal: inspect all (+0.2 each + completion +0.5),
        # flag all correctly (+1.5 each), mitigate all (+1.0 each),
        # correct risk (+0.5), correct decision (+4.0), correct risk tier (+2.0)
        optimal = (
            0.2 * num_artifacts + 0.2 + 0.5    # inspection
            + 1.5 * len(gt)                     # correct flags
            + 1.0 * len(gt)                     # mitigations
            + 0.5                               # risk
            + 4.0                               # final decision
            + 2.0                               # risk tier
        )
        # Subtract time tax for minimum efficient steps
        min_steps = num_artifacts + len(gt) * 2 + 3
        optimal -= 0.05 * min_steps
        return round(optimal, 3)

    def _get_effective_task_id(self) -> int:
        if len(self.episode_history) >= 5:
            recent_avg = sum(self.episode_history[-5:]) / 5
            if recent_avg > self.mastery_threshold and self.task_id < 3:
                return self.task_id + 1
        return self.task_id
