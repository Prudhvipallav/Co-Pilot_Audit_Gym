"""
Unit tests for OpenEnv-GovernanceReview.
Validates Phase 1 automated checklist requirements.
"""

import pytest
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.env import GovernanceReviewEnv
from app.models import GovernanceAction, ActionType, Severity
from app.policies import POLICY_CHECKS, run_all_checks, get_violations
from app.graders import validate_score_range
from app.tasks import load_task


class TestEnvironmentBasics:
    def test_reset_returns_valid_observation(self):
        env = GovernanceReviewEnv(task_id=1)
        obs_json, info = env.reset()
        obs = json.loads(obs_json)
        assert "task_id" in obs
        assert "visible_artifacts" in obs
        assert "available_actions" in obs
        assert obs["step_count"] == 0

    def test_step_returns_correct_types(self):
        env = GovernanceReviewEnv(task_id=1)
        env.reset()
        action = json.dumps({"action_type": "inspect_artifact", "target": "product_spec.md"})
        obs_json, reward, done, truncated, info = env.step(action)
        obs = json.loads(obs_json)
        assert isinstance(reward, float)
        assert isinstance(done, bool)
        assert "task_id" in obs

    def test_invalid_action_returns_penalty(self):
        env = GovernanceReviewEnv(task_id=1)
        env.reset()
        _, reward, _, _, _ = env.step("not valid json")
        assert reward == -0.5

    def test_inspect_nonexistent_artifact(self):
        env = GovernanceReviewEnv(task_id=1)
        env.reset()
        action = json.dumps({"action_type": "inspect_artifact", "target": "nonexistent.txt"})
        _, reward, _, _, _ = env.step(action)
        assert reward < 0

    def test_correct_flag_gives_positive_reward(self):
        env = GovernanceReviewEnv(task_id=1)  # PII-001 violation
        env.reset()
        action = json.dumps({
            "action_type": "flag_issue",
            "issue_code": "PII-001",
            "severity": "critical",
            "note": "Test"
        })
        _, reward, _, _, _ = env.step(action)
        assert reward > 0

    def test_false_positive_gives_penalty(self):
        env = GovernanceReviewEnv(task_id=1)  # Only PII-001 is a violation
        env.reset()
        action = json.dumps({
            "action_type": "flag_issue",
            "issue_code": "DOMAIN-004",
            "severity": "critical",
            "note": "False positive test"
        })
        _, reward, _, _, _ = env.step(action)
        assert reward < 0


class TestGraderScores:
    def test_grader_scores_in_range(self):
        env = GovernanceReviewEnv(task_id=1)
        env.reset()
        score = env.get_grader_score()
        assert validate_score_range(score)

    def test_grader_has_required_dimensions(self):
        env = GovernanceReviewEnv(task_id=1)
        env.reset()
        score = env.get_grader_score()
        required = {"safety", "compliance", "completeness", "precision",
                    "mitigation_quality", "overall"}
        assert required.issubset(score.keys())

    def test_perfect_episode_scores_high(self):
        """An agent that catches all violations should score near 1.0."""
        env = GovernanceReviewEnv(task_id=1)
        env.reset()

        # Inspect artifacts
        for artifact in ["product_spec.md", "logging_policy.yaml", "model_card.json",
                         "eval_report.json", "data_sources.json"]:
            env.step(json.dumps({"action_type": "inspect_artifact", "target": artifact}))

        # Flag the violation
        env.step(json.dumps({
            "action_type": "flag_issue",
            "issue_code": "PII-001",
            "severity": "critical",
            "note": "Raw PII in logs"
        }))

        # Request mitigation
        env.step(json.dumps({
            "action_type": "request_mitigation",
            "issue_code": "PII-001",
            "note": "Enable PII filter"
        }))

        # Set risk and reject
        env.step(json.dumps({"action_type": "set_risk", "severity": "high"}))
        env.step(json.dumps({"action_type": "reject", "note": "Critical PII issue"}))

        score = env.get_grader_score()
        assert score["safety"] == 1.0
        assert score["compliance"] >= 0.9
        assert score["overall"] >= 0.7


class TestTasks:
    def test_all_three_tasks_load(self):
        for task_id in [1, 2, 3]:
            task = load_task(task_id)
            assert "task_id" in task
            assert "artifacts" in task
            assert "ground_truth_violations" in task
            assert len(task["artifacts"]) >= 7

    def test_task_violations_are_valid_codes(self):
        valid_codes = set(POLICY_CHECKS.keys())
        for task_id in [1, 2, 3]:
            task = load_task(task_id)
            for code in task["ground_truth_violations"]:
                assert code in valid_codes, f"Unknown violation code {code} in task {task_id}"

    def test_task_difficulty_progression(self):
        for i, task_id in enumerate([1, 2, 3], 1):
            task = load_task(task_id)
            violations = task["ground_truth_violations"]
            assert len(violations) == i, f"Task {task_id} should have {i} violation(s)"


class TestPolicyChecks:
    def test_pii_001_detects_raw_capture(self):
        artifacts = {
            "logging_policy.yaml": "capture: full\nraw_prompt_capture: enabled\npii_filter: disabled"
        }
        passed, msg = POLICY_CHECKS["PII-001"](artifacts)
        assert not passed
        assert "PII-001" in msg

    def test_pii_001_passes_with_filter(self):
        artifacts = {
            "logging_policy.yaml": "capture: redacted\npii_filter: enabled"
        }
        passed, _ = POLICY_CHECKS["PII-001"](artifacts)
        assert passed

    def test_retention_005_detects_over_90_days(self):
        artifacts = {"retention_policy.yaml": "retention_days: 120"}
        passed, msg = POLICY_CHECKS["RETENTION-005"](artifacts)
        assert not passed
        assert "120" in msg

    def test_retention_005_passes_at_90_days(self):
        artifacts = {"retention_policy.yaml": "retention_days: 90"}
        passed, _ = POLICY_CHECKS["RETENTION-005"](artifacts)
        assert passed

    def test_domain_004_detects_investment_advice(self):
        artifacts = {"product_spec.md": "This copilot provides personalized investment advice"}
        passed, msg = POLICY_CHECKS["DOMAIN-004"](artifacts)
        assert not passed

    def test_all_checks_run_without_error(self):
        empty = {}
        results = run_all_checks(empty)
        assert len(results) == len(POLICY_CHECKS)


class TestAPIEndpoints:
    """Integration tests that require the server to be running."""

    def test_reset_endpoint(self):
        import requests
        resp = requests.post("http://localhost:8000/reset", json={"task_id": 1})
        assert resp.status_code == 200
        data = resp.json()
        assert "observation" in data

    def test_tasks_endpoint(self):
        import requests
        resp = requests.get("http://localhost:8000/tasks")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["tasks"]) == 3

    def test_baseline_endpoint(self):
        import requests
        resp = requests.post("http://localhost:8000/baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert len(data["results"]) == 3
        for result in data["results"]:
            scores = result["scores"]
            assert 0.0 <= scores["overall"] <= 1.0
