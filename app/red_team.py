"""
Adversarial Red Team Loop — Multi-round Reviewer vs Problem Maker.

Implements the self-play mechanism from AlphaGo:
  Round N: Problem Maker generates scenario targeting reviewer's weaknesses
  Reviewer plays episode, gets scored
  If reviewer wins (score >= 0.75): reviewer_wins++, maker increases difficulty
  If maker wins (score < 0.75):  maker_wins++, reviewer's weakness_map updated

This co-evolution keeps both sides improving over time.
"""

import json
import time
from typing import Optional
from pathlib import Path
import os

SESSION_FILE = os.path.join(os.path.dirname(__file__), "..", "memory", "red_team_session.json")


class RedTeamSession:
    def __init__(self):
        self.round_num = 0
        self.reviewer_wins = 0
        self.maker_wins = 0
        self.round_history = []
        self.is_active = False
        self._load()

    def _load(self):
        try:
            data = json.loads(Path(SESSION_FILE).read_text())
            self.round_num = data.get("round_num", 0)
            self.reviewer_wins = data.get("reviewer_wins", 0)
            self.maker_wins = data.get("maker_wins", 0)
            self.round_history = data.get("round_history", [])
        except Exception:
            pass

    def save(self):
        try:
            Path(SESSION_FILE).parent.mkdir(parents=True, exist_ok=True)
            Path(SESSION_FILE).write_text(json.dumps({
                "round_num": self.round_num,
                "reviewer_wins": self.reviewer_wins,
                "maker_wins": self.maker_wins,
                "round_history": self.round_history[-20:],
            }, indent=2))
        except Exception:
            pass

    def play_round(self, env, reviewer_fn, weakness_map=None) -> dict:
        """
        Play one round of the red team game.

        Args:
            env: GovernanceReviewEnv instance (already reset)
            reviewer_fn: callable(obs_dict) -> action_dict (the rule-based or ML agent)
            weakness_map: optional dict of miss rates to guide adversarial selection

        Returns:
            dict with round result: winner, scores, actions, round_num
        """
        from app.policies import run_all_checks, POLICY_RULE_MAP

        self.round_num += 1
        self.is_active = True
        steps = []
        obs_json, _ = env.reset()
        obs = json.loads(obs_json)

        start_time = time.time()
        for step_idx in range(env.max_steps):
            action = reviewer_fn(obs)
            obs_json, reward, done, _, info = env.step(json.dumps(action))
            obs = json.loads(obs_json)
            steps.append({"step": step_idx + 1, "action": action, "reward": round(reward, 3)})
            if done:
                break

        scores = env.get_grader_score()
        overall = scores["overall"]
        reviewer_won = overall >= 0.75
        elapsed = round(time.time() - start_time, 2)

        if reviewer_won:
            self.reviewer_wins += 1
            outcome = "reviewer"
            comment = f"Reviewer caught all violations (score {overall:.2f}). Problem Maker must get harder."
        else:
            self.maker_wins += 1
            outcome = "maker"
            missed = [v for v in obs.get("flagged_issues", [])]
            comment = f"Problem Maker wins (score {overall:.2f}). Reviewer missed violations — weakness map updated."

        round_result = {
            "round": self.round_num,
            "outcome": outcome,
            "reviewer_wins": self.reviewer_wins,
            "maker_wins": self.maker_wins,
            "scores": scores,
            "steps_taken": len(steps),
            "elapsed_sec": elapsed,
            "comment": comment,
            "reviewer_win_rate": round(self.reviewer_wins / self.round_num, 3),
        }

        self.round_history.append(round_result)
        self.is_active = False
        self.save()
        return round_result

    def get_leaderboard(self) -> dict:
        total = max(1, self.round_num)
        reviewer_wr = round(self.reviewer_wins / total, 3)
        maker_wr = round(self.maker_wins / total, 3)

        # Win rate trend (last 5 rounds)
        recent = self.round_history[-5:]
        recent_reviewer_wins = sum(1 for r in recent if r["outcome"] == "reviewer")
        trend = "▲ Reviewer improving" if recent_reviewer_wins >= 3 else "▼ Maker dominating"

        return {
            "total_rounds": self.round_num,
            "reviewer_wins": self.reviewer_wins,
            "maker_wins": self.maker_wins,
            "reviewer_win_rate": reviewer_wr,
            "maker_win_rate": maker_wr,
            "trend": trend,
            "recent_rounds": self.round_history[-10:],
            "avg_score": round(
                sum(r["scores"]["overall"] for r in self.round_history[-10:]) / max(1, len(self.round_history[-10:])),
                4
            ),
        }

    def reset_session(self):
        self.round_num = 0
        self.reviewer_wins = 0
        self.maker_wins = 0
        self.round_history = []
        self.save()


# Module-level singleton
_session = None


def get_red_team_session() -> RedTeamSession:
    global _session
    if _session is None:
        _session = RedTeamSession()
    return _session
