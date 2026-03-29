"""
ELO-Based Adaptive Difficulty Tracker for CopilotAudit-Gym.

Tracks agent performance using ELO chess ratings and selects 
tasks slightly above the agent's current level — the same mechanism
used by chess.com and AlphaGo to guarantee optimal learning.

ELO ranges:
  800-1100: Task 1 (Easy)   — 1 violation, obvious artifact
  1100-1400: Task 2 (Medium) — 2 violations, cross-artifact detection
  1400+:     Task 3 (Hard)   — 3 violations, domain knowledge required
"""

import json
import os
from pathlib import Path

ELO_FILE = os.path.join(os.path.dirname(__file__), "..", "memory", "elo_state.json")

# ELO task thresholds
ELO_EASY_MAX = 1100
ELO_MEDIUM_MAX = 1400
ELO_STARTING = 1000
K_FACTOR = 32  # Standard chess K-factor


class EloTracker:
    def __init__(self):
        self.elo = ELO_STARTING
        self.round_history: list = []
        self._load()

    def _load(self):
        try:
            data = json.loads(Path(ELO_FILE).read_text())
            self.elo = data.get("elo", ELO_STARTING)
            self.round_history = data.get("round_history", [])
        except Exception:
            self.elo = ELO_STARTING
            self.round_history = []

    def save(self):
        try:
            Path(ELO_FILE).parent.mkdir(parents=True, exist_ok=True)
            Path(ELO_FILE).write_text(json.dumps({
                "elo": round(self.elo, 1),
                "round_history": self.round_history[-50:],  # Keep last 50
            }, indent=2))
        except Exception:
            pass

    def update(self, overall_score: float):
        """
        Update ELO based on episode outcome.
        Score >= 0.75 → reviewer wins; < 0.75 → maker wins.
        Uses standard ELO formula with task-calibrated opponent rating.
        """
        won = overall_score >= 0.75
        task_id = self.select_task_id()

        # Opponent rating = midpoint of the selected task's ELO range
        task_elo_map = {1: 950, 2: 1250, 3: 1550}
        opponent_elo = task_elo_map.get(task_id, 1000)

        expected = 1 / (1 + 10 ** ((opponent_elo - self.elo) / 400))
        actual = 1.0 if won else 0.0
        self.elo += K_FACTOR * (actual - expected)
        self.elo = max(400.0, self.elo)  # Floor

        self.round_history.append({
            "elo_before": round(self.elo - K_FACTOR * (actual - expected), 1),
            "elo_after": round(self.elo, 1),
            "score": round(overall_score, 4),
            "task_id": task_id,
            "won": won,
        })
        self.save()
        return round(self.elo, 1)

    def select_task_id(self) -> int:
        """Select task slightly above current level — optimal learning zone."""
        # Target ELO slightly above current to keep it challenging
        target = self.elo + 100

        if target < ELO_EASY_MAX:
            return 1
        elif target < ELO_MEDIUM_MAX:
            return 2
        else:
            return 3

    def get_grade(self) -> str:
        if self.elo >= 1600:
            return "Expert 🏆"
        elif self.elo >= 1400:
            return "Advanced ⭐"
        elif self.elo >= 1200:
            return "Proficient 📈"
        elif self.elo >= 1000:
            return "Developing 📚"
        else:
            return "Beginner 🌱"

    def to_dict(self) -> dict:
        return {
            "elo": round(self.elo, 1),
            "grade": self.get_grade(),
            "recommended_task": self.select_task_id(),
            "round_history": self.round_history[-10:],
        }


# Module-level singleton
_tracker = None


def get_elo_tracker() -> EloTracker:
    global _tracker
    if _tracker is None:
        _tracker = EloTracker()
    return _tracker
