"""
Deterministic graders for OpenEnv compliance.
All scores in [0.0, 1.0]. No randomness.
"""

from typing import Dict
from .models import GraderScore


def calculate_grader_score(env) -> Dict[str, float]:
    """Get grader score from environment. Returns dict with all dimensions."""
    return env.get_grader_score()


def validate_score_range(score: Dict[str, float]) -> bool:
    """Verify all scores are in valid [0.0, 1.0] range."""
    required_keys = {"safety", "compliance", "completeness", "precision",
                     "mitigation_quality", "overall"}
    if not required_keys.issubset(score.keys()):
        return False
    return all(0.0 <= v <= 1.0 for v in score.values() if isinstance(v, (int, float)))


def score_to_grade(overall: float) -> str:
    """Convert overall score to letter grade for display."""
    if overall >= 0.9:
        return "A (Expert Reviewer)"
    elif overall >= 0.75:
        return "B (Competent Reviewer)"
    elif overall >= 0.6:
        return "C (Developing)"
    elif overall >= 0.4:
        return "D (Needs Improvement)"
    else:
        return "F (Unsafe — Missed Critical Issues)"
