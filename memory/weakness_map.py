"""
Weakness Map — tracks per-violation miss rates across episodes.

After N episodes, the weakness_map tells the adversarial maker
which violations the agent consistently fails to catch so it can
generate targeted tasks.

Format in agent_memory.json:
  "weakness_map": {
    "PII-001":        {"total": 10, "missed": 7, "miss_rate": 0.70},
    "ESCALATION-003": {"total": 8,  "missed": 2, "miss_rate": 0.25},
    ...
  }
"""

import json
import os
from collections import defaultdict
from typing import Dict, List

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "agent_memory.json")

ALL_CODES = ["PII-001", "ACCESS-002", "ESCALATION-003", "DOMAIN-004",
             "RETENTION-005", "TRAINING-006", "AUDIT-007", "EVAL-008"]


def load_weakness_map() -> Dict[str, dict]:
    """Load current weakness map from memory. Returns default if none."""
    if not os.path.exists(MEMORY_PATH):
        return _default_map()
    try:
        with open(MEMORY_PATH) as f:
            data = json.load(f)
        return data.get("weakness_map", _default_map())
    except Exception:
        return _default_map()


def _default_map() -> Dict[str, dict]:
    return {
        code: {"total": 0, "missed": 0, "miss_rate": 0.5}
        for code in ALL_CODES
    }


def update_weakness_map(
    ground_truth: List[str],
    caught: List[str],
    missed: List[str],
):
    """
    Update miss rates from one episode. Uses exponential moving average
    so recent performance weighs more than old episodes.
    """
    if not os.path.exists(MEMORY_PATH):
        return

    with open(MEMORY_PATH) as f:
        memory = json.load(f)

    wmap = memory.get("weakness_map", _default_map())
    alpha = 0.3  # EMA smoothing factor (higher = more weight on recent)

    for code in ground_truth:
        if code not in wmap:
            wmap[code] = {"total": 0, "missed": 0, "miss_rate": 0.5}
        wmap[code]["total"] += 1
        was_missed = code in missed
        if was_missed:
            wmap[code]["missed"] += 1

        # Exponential moving average of miss rate
        current = wmap[code]["miss_rate"]
        new_outcome = 1.0 if was_missed else 0.0
        wmap[code]["miss_rate"] = round(alpha * new_outcome + (1 - alpha) * current, 4)

    memory["weakness_map"] = wmap
    with open(MEMORY_PATH, "w") as f:
        json.dump(memory, f, indent=2)


def get_violation_weights() -> Dict[str, float]:
    """
    Return violation selection weights for adversarial maker.
    Higher miss_rate → more likely to be included in next task.
    Violations with zero episodes get moderate weight (0.5).
    """
    wmap = load_weakness_map()
    # Add small floor weight so even mastered violations sometimes appear
    weights = {
        code: max(0.15, data["miss_rate"])
        for code, data in wmap.items()
    }
    return weights


def get_weakness_summary() -> str:
    """Human-readable summary for orchestrator logging."""
    wmap = load_weakness_map()
    sorted_codes = sorted(wmap.items(), key=lambda x: -x[1]["miss_rate"])
    lines = ["=== Weakness Map ==="]
    for code, data in sorted_codes:
        bar = "█" * int(data["miss_rate"] * 10) + "░" * (10 - int(data["miss_rate"] * 10))
        lines.append(
            f"  {code:<16} {bar} {data['miss_rate']:.0%} miss "
            f"({data['missed']}/{data['total']} episodes)"
        )
    return "\n".join(lines)
