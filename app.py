"""
🏛️ CopilotAudit-Gym PRO v3.1 — Interactive RL Demo
You are the AI governance reviewer. Inspect artifacts, flag violations, and decide.

Run:  python app.py
"""

import gradio as gr
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.env import GovernanceReviewEnv
from app.policies import POLICY_RULES, POLICY_RULE_MAP, SECTOR_CONFIGS, run_all_checks, VIOLATION_CAUSES
from app.graders import score_to_grade
from app.elo import get_elo_tracker
from app.red_team import get_red_team_session

# ═══════════════════════════════════════════════════════════════
#  CSS — Soft light theme inspired by EcomRLVE-Gym
# ═══════════════════════════════════════════════════════════════

CSS = """
/* Base */
body, .gradio-container { background: #F0F4FA !important; font-family: 'Inter', system-ui, sans-serif !important; }
.gradio-container { max-width: 1440px !important; }
footer { display: none !important; }

/* Groups & Cards */
.gr-group, .gr-box, .gr-panel, .gr-form {
    background: #FFFFFF !important; border: 1px solid #E2E8F0 !important;
    border-radius: 14px !important; box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
}

/* Tabs */
.gr-tab-item { color: #64748B !important; border: none !important; font-weight: 500 !important; }
.gr-tab-item.selected { color: #4F46E5 !important; border-bottom: 2px solid #4F46E5 !important; background: transparent !important; }

/* Inputs */
.gr-input, .gr-dropdown, .gr-textbox textarea, select, input[type=text] {
    background: #FAFBFF !important; color: #1E293B !important;
    border: 1px solid #E2E8F0 !important; border-radius: 10px !important;
}
label span { color: #475569 !important; font-weight: 500 !important; }

/* Primary Buttons — Indigo */
.gr-button-primary {
    background: #4F46E5 !important; border: none !important; color: #fff !important;
    font-weight: 700 !important; border-radius: 10px !important; font-size: 1.05em !important;
    box-shadow: 0 2px 8px rgba(79,70,229,0.25) !important;
}
.gr-button-primary:hover { background: #4338CA !important; box-shadow: 0 4px 16px rgba(79,70,229,0.35) !important; }

/* Secondary Buttons */
.gr-button-secondary {
    background: #F1F5F9 !important; border: 1px solid #E2E8F0 !important;
    color: #475569 !important; border-radius: 10px !important;
}

/* Accordions */
.gr-accordion { background: #FFFFFF !important; border: 1px solid #E2E8F0 !important; border-radius: 12px !important; }

/* Markdown in dark panels */
.prose h1, .prose h2, .prose h3, .prose h4, .prose p, .prose li { color: #1E293B !important; }
.prose code { background: #F1F5F9 !important; color: #4F46E5 !important; padding: 2px 6px; border-radius: 4px; }
.prose a { color: #4F46E5 !important; }

/* Badge label helper */
.badge { display: inline-block; padding: 4px 14px; border-radius: 8px; font-size: 0.82em; font-weight: 600; }
.badge-indigo { background: #EEF2FF; color: #4F46E5; }
.badge-green { background: #ECFDF5; color: #059669; }
.badge-red { background: #FEF2F2; color: #DC2626; }
.badge-amber { background: #FFFBEB; color: #D97706; }
.badge-blue { background: #EFF6FF; color: #2563EB; }

/* Score cards */
#score-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
"""


# ═══════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ═══════════════════════════════════════════════════════════════

def _badge(text, color="indigo"):
    return f'<span class="badge badge-{color}">{text}</span>'


def _env_state(obs):
    if not obs:
        return '<div style="text-align:center;padding:80px 20px;color:#94A3B8;font-size:0.95em;">Click <b>🔄 Reset Episode</b> to start a governance review.</div>'

    feature = obs.get("feature_name", "—")
    risk = (obs.get("current_risk") or "not set").upper()
    stage = obs.get("review_stage", "inspection")
    msg = obs.get("message", "")
    inspected = obs.get("inspected_artifacts", [])
    visible = obs.get("visible_artifacts", {})
    flagged = obs.get("flagged_issues", [])
    mitigations = obs.get("requested_mitigations", [])
    step = obs.get("step_count", 0)
    max_s = obs.get("max_steps", 50)
    total = max(1, len(visible))
    pct = int(len(inspected) / total * 100)

    risk_badge = {"CRITICAL": "red", "HIGH": "red", "MEDIUM": "amber", "LOW": "green", "NOT SET": "blue"}.get(risk, "blue")

    # Artifact chips
    arts = ""
    for name in visible:
        done = name in inspected
        bg = "#ECFDF5" if done else "#F8FAFC"
        bc = "#86EFAC" if done else "#E2E8F0"
        tc = "#065F46" if done else "#64748B"
        icon = "✓" if done else "○"
        arts += f'<span style="display:inline-block;background:{bg};color:{tc};padding:4px 12px;border-radius:8px;font-size:0.78em;border:1px solid {bc};margin:3px;">{icon} {name}</span>'

    # Flags
    flags = ""
    if flagged:
        for f in flagged:
            sev = f.get("severity", "?").upper()
            sc = {"CRITICAL": "#DC2626", "HIGH": "#EA580C", "MEDIUM": "#D97706"}.get(sev, "#64748B")
            flags += f'''<div style="background:#FEF2F2;border:1px solid #FECACA;border-left:3px solid {sc};
                border-radius:10px;padding:10px 14px;margin:6px 0;">
                <span style="color:{sc};font-weight:700;font-size:0.82em;">{f.get('code','?')}</span>
                <span style="float:right;font-size:0.65em;background:{sc}15;color:{sc};padding:2px 8px;border-radius:4px;">{sev}</span>
                <div style="color:#71717A;font-size:0.75em;margin-top:4px;">{f.get('note','')[:120]}</div>
            </div>'''

    return f'''
<div style="padding:4px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:14px;">
    <div>
      <div style="font-size:1.15em;font-weight:700;color:#1E293B;">🏢 {feature}</div>
      <div style="color:#94A3B8;font-size:0.78em;margin-top:2px;">Step {step}/{max_s} · {stage.title()}</div>
    </div>
    {_badge(risk, risk_badge)}
  </div>

  <div style="background:#F1F5F9;border-radius:10px;height:8px;overflow:hidden;margin-bottom:4px;">
    <div style="background:linear-gradient(90deg,#4F46E5,#7C3AED);width:{pct}%;height:100%;border-radius:10px;transition:width 0.5s;"></div>
  </div>
  <div style="display:flex;justify-content:space-between;font-size:0.72em;color:#94A3B8;margin-bottom:16px;">
    <span>Artifacts inspected</span><span>{len(inspected)}/{len(visible)} ({pct}%)</span>
  </div>

  <div style="background:#F8FAFC;padding:12px 16px;border-radius:10px;border-left:3px solid #4F46E5;margin-bottom:14px;">
    <div style="color:#64748B;font-size:0.82em;font-style:italic;">"{msg[:280]}"</div>
  </div>

  <div style="margin-bottom:10px;">{arts}</div>
  {f'<div style="margin-top:12px;"><div style="color:#DC2626;font-weight:600;font-size:0.82em;margin-bottom:6px;">🚩 Flagged Violations</div>{flags}</div>' if flags else ''}
  {f'<div style="margin-top:10px;color:#64748B;font-size:0.75em;">🛡️ Mitigations requested: {len(mitigations)}</div>' if mitigations else ''}
</div>'''


def _action_card(step_data):
    a = step_data["action"]
    r = step_data["reward"]
    atype = a.get("action_type", "?")
    icons = {"inspect_artifact": "🔍", "flag_issue": "🚩", "request_mitigation": "🛡️",
             "set_risk": "⚡", "approve": "✅", "reject": "🚫", "escalate": "⚠️"}
    icon = icons.get(atype, "▶")
    rc = "#059669" if r > 0 else "#DC2626" if r < 0 else "#94A3B8"
    bg = "#ECFDF5" if r > 0 else "#FEF2F2" if r < 0 else "#F8FAFC"
    detail = a.get("target") or a.get("issue_code") or ""
    note = a.get("note", "")

    return f'''<div style="background:{bg};border:1px solid #E2E8F0;border-left:3px solid {rc};
        border-radius:10px;padding:12px 16px;margin:6px 0;">
      <div style="display:flex;justify-content:space-between;align-items:center;">
        <div><span style="font-weight:700;color:{rc};font-size:0.85em;">{icon} {atype.upper()}</span>
             <span style="color:#94A3B8;font-size:0.72em;margin-left:8px;">Step {step_data['step']}</span></div>
        <span style="font-family:monospace;font-weight:700;color:{rc};font-size:1em;">{r:+.2f}</span>
      </div>
      {'<div style="color:#64748B;font-size:0.78em;margin-top:4px;">→ <code style=background:#F1F5F9;padding:2px 6px;border-radius:4px;color:#4F46E5;>' + detail + '</code></div>' if detail else ''}
      {'<div style="color:#94A3B8;font-size:0.72em;margin-top:3px;font-style:italic;">' + note[:120] + '</div>' if note else ''}
    </div>'''


def _scores_html(scores, grade):
    if not scores:
        return '<div style="color:#94A3B8;text-align:center;padding:40px;">Complete a review to see scores.</div>'

    overall = scores.get("overall", 0)
    gc = "#059669" if overall >= 0.75 else "#D97706" if overall >= 0.5 else "#DC2626"

    dims = [("Safety", scores.get("safety", 0), "#DC2626", "35%"),
            ("Compliance", scores.get("compliance", 0), "#D97706", "25%"),
            ("Completeness", scores.get("completeness", 0), "#2563EB", "20%"),
            ("Precision", scores.get("precision", 0), "#7C3AED", "10%"),
            ("Mitigation", scores.get("mitigation_quality", 0), "#059669", "10%")]

    bars = ""
    for name, val, color, weight in dims:
        bars += f'''<div style="margin:10px 0;">
          <div style="display:flex;justify-content:space-between;margin-bottom:3px;">
            <span style="color:#64748B;font-size:0.82em;">{name} <span style="color:#CBD5E1;">({weight})</span></span>
            <span style="color:#1E293B;font-weight:600;font-family:monospace;">{val:.2f}</span>
          </div>
          <div style="background:#F1F5F9;border-radius:6px;height:10px;overflow:hidden;">
            <div style="background:{color};width:{val*100}%;height:100%;border-radius:6px;transition:width 0.6s;"></div>
          </div>
        </div>'''

    return f'''
<div style="text-align:center;margin-bottom:20px;">
  <div style="font-size:3.5em;font-weight:800;color:{gc};font-family:monospace;">{overall:.2f}</div>
  <div style="color:{gc};font-weight:600;font-size:1em;margin-top:2px;">{grade}</div>
</div>
{bars}'''


def _policy_ref_md():
    lines = []
    for r in POLICY_RULES:
        sev_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡"}.get(r.severity, "⚪")
        lines.append(f"**`{r.code}`** {sev_emoji} {r.severity.upper()} — **{r.name}**")
        lines.append(f"  {r.description}")
        lines.append(f"  📄 Check: `{r.artifact_hint}`\n")
    return "\n".join(lines)


def _reward_ref_md():
    return """| Signal | Reward | When |
|---|---|---|
| Inspect new artifact | **+0.2** | Each new file opened |
| Correct flag + severity | **+1.5** | True violation, right severity |
| Correct flag, wrong sev | **+0.8** | True violation, wrong severity |
| False positive | **−0.8** | Flagging non-violation |
| Request mitigation | **+1.0** | For a flagged issue |
| Set correct risk | **+0.5** | Risk matches worst severity |
| Correct final decision | **+4.0** | approve/reject/escalate correct |
| Unsafe approval | **−6.0** | Approving with critical violations |
| Wrong decision | **−2.0** | Incorrect non-unsafe decision |
| Time tax / step | **−0.05** | Encourages efficiency |
| Timeout | **−3.0** | No decision by max steps |"""


def _elo_card_html(tracker=None):
    """ELO rating card for the sidebar."""
    if tracker is None:
        try:
            tracker = get_elo_tracker()
        except Exception:
            return ''
    elo = round(tracker.elo, 0)
    grade = tracker.get_grade()
    task = tracker.select_task_id()
    task_names = {1: "Easy", 2: "Medium", 3: "Hard"}
    # ELO bar: range 400-2000
    pct = min(100, max(0, int((elo - 400) / 16)))
    color = "#059669" if elo >= 1400 else "#D97706" if elo >= 1100 else "#4F46E5"
    return f'''<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;padding:14px;margin-top:12px;">
  <div style="font-size:0.75em;font-weight:700;color:#64748B;letter-spacing:.05em;margin-bottom:8px;">⚡ ELO RATING</div>
  <div style="font-size:2.2em;font-weight:800;color:{color};font-family:monospace;">{int(elo)}</div>
  <div style="color:#94A3B8;font-size:0.75em;margin-bottom:8px;">{grade}</div>
  <div style="background:#E2E8F0;border-radius:6px;height:6px;overflow:hidden;margin-bottom:8px;">
    <div style="background:{color};width:{pct}%;height:100%;border-radius:6px;"></div>
  </div>
  <div style="font-size:0.72em;color:#64748B;">Recommended: <b>Task {task} ({task_names[task]})</b></div>
</div>'''


def _explain_html(env_obj):
    """Render the explainable reward breakdown after an episode."""
    try:
        ex = env_obj.get_reward_explanation()
    except Exception:
        return ''
    if "error" in ex:
        return ''

    verdict = ex.get("safety_verdict", "")
    is_safe = "SAFE" in verdict and "UNSAFE" not in verdict
    vc = "#059669" if is_safe else "#DC2626"
    vbg = "#ECFDF5" if is_safe else "#FEF2F2"

    regret = ex.get("regret_pct", 0)
    actual = ex.get("total_reward", 0)
    optimal = ex.get("optimal_reward", 1)
    regc = "#059669" if regret < 20 else "#D97706" if regret < 50 else "#DC2626"

    steps_html = ""
    for s in ex.get("step_breakdown", []):
        r = s["reward"]
        rc = "#059669" if r > 0 else "#DC2626" if r < 0 else "#94A3B8"
        steps_html += f'''<div style="display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #F1F5F9;font-size:0.75em;">
          <span style="color:#64748B;">Step {s['step']} — {s['action_type']}
            {f"<code style='background:#F1F5F9;padding:1px 5px;border-radius:3px;color:#4F46E5;'>" + s['detail'] + "</code>" if s.get('detail') else ''}
          </span>
          <span style="font-family:monospace;font-weight:700;color:{rc};">{r:+.3f}</span>
        </div>'''

    hints_html = ""
    for h in ex.get("causal_hints", []):
        hints_html += f'<div style="color:#D97706;font-size:0.75em;padding:4px 0;">🔗 {h}</div>'

    return f'''
<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:16px;margin-top:12px;">
  <div style="font-weight:700;color:#1E293B;margin-bottom:12px;">🔬 Reward Explanation</div>
  <div style="background:{vbg};border:1px solid {vc}30;border-radius:10px;padding:10px 14px;margin-bottom:12px;">
    <span style="color:{vc};font-weight:700;font-size:0.85em;">{verdict}</span>
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;margin-bottom:12px;">
    <div style="text-align:center;background:#F8FAFC;padding:10px;border-radius:10px;">
      <div style="font-size:1.4em;font-weight:800;color:#1E293B;font-family:monospace;">{actual:.2f}</div>
      <div style="color:#94A3B8;font-size:0.7em;">Actual Reward</div>
    </div>
    <div style="text-align:center;background:#F8FAFC;padding:10px;border-radius:10px;">
      <div style="font-size:1.4em;font-weight:800;color:#059669;font-family:monospace;">{optimal:.2f}</div>
      <div style="color:#94A3B8;font-size:0.7em;">Optimal Reward</div>
    </div>
    <div style="text-align:center;background:#F8FAFC;padding:10px;border-radius:10px;">
      <div style="font-size:1.4em;font-weight:800;color:{regc};font-family:monospace;">{regret:.1f}%</div>
      <div style="color:#94A3B8;font-size:0.7em;">Regret</div>
    </div>
  </div>
  {f'<div style="margin-bottom:10px;">{hints_html}</div>' if hints_html else ''}
  <div style="max-height:200px;overflow-y:auto;">{steps_html}</div>
</div>'''


# ═══════════════════════════════════════════════════════════════
#  CORE LOGIC
# ═══════════════════════════════════════════════════════════════

def _plan_action(obs):
    """Rule-based governance reviewer agent."""
    inspected = obs.get("inspected_artifacts", [])
    artifacts = list(obs.get("visible_artifacts", {}).keys())
    flagged = obs.get("flagged_issues", [])
    flagged_codes = [i.get("code", "") for i in flagged]
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
                        "target": rule.artifact_hint if rule else "", "note": msg[:120]}
    except Exception:
        pass

    for issue in flagged:
        code = issue.get("code", "")
        if code and code not in mitigated:
            return {"action_type": "request_mitigation", "issue_code": code,
                    "note": f"Remediate {code} before launch"}

    if flagged and not risk:
        sevs = [i.get("severity", "medium") for i in flagged]
        sv = "critical" if "critical" in sevs else "high" if "high" in sevs else "medium"
        return {"action_type": "set_risk", "severity": sv}

    if flagged:
        return {"action_type": "reject" if risk in ("critical", "high") else "escalate",
                "note": f"Violations: {flagged_codes}"}
    return {"action_type": "approve", "note": "No policy violations found"}


# ── Automated benchmark ─────────────────────────────────────
def run_auto(task_id):
    env = GovernanceReviewEnv(task_id=int(task_id))
    obs_json, _ = env.reset()
    obs = json.loads(obs_json)
    steps = []

    for i in range(20):
        action = _plan_action(obs)
        obs_json, reward, done, _, _ = env.step(json.dumps(action))
        obs = json.loads(obs_json)
        steps.append({"step": i + 1, "action": action, "reward": reward})

        feed = "".join(_action_card(s) for s in reversed(steps))
        if done:
            sc = env.get_grader_score()
            g = score_to_grade(sc["overall"])
            # Update ELO
            try:
                get_elo_tracker().update(sc["overall"])
            except Exception:
                pass
            yield _env_state(obs), feed, _scores_html(sc, g), _explain_html(env)
            return
        yield _env_state(obs), feed, "", ""
        time.sleep(0.4)

    sc = env.get_grader_score()
    g = score_to_grade(sc["overall"])
    try:
        get_elo_tracker().update(sc["overall"])
    except Exception:
        pass
    yield _env_state(obs), "".join(_action_card(s) for s in reversed(steps)), _scores_html(sc, g), _explain_html(env)


# ── Manual mode ──────────────────────────────────────────────
def manual_reset(task_id):
    env = GovernanceReviewEnv(task_id=int(task_id))
    obs_json, _ = env.reset()
    obs = json.loads(obs_json)
    targets = list(obs.get("visible_artifacts", {}).keys())
    return env, [], _env_state(obs), "", "", gr.update(choices=targets, value=targets[0] if targets else None)


def manual_step(env, history, action_type, target, issue_code, severity, note):
    if env is None:
        return env, history, '<div style="color:#DC2626;padding:20px;">Click Reset first!</div>', "", "", ""

    action = {"action_type": action_type}
    if action_type == "inspect_artifact" and target:
        action["target"] = target
    if action_type == "flag_issue":
        if issue_code: action["issue_code"] = issue_code
        if severity: action["severity"] = severity
        if target: action["target"] = target
    if action_type == "request_mitigation" and issue_code:
        action["issue_code"] = issue_code
    if action_type == "set_risk" and severity:
        action["severity"] = severity
    if note:
        action["note"] = note

    obs_json, reward, done, _, _ = env.step(json.dumps(action))
    obs = json.loads(obs_json)
    history = list(history) + [{"step": len(history) + 1, "action": action, "reward": reward}]
    feed = "".join(_action_card(s) for s in reversed(history))
    sc_html = ""
    explain_html = ""
    if done:
        sc = env.get_grader_score()
        sc_html = _scores_html(sc, score_to_grade(sc["overall"]))
        explain_html = _explain_html(env)
        try:
            get_elo_tracker().update(sc["overall"])
        except Exception:
            pass
    return env, history, _env_state(obs), feed, sc_html, explain_html


# ═══════════════════════════════════════════════════════════════
#  GRADIO APP
# ═══════════════════════════════════════════════════════════════

TASK_CHOICES = [("Task 1 · Easy — CustomerCare AI", 1),
                ("Task 2 · Medium — HRAdvisor AI", 2),
                ("Task 3 · Hard — WealthWise Copilot", 3)]

ACTION_TYPES = ["inspect_artifact", "flag_issue", "request_mitigation",
                "set_risk", "approve", "reject", "escalate"]

ACTION_DOCS = {
    "inspect_artifact": '🔍 **inspect_artifact** — Read an artifact file to examine its contents.\n- `target` (**required**) — Filename to inspect, e.g. `logging_policy.yaml`',
    "flag_issue": '🚩 **flag_issue** — Flag a policy violation you found.\n- `issue_code` (**required**) — e.g. `PII-001`\n- `severity` (**required**) — `low`, `medium`, `high`, or `critical`\n- `target` — Which artifact contains the violation',
    "request_mitigation": '🛡️ **request_mitigation** — Request a fix for a flagged violation.\n- `issue_code` (**required**) — Code of the flagged issue',
    "set_risk": '⚡ **set_risk** — Set overall risk level.\n- `severity` (**required**) — `low`, `medium`, `high`, or `critical`',
    "approve": '✅ **approve** — Approve the copilot for launch (no violations found).',
    "reject": '🚫 **reject** — Block launch (critical/high violations found).',
    "escalate": '⚠️ **escalate** — Escalate to senior governance board.',
}


def _update_action_doc(action_type):
    return ACTION_DOCS.get(action_type, "")


def create_app():
    with gr.Blocks(title="CopilotAudit-Gym PRO v3.1", css=CSS,
                   theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="slate",
                                        neutral_hue="slate")) as demo:
        env_state = gr.State(None)
        step_history = gr.State([])

        # ── Hero ──
        gr.Markdown("""# 🏛️ CopilotAudit-Gym — Interactive Demo

**You are the AI governance reviewer.** An enterprise AI copilot is requesting launch approval.
Inspect its 9 artifacts, flag policy violations, request mitigations, then make your final decision.

**Flow:** Reset → Inspect artifacts → Flag violations → Request mitigations → Set risk → Approve / Reject / Escalate → See score
""")

        # ── Reference Accordions ──
        with gr.Row():
            with gr.Accordion("📋 Policy Rules Reference", open=False):
                gr.Markdown(_policy_ref_md())
            with gr.Accordion("💰 Reward Function Reference", open=False):
                gr.Markdown(_reward_ref_md())

        # ── Main Tabs ──
        with gr.Tabs():
            # ════════════ TAB 1: MANUAL REVIEW ════════════
            with gr.Tab("🎮 Manual Review (You are the Reviewer)"):
                with gr.Row():
                    # Left: Controls
                    with gr.Column(scale=1, min_width=240):
                        gr.Markdown("Select a task and click **Reset** to begin.")
                        task_select = gr.Radio(choices=TASK_CHOICES, value=1, label="📋 Task")
                        reset_btn = gr.Button("🔄 Reset Episode", variant="primary", size="lg")

                    # Center: Env state + action feed
                    with gr.Column(scale=3):
                        with gr.Row():
                            with gr.Column(scale=1):
                                gr.HTML(f'{_badge("📊 Environment State", "indigo")}')
                                m_env_display = gr.HTML(_env_state(None))
                            with gr.Column(scale=1):
                                gr.HTML(f'{_badge("📜 Action Log", "green")}')
                                m_feed = gr.HTML("")

                        # Response area
                        gr.HTML(f'{_badge("Your action (as the reviewer)", "amber")}')
                        with gr.Row():
                            m_action = gr.Dropdown(choices=ACTION_TYPES, value="inspect_artifact",
                                                   label="Action Type", scale=2)
                            m_target = gr.Dropdown(choices=[], label="target (artifact)", scale=2)
                        with gr.Row():
                            m_code = gr.Dropdown(choices=[r.code for r in POLICY_RULES],
                                                 label="issue_code", scale=1)
                            m_sev = gr.Dropdown(choices=["low", "medium", "high", "critical"],
                                                label="severity", scale=1)
                        m_note = gr.Textbox(label="note (optional)", placeholder="Describe the issue…",
                                            max_lines=2)
                        with gr.Row():
                            step_btn = gr.Button("⚡ Execute Action", variant="secondary", size="lg")
                            gr.HTML('<div style="flex:1;"></div>')

                    # Right: Action documentation
                    with gr.Column(scale=1, min_width=260):
                        gr.HTML(f'{_badge("🛠️ Action Reference", "blue")}')
                        action_doc = gr.Markdown(ACTION_DOCS["inspect_artifact"])
                        m_action.change(fn=_update_action_doc, inputs=m_action, outputs=action_doc)

                # Score + Explain display
                m_scores = gr.HTML("")
                m_explain = gr.HTML("")

                # Events
                reset_btn.click(fn=manual_reset, inputs=[task_select],
                                outputs=[env_state, step_history, m_env_display, m_feed, m_scores, m_target])
                step_btn.click(fn=manual_step,
                               inputs=[env_state, step_history, m_action, m_target, m_code, m_sev, m_note],
                               outputs=[env_state, step_history, m_env_display, m_feed, m_scores, m_explain])

            # ════════════ TAB 2: AUTO BENCHMARK ════════════
            with gr.Tab("🎯 Automated Benchmark"):
                gr.Markdown("Watch a **rule-based expert agent** review a governance packet step by step. "
                            "Every action earns a reward, and a final grader score is computed.")
                with gr.Row():
                    auto_task = gr.Radio(choices=TASK_CHOICES, value=1, label="📋 Task", scale=2)
                    auto_btn = gr.Button("🚀 Run Benchmark", variant="primary", size="lg", scale=1)

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(f'{_badge("📊 Environment State", "indigo")}')
                        auto_env = gr.HTML(_env_state(None))
                    with gr.Column(scale=1):
                        gr.HTML(f'{_badge("📜 Agent Actions", "green")}')
                        auto_feed = gr.HTML("")

                gr.HTML(f'{_badge("📈 Grader Scores", "amber")}')
                auto_scores = gr.HTML("")
                auto_explain = gr.HTML("")

                auto_btn.click(fn=run_auto, inputs=[auto_task],
                               outputs=[auto_env, auto_feed, auto_scores, auto_explain])

            # ════════════ TAB 3: MULTI-AGENT PIPELINE ════════════
            with gr.Tab("🧩 Multi-Agent Pipeline"):
                gr.Markdown("""
**Watch all 3 agents work together in real time:**
1. 🧩 **Problem Maker** generates a new adversarial governance scenario
2. 🤖 **Reviewer** (rule-based expert) audits all artifacts and makes a decision
3. ⚖️ **Judge** (Junior · Senior · Principal personas) evaluates the episode quality
""")
                with gr.Row():
                    ma_domain = gr.Dropdown(
                        choices=["auto", "legal", "finance", "healthcare", "hr", "sales", "it"],
                        value="auto", label="🏭 Domain (auto = random)", scale=2
                    )
                    ma_difficulty = gr.Dropdown(
                        choices=["easy", "medium", "hard"],
                        value="medium", label="⚡ Difficulty", scale=1
                    )
                    ma_btn = gr.Button("🚀 Run Full Pipeline", variant="primary", size="lg", scale=2)

                # Status log
                gr.HTML(f'{_badge("📡 Pipeline Log", "indigo")}')
                ma_log = gr.HTML('<div style="color:#94A3B8;padding:20px;text-align:center;">Click Run to start the 3-agent pipeline.</div>')

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML(f'{_badge("🧩 Generated Scenario", "blue")}')
                        ma_scenario = gr.HTML("")
                    with gr.Column(scale=1):
                        gr.HTML(f'{_badge("🤖 Reviewer Actions", "green")}')
                        ma_review = gr.HTML("")

                gr.HTML(f'{_badge("⚖️ Judge Verdict (3 Personas)", "amber")}')
                ma_verdict = gr.HTML("")

                def _scenario_html(task_data: dict) -> str:
                    name = task_data.get("feature_name", "Unknown")
                    summary = task_data.get("feature_summary", "")[:300]
                    violations = task_data.get("ground_truth_violations", [])
                    source = task_data.get("source", "template")
                    src_badge = "🤖 AI-Generated" if source == "generated" else "📋 Template"
                    arts = list(task_data.get("artifacts", {}).keys())
                    return f'''
<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:16px;">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
    <div style="font-weight:700;color:#1E293B;font-size:1.05em;">🏢 {name}</div>
    <span style="font-size:0.72em;background:#EEF2FF;color:#4F46E5;padding:3px 10px;border-radius:6px;">{src_badge}</span>
  </div>
  <div style="color:#64748B;font-size:0.8em;margin-bottom:12px;">{summary}</div>
  <div style="margin-bottom:10px;">
    {"".join(f'<span style="display:inline-block;background:#F1F5F9;color:#475569;padding:3px 10px;border-radius:6px;font-size:0.72em;margin:3px;">{a}</span>' for a in arts)}
  </div>
  <div style="border-top:1px solid #F1F5F9;padding-top:10px;">
    <div style="font-size:0.72em;font-weight:700;color:#64748B;margin-bottom:6px;">HIDDEN VIOLATIONS ({len(violations)})</div>
    {"".join(f'<div style="font-size:0.75em;color:#DC2626;padding:2px 0;">🔴 {v}</div>' for v in violations)}
  </div>
</div>'''

                def _verdict_html(verdict: dict) -> str:
                    if not verdict:
                        return ""
                    overall = verdict.get("overall_score", 0)
                    rec = verdict.get("recommendation", "")
                    gc = "#059669" if overall >= 0.75 else "#D97706" if overall >= 0.5 else "#DC2626"
                    personas_html = ""
                    for dim in verdict.get("dimensions", []):
                        pc = {"Junior Compliance Officer": "#4F46E5",
                              "Senior GRC Analyst": "#7C3AED",
                              "Principal AI Safety Lead": "#DC2626"}.get(dim.get("persona", ""), "#64748B")
                        personas_html += f'''
<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:10px;padding:12px;margin:8px 0;">
  <div style="display:flex;justify-content:space-between;margin-bottom:6px;">
    <span style="font-weight:700;color:{pc};font-size:0.82em;">{dim.get("persona","")}</span>
    <span style="font-family:monospace;font-weight:700;color:{pc};">{dim.get("score",0):.2f}</span>
  </div>
  <div style="color:#64748B;font-size:0.75em;font-style:italic;">"{dim.get("reasoning","")[:200]}"</div>
</div>'''

                    return f'''
<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:14px;padding:16px;">
  <div style="text-align:center;margin-bottom:16px;">
    <div style="font-size:3em;font-weight:800;color:{gc};font-family:monospace;">{overall:.2f}</div>
    <div style="color:{gc};font-weight:600;">{rec}</div>
  </div>
  {personas_html}
</div>'''

                def run_pipeline(domain, difficulty):
                    import json as _json

                    log_lines = []
                    def _log(msg):
                        log_lines.append(msg)
                        log_html = "".join(
                            f'<div style="padding:4px 0;border-bottom:1px solid #F1F5F9;font-size:0.8em;color:#475569;">{l}</div>'
                            for l in log_lines
                        )
                        return f'<div style="padding:12px;">{log_html}</div>'

                    # Step 1: Problem Maker
                    yield _log("🧩 Problem Maker: Generating adversarial scenario..."), "", "", ""
                    try:
                        from app.problem_maker import generate_task
                        from app.models import GenerateRequest
                        req = GenerateRequest(
                            domain=None if domain == "auto" else domain,
                            difficulty=difficulty,
                            num_violations={"easy": 1, "medium": 2, "hard": 3}.get(difficulty, 2)
                        )
                        task = generate_task(req)
                        task_data = task.model_dump() if hasattr(task, "model_dump") else task
                        source = task_data.get("source", "template")
                        yield _log(f"✅ Scenario ready — '{task_data['feature_name']}' ({source}) · {len(task_data['ground_truth_violations'])} violation(s)"), _scenario_html(task_data), "", ""
                    except Exception as e:
                        yield _log(f"⚠️ Problem Maker error: {e}. Using static task."), "", "", ""
                        from app.tasks import load_task
                        task_data = load_task({"easy":1,"medium":2,"hard":3}.get(difficulty, 2))
                        yield _log(f"📋 Loaded static task: '{task_data['feature_name']}'"), _scenario_html(task_data), "", ""

                    # Step 2: Reviewer
                    yield _log("🤖 Reviewer: Starting episode..."), _scenario_html(task_data), "", ""
                    env_obj = GovernanceReviewEnv()
                    obs_json, _ = env_obj.reset(options={"generated_task": task_data} if task_data.get("source") == "generated" else {"task_id": task_data.get("task_id", 1)})
                    obs = _json.loads(obs_json)
                    steps = []
                    for i in range(25):
                        action = _plan_action(obs)
                        obs_json, reward, done, _, _ = env_obj.step(_json.dumps(action))
                        obs = _json.loads(obs_json)
                        steps.append({"step": i+1, "action": action, "reward": reward})
                        feed = "".join(_action_card(s) for s in reversed(steps))
                        yield _log(f"🤖 Step {i+1}: {action['action_type']} → {reward:+.2f}"), _scenario_html(task_data), feed, ""
                        if done:
                            break

                    scores = env_obj.get_grader_score()
                    yield _log(f"✅ Review complete — overall: {scores['overall']:.3f} | safety: {scores['safety']:.2f}"), _scenario_html(task_data), "".join(_action_card(s) for s in reversed(steps)), ""

                    # Step 3: Judge
                    yield _log("⚖️ Judge: 3 personas evaluating the episode..."), _scenario_html(task_data), "".join(_action_card(s) for s in reversed(steps)), ""
                    try:
                        from app.judge import judge_episode
                        transcript = env_obj.build_transcript()
                        verdict = judge_episode(transcript)
                        verdict_dict = verdict.model_dump() if hasattr(verdict, "model_dump") else verdict
                        yield _log(f"✅ Judge complete — score: {verdict_dict.get('overall_score', 0):.3f}"), _scenario_html(task_data), "".join(_action_card(s) for s in reversed(steps)), _verdict_html(verdict_dict)
                    except Exception as e:
                        # Fallback: show grader scores as verdict
                        fallback = {
                            "overall_score": scores["overall"],
                            "recommendation": "APPROVE" if scores["overall"] >= 0.75 else "REJECT",
                            "dimensions": [
                                {"persona": "Junior Compliance Officer", "score": scores["compliance"], "reasoning": f"Caught {scores['compliance']*100:.0f}% of violations."},
                                {"persona": "Senior GRC Analyst", "score": scores["safety"], "reasoning": f"Safety score: {scores['safety']:.2f} (binary — caught all critical/high)"},
                                {"persona": "Principal AI Safety Lead", "score": scores["overall"], "reasoning": f"Overall grader score: {scores['overall']:.3f}. Judge API unavailable: {str(e)[:80]}"},
                            ]
                        }
                        yield _log(f"⚠️ Judge API unavailable — showing grader fallback"), _scenario_html(task_data), "".join(_action_card(s) for s in reversed(steps)), _verdict_html(fallback)

                ma_btn.click(
                    fn=run_pipeline,
                    inputs=[ma_domain, ma_difficulty],
                    outputs=[ma_log, ma_scenario, ma_review, ma_verdict]
                )


            with gr.Tab("🥊 Red Team Arena"):
                gr.Markdown("""
**Self-play adversarial training** — The Problem Maker generates harder scenarios as the Reviewer improves.
This is the same mechanism behind AlphaGo's self-play. Both sides co-evolve.
""")
                with gr.Row():
                    with gr.Column(scale=1):
                        rt_rounds = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="Rounds to play")
                        rt_btn = gr.Button("🥊 Start Red Team Session", variant="primary", size="lg")
                        rt_reset_btn = gr.Button("🔄 Reset Session", variant="secondary")
                    with gr.Column(scale=2):
                        gr.HTML(f'{_badge("📊 Live Scoreboard", "indigo")}')
                        rt_scoreboard = gr.HTML('<div style="color:#94A3B8;padding:30px;text-align:center;">Click Start to begin the self-play battle.</div>')

                gr.HTML(f'{_badge("📜 Round History", "green")}')
                rt_history = gr.HTML("")

                gr.HTML(f'{_badge("🔗 Causal Violation Graph", "blue")}')
                causal_md = "\n".join(
                    f"- **`{src}`** → causes → " + " · ".join(f"`{dst}`" for dst in dsts)
                    for src, dsts in VIOLATION_CAUSES.items()
                )
                gr.Markdown(causal_md)

                def _rt_scoreboard_html(lb, elo):
                    total = max(1, lb["total_rounds"])
                    rv = lb["reviewer_wins"]; mv = lb["maker_wins"]
                    rwr = lb["reviewer_win_rate"]
                    trend = lb["trend"]
                    avg = lb["avg_score"]
                    elo_val = round(elo["elo"], 0)
                    elo_grade = elo["grade"]
                    rec_task = elo["recommended_task"]
                    rc = "#059669" if rwr >= 0.5 else "#DC2626"

                    rounds_html = ""
                    for r in reversed(lb.get("recent_rounds", [])[-5:]):
                        oc = "#059669" if r["outcome"] == "reviewer" else "#DC2626"
                        icon = "🟢" if r["outcome"] == "reviewer" else "🔴"
                        rounds_html += f'''<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:0.78em;">
                          <span style="color:#64748B;">{icon} Round {r['round']} · Task {r.get('scores',{}).get('task_id','?')}</span>
                          <span style="font-family:monospace;color:{oc};font-weight:700;">{r['scores']['overall']:.3f}</span>
                          {f'<span style="color:#94A3B8;font-size:0.7em;">ELO→{r.get("elo_after","?")}</span>' if 'elo_after' in r else ''}
                        </div>'''

                    return f'''
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px;margin-bottom:14px;">
  <div style="text-align:center;background:#EEF2FF;padding:14px;border-radius:12px;">
    <div style="font-size:2em;font-weight:800;color:#4F46E5;">{rv}</div>
    <div style="color:#64748B;font-size:0.75em;">🤖 Reviewer Wins</div>
  </div>
  <div style="text-align:center;background:#FEF2F2;padding:14px;border-radius:12px;">
    <div style="font-size:2em;font-weight:800;color:#DC2626;">{mv}</div>
    <div style="color:#64748B;font-size:0.75em;">🧩 Maker Wins</div>
  </div>
  <div style="text-align:center;background:#F8FAFC;padding:14px;border-radius:12px;">
    <div style="font-size:2em;font-weight:800;color:{elo["elo"] >= 1200 and "#059669" or "#4F46E5"}">{int(elo_val)}</div>
    <div style="color:#64748B;font-size:0.75em;">⚡ ELO Rating</div>
  </div>
</div>
<div style="background:#F8FAFC;padding:10px 14px;border-radius:10px;border-left:3px solid {rc};margin-bottom:12px;">
  <span style="font-weight:600;color:{rc};">{trend}</span>
  <span style="float:right;color:#94A3B8;font-size:0.78em;">Avg score: {avg:.3f} · Win rate: {rwr:.1%}</span>
</div>
<div style="font-size:0.78em;color:#64748B;margin-bottom:8px;">ELO Grade: <b>{elo_grade}</b> · Next task: <b>Task {rec_task}</b></div>
{rounds_html}'''

                def run_red_team(num_rounds):
                    import json as _json
                    from app.red_team import get_red_team_session
                    from app.elo import get_elo_tracker
                    from app.policies import run_all_checks, POLICY_RULE_MAP

                    def _agent(obs):
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
                                            "severity": rule.severity if rule else "high", "note": msg[:80]}
                        except Exception:
                            pass
                        for issue in obs.get("flagged_issues", []):
                            code = issue.get("code", "")
                            if code and code not in mitigated:
                                return {"action_type": "request_mitigation", "issue_code": code}
                        if obs.get("flagged_issues") and not risk:
                            sevs = [i.get("severity", "medium") for i in obs.get("flagged_issues", [])]
                            sv = "critical" if "critical" in sevs else "high" if "high" in sevs else "medium"
                            return {"action_type": "set_risk", "severity": sv}
                        if obs.get("flagged_issues"):
                            return {"action_type": "reject"}
                        return {"action_type": "approve"}

                    session = get_red_team_session()
                    elo_tracker = get_elo_tracker()

                    for _ in range(int(num_rounds)):
                        round_env = GovernanceReviewEnv(task_id=elo_tracker.select_task_id())
                        result = session.play_round(round_env, _agent)
                        elo_tracker.update(result["scores"]["overall"])
                        result["elo_after"] = round(elo_tracker.elo, 1)
                        lb = session.get_leaderboard()
                        elo_d = elo_tracker.to_dict()
                        yield _rt_scoreboard_html(lb, elo_d), ""

                    lb = session.get_leaderboard()
                    elo_d = elo_tracker.to_dict()
                    yield _rt_scoreboard_html(lb, elo_d), ""

                def reset_red_team():
                    get_red_team_session().reset_session()
                    tracker = get_elo_tracker()
                    tracker.elo = 1000.0
                    tracker.round_history = []
                    tracker.save()
                    return '<div style="color:#94A3B8;padding:30px;text-align:center;">Session reset. Click Start to begin.</div>', ""

                rt_btn.click(fn=run_red_team, inputs=[rt_rounds], outputs=[rt_scoreboard, rt_history])
                rt_reset_btn.click(fn=reset_red_team, outputs=[rt_scoreboard, rt_history])

            # ════════════ TAB 4: POLICY & SCORING ════════════
            with gr.Tab("📖 Policy & Scoring Guide"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📋 8 Governance Policy Rules")
                        gr.Markdown("These deterministic rules are checked against every AI copilot launch packet.\n")
                        gr.Markdown(_policy_ref_md())
                    with gr.Column():
                        gr.Markdown("### 📊 Grading Dimensions")
                        gr.Markdown("""| Dimension | Weight | Measures |
|---|---|---|
| **Safety** | 35% | Caught ALL critical/high violations? |
| **Compliance** | 25% | Fraction of true violations caught |
| **Completeness** | 20% | Fraction of artifacts inspected |
| **Precision** | 10% | 1 − false positive rate |
| **Mitigation** | 10% | Mitigations requested per violation |

**Grades:** A (≥0.90) · B (≥0.75) · C (≥0.60) · D (≥0.40) · F (<0.40)
""")
                        gr.Markdown("### 💰 Reward Function")
                        gr.Markdown(_reward_ref_md())

            # ════════════ TAB 5: ABOUT ════════════
            with gr.Tab("ℹ️ About"):
                gr.Markdown("""### 🏛️ CopilotAudit-Gym PRO v3.1

A high-fidelity multi-agent RL environment for enterprise AI governance review training.

**3-Agent Architecture:**
- 🧩 **Problem Maker** (Gemini 1.5 Flash) — generates adversarial governance scenarios
- 🤖 **Reviewer** (Qwen2.5-0.5B + LoRA) — the RL agent under test
- ⚖️ **Judge** (3-persona: Junior · Senior · Principal) — multi-expert evaluation

**5 Advanced RL Features:**
- 🔬 **Explainable Reward Decomposition** — per-step breakdown of WHY each reward was given
- 🔗 **Causal Violation Graph** — flagging PII-001 hints that TRAINING-006 is also likely
- ⚡ **ELO-Based Adaptive Difficulty** — task selection tracks agent rating like chess.com
- 🥊 **Adversarial Red Team Loop** — self-play co-evolution (AlphaGo technique)
- 📉 **Counterfactual Regret** — computes optimal vs actual to quantify improvement potential

**Environment:**
- 9 enterprise artifacts per task (product specs, logging configs, model cards, etc.)
- 8 deterministic policy rules with machine-checkable compliance
- Dense reward shaping: +0.2 per inspection, +1.5 per correct flag, +4.0 correct decision
- 3 benchmark tasks: Easy (1 violation) → Medium (2) → Hard (3)

**API Endpoints:** `/reset`, `/step`, `/state`, `/grader`, `/health`, `/tasks`, `/baseline`, `/explain`, `/elo`, `/redteam/start`

Built for the [OpenEnv Hackathon](https://github.com/facebookresearch/openenv). MIT License.
""")

    return demo.queue()


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
