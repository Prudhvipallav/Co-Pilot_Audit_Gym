"""
Real-time demo using Gradio.
Shows live agent reviewing a governance packet with step-by-step visualization.

Install: pip install gradio
Run: python demo/app.py
"""

import gradio as gr
import requests
import json
import time
import os

BASE_URL = os.getenv("GOVERNANCE_ENV_URL", "http://localhost:8000")


def format_observation(obs: dict) -> str:
    """Format observation for display."""
    lines = [
        f"**🏢 Feature:** {obs.get('feature_name', 'N/A')}",
        f"**📋 Summary:** {obs.get('feature_summary', 'N/A')}",
        f"**🎯 Stage:** {obs.get('review_stage', 'N/A')} | **Step:** {obs.get('step_count', 0)}/{obs.get('max_steps', 50)}",
        "",
        f"**💬 Message:** {obs.get('message', '')}",
        "",
        "**📁 Artifacts Available:**"
    ]
    for name in obs.get("visible_artifacts", {}):
        inspected = "✅" if name in obs.get("inspected_artifacts", []) else "⬜"
        lines.append(f"  {inspected} `{name}`")

    if obs.get("flagged_issues"):
        lines.append("\n**🚩 Flagged Issues:**")
        for issue in obs["flagged_issues"]:
            lines.append(f"  • {issue['code']} ({issue['severity']}) — {issue.get('note', '')[:80]}")

    if obs.get("requested_mitigations"):
        lines.append("\n**🔧 Mitigations Requested:**")
        for m in obs["requested_mitigations"]:
            lines.append(f"  • {m.get('issue_code', '?')} — {m.get('note', '')[:80]}")

    if obs.get("current_risk"):
        lines.append(f"\n**⚠️ Risk Level:** {obs['current_risk'].upper()}")

    if obs.get("open_questions"):
        lines.append("\n**❓ Open Questions:**")
        for q in obs["open_questions"]:
            lines.append(f"  • {q}")

    return "\n".join(lines)


def run_demo_episode(task_id: int, progress=gr.Progress()):
    """Run a complete episode and stream updates."""
    # Reset
    resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    if resp.status_code != 200:
        yield "❌ Server not running. Start with: uvicorn app.main:app --reload", "", ""
        return

    obs = resp.json()["observation"]
    all_steps = []
    reward_total = 0.0

    # Pre-scripted demo actions per task
    demo_actions = {
        1: [
            {"action_type": "inspect_artifact", "target": "product_spec.md"},
            {"action_type": "inspect_artifact", "target": "logging_policy.yaml", "note": "Checking logging config for PII risks"},
            {"action_type": "inspect_artifact", "target": "data_sources.json"},
            {"action_type": "inspect_artifact", "target": "model_card.json"},
            {"action_type": "inspect_artifact", "target": "eval_report.json"},
            {"action_type": "flag_issue", "issue_code": "PII-001", "severity": "critical", "target": "logging_policy.yaml",
             "note": "raw_prompt_capture: enabled with pii_filter: disabled — SSNs will be logged in plaintext"},
            {"action_type": "request_mitigation", "issue_code": "PII-001",
             "note": "Enable PII filter before launch. Disable raw_prompt_capture or implement redaction."},
            {"action_type": "set_risk", "severity": "high"},
            {"action_type": "reject", "note": "Critical PII logging violation must be resolved before any deployment."},
        ],
        2: [
            {"action_type": "inspect_artifact", "target": "product_spec.md"},
            {"action_type": "inspect_artifact", "target": "model_card.json"},
            {"action_type": "inspect_artifact", "target": "retention_policy.yaml"},
            {"action_type": "inspect_artifact", "target": "human_oversight.md"},
            {"action_type": "inspect_artifact", "target": "eval_report.json"},
            {"action_type": "flag_issue", "issue_code": "RETENTION-005", "severity": "medium",
             "target": "retention_policy.yaml",
             "note": "retention_days: 120 exceeds the corporate 90-day maximum"},
            {"action_type": "flag_issue", "issue_code": "ESCALATION-003", "severity": "high",
             "target": "human_oversight.md",
             "note": "Model card shows high risk, but escalation plan is pending (not documented)"},
            {"action_type": "request_mitigation", "issue_code": "RETENTION-005",
             "note": "Reduce retention_days to 90 or obtain legal waiver with specific justification"},
            {"action_type": "request_mitigation", "issue_code": "ESCALATION-003",
             "note": "Document escalation path before launch — complete ticket HR-2024-0892"},
            {"action_type": "set_risk", "severity": "medium"},
            {"action_type": "escalate", "note": "Two fixable violations. Escalate to HR Ops + Legal for remediation sign-off."},
        ],
        3: [
            {"action_type": "inspect_artifact", "target": "product_spec.md"},
            {"action_type": "inspect_artifact", "target": "model_card.json"},
            {"action_type": "inspect_artifact", "target": "logging_policy.yaml"},
            {"action_type": "inspect_artifact", "target": "data_sources.json"},
            {"action_type": "inspect_artifact", "target": "eval_report.json"},
            {"action_type": "inspect_artifact", "target": "incident_history.json"},
            {"action_type": "flag_issue", "issue_code": "DOMAIN-004", "severity": "critical",
             "target": "product_spec.md",
             "note": "Provides investment allocation suggestions — financial investment advice is a prohibited domain"},
            {"action_type": "flag_issue", "issue_code": "TRAINING-006", "severity": "critical",
             "target": "model_card.json",
             "note": "External provider FinSight AI has 'training on prompts: enabled' — employee financial data will train their model"},
            {"action_type": "flag_issue", "issue_code": "PII-001", "severity": "critical",
             "target": "logging_policy.yaml",
             "note": "raw_prompt_capture: enabled with pii_filter: planned_for_v2 — salary/SSN data will be logged"},
            {"action_type": "request_mitigation", "issue_code": "DOMAIN-004",
             "note": "Remove investment advice features. Replace with education-only content and link to licensed advisors."},
            {"action_type": "request_mitigation", "issue_code": "TRAINING-006",
             "note": "Renegotiate contract with FinSight AI to disable training on prompts. Get written confirmation."},
            {"action_type": "request_mitigation", "issue_code": "PII-001",
             "note": "Implement PII filter before any launch. Do not accept v1.1 promise — block until implemented."},
            {"action_type": "set_risk", "severity": "critical"},
            {"action_type": "reject", "note": "3 critical violations including prohibited domain and external training data exposure. Hard reject."},
        ]
    }

    actions = demo_actions.get(task_id, demo_actions[1])
    total = len(actions)

    for i, action in enumerate(actions):
        progress((i + 1) / total, desc=f"Step {i+1}: {action['action_type']}")

        action_display = f"**Action {i+1}:** `{action['action_type']}`"
        if action.get("issue_code"):
            action_display += f" — **{action['issue_code']}**"
        if action.get("target"):
            action_display += f" → `{action.get('target')}`"
        if action.get("note"):
            action_display += f"\n> {action['note']}"

        resp = requests.post(f"{BASE_URL}/step", json={"action": json.dumps(action)})
        result = resp.json()
        reward = result["reward"]
        reward_total += reward
        obs = result["observation"]

        step_info = f"{'🔴' if reward < 0 else '🟢'} Reward: **{reward:+.2f}** | Total: **{reward_total:+.2f}**"
        all_steps.append(f"{action_display}\n{step_info}\n*{result['observation']['message'][:120]}*")

        steps_display = "\n\n---\n\n".join(reversed(all_steps[-6:]))
        obs_display = format_observation(obs)

        yield obs_display, steps_display, ""

        time.sleep(0.4)
        if result["done"]:
            break

    # Final score
    grader_resp = requests.get(f"{BASE_URL}/grader")
    grader = grader_resp.json()
    scores = grader["scores"]

    score_display = f"""
## 📊 Final Scores

| Dimension | Score | Meaning |
|-----------|-------|---------|
| 🛡️ Safety | **{scores['safety']:.2f}** | Caught all critical violations |
| ✅ Compliance | **{scores['compliance']:.2f}** | Violations correctly identified |
| 🔍 Completeness | **{scores['completeness']:.2f}** | Artifacts inspected |
| 🎯 Precision | **{scores['precision']:.2f}** | No false positives |
| 🔧 Mitigation | **{scores['mitigation_quality']:.2f}** | Remediation requested |
| 🏆 **Overall** | **{scores['overall']:.2f}** | Composite score |

**Grade:** {grader['grade']}
"""
    yield format_observation(obs), "\n\n---\n\n".join(reversed(all_steps)), score_display


def create_demo():
    with gr.Blocks(
        title="OpenEnv-GovernanceReview Demo",
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px !important }"
    ) as demo:
        gr.Markdown("""
# 🏛️ OpenEnv — AI Governance Review Demo
**Watch an AI agent review enterprise copilot launch packets for policy compliance in real-time.**

This demonstrates the OpenEnv-GovernanceReview environment.
The agent inspects artifacts, flags policy violations, requests mitigations, and makes approval decisions.
        """)

        with gr.Row():
            task_selector = gr.Radio(
                choices=[
                    ("🟢 Task 1 — Customer Support Copilot (Easy)", 1),
                    ("🟡 Task 2 — HR Policy Assistant (Medium)", 2),
                    ("🔴 Task 3 — Financial Planning Copilot (Hard)", 3),
                ],
                value=1,
                label="Select Task"
            )
            run_btn = gr.Button("▶ Run Agent Demo", variant="primary", scale=0)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Current Observation")
                obs_display = gr.Markdown("Click 'Run Agent Demo' to start")

            with gr.Column(scale=1):
                gr.Markdown("### 🤖 Agent Actions")
                steps_display = gr.Markdown("Steps will appear here...")

        score_display = gr.Markdown("")

        run_btn.click(
            fn=run_demo_episode,
            inputs=[task_selector],
            outputs=[obs_display, steps_display, score_display]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
