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
import sys
import random
from datetime import datetime

# Ensure project root is in path for relative imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.policies import SECTOR_CONFIGS

BASE_URL = os.getenv("GOVERNANCE_ENV_URL", "http://localhost:8000")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def get_persona_card(sector_key: str) -> str:
    """Generate Markdown for the persona card."""
    cfg = SECTOR_CONFIGS.get(sector_key, SECTOR_CONFIGS["tech"])
    return f"""
<div style="padding: 15px; border-radius: 10px; background: #1e293b; border: 1px solid #334155;">
    <h3 style="margin: 0; color: #38bdf8;">👤 {cfg.persona_name}</h3>
    <p style="margin: 5px 0; font-size: 0.9em; color: #94a3b8;"><i>{cfg.persona_style}</i></p>
    <div style="margin-top: 10px; font-size: 0.85em; color: #cbd5e1;">
        <b>🎯 Focus Areas:</b><br>
        • {", ".join(cfg.critical_codes)} (Critical)<br>
        • {", ".join(cfg.high_codes)} (High)
    </div>
</div>
"""


def format_observation_html(obs: dict) -> str:
    """Format observation as a high-fidelity HTML card."""
    feature = obs.get('feature_name', 'N/A')
    domain = obs.get('domain', 'N/A')
    risk = obs.get('current_risk', 'not set').upper()
    stage = obs.get('review_stage', 'inspection')
    message = obs.get('message', '')
    
    # Progress Calculation
    inspected = len(obs.get("inspected_artifacts", []))
    total = len(obs.get("visible_artifacts", {}))
    progress = (inspected / total * 100) if total > 0 else 0
    
    html = f"""
    <div style="background: #1e293b; color: white; padding: 20px; border-radius: 12px; border: 1px solid #334155; font-family: sans-serif;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <h2 style="margin: 0; color: #38bdf8;">🏢 {feature}</h2>
            <span style="background: #0ea5e9; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold;">{domain}</span>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px;">
            <div style="background: #0f172a; padding: 10px; border-radius: 8px;">
                <div style="color: #94a3b8; font-size: 0.75em; text-transform: uppercase;">Current Risk</div>
                <div style="font-size: 1.2em; font-weight: bold; color: {'#ef4444' if 'CRITICAL' in risk or 'HIGH' in risk else '#fbbf24' if 'MEDIUM' in risk else '#10b981'}">{risk}</div>
            </div>
            <div style="background: #0f172a; padding: 10px; border-radius: 8px;">
                <div style="color: #94a3b8; font-size: 0.75em; text-transform: uppercase;">Review Stage</div>
                <div style="font-size: 1.2em; font-weight: bold; color: #38bdf8;">{stage.capitalize()}</div>
            </div>
        </div>

        <div style="background: #334155; border-radius: 10px; height: 8px; margin-bottom: 5px; overflow: hidden;">
            <div style="background: #38bdf8; width: {progress}%; height: 100%; transition: width 0.5s ease;"></div>
        </div>
        <div style="display: flex; justify-content: space-between; font-size: 0.8em; color: #94a3b8; margin-bottom: 20px;">
            <span>Artifact Inspection</span>
            <span>{inspected}/{total} ({progress:.0f}%)</span>
        </div>

        <div style="background: #0f172a; padding: 15px; border-radius: 8px; border-left: 4px solid #38bdf8; margin-bottom: 20px;">
            <p style="margin: 0; font-style: italic; color: #cbd5e1;">"{message}"</p>
        </div>

        <h4 style="margin: 0 0 10px 0; color: #94a3b8; font-size: 0.9em;">📂 PROJECT ARTIFACTS</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 8px;">
    """
    for name in obs.get("visible_artifacts", {}):
        is_inspected = name in obs.get("inspected_artifacts", [])
        html += f"""
        <div style="background: {'#065f46' if is_inspected else '#1e293b'}; color: {'#34d399' if is_inspected else '#94a3b8'}; 
                    padding: 4px 10px; border-radius: 6px; border: 1px solid {'#059669' if is_inspected else '#334155'}; font-size: 0.85em;">
            {'✅' if is_inspected else '⚪'} {name}
        </div>
        """
    
    if obs.get("flagged_issues"):
        html += """</div><h4 style="margin: 20px 0 10px 0; color: #ef4444; font-size: 0.9em;">🚩 IDENTIFIED VIOLATIONS</h4><div style="display: grid; gap: 8px;">"""
        for issue in obs["flagged_issues"]:
            html += f"""
            <div style="background: #450a0a; border: 1px solid #991b1b; padding: 10px; border-radius: 8px;">
                <div style="display: flex; justify-content: space-between; font-weight: bold; color: #f87171; font-size: 0.9em;">
                    <span>{issue['code']}</span>
                    <span>{issue['severity'].upper()}</span>
                </div>
                <div style="font-size: 0.8em; color: #fca5a5; margin-top: 4px;">{issue.get('note', '')[:100]}...</div>
            </div>
            """
    
    html += "</div></div>"
    return html


def run_demo_episode(task_id: int, mode: str, sector: str, progress=gr.Progress()):
    """Run a complete episode and stream updates with high-fidelity UI."""
    # Check for API key
    current_key = os.getenv("GEMINI_API_KEY")
    if mode == "Live Qwen mode" and not current_key:
        yield "### ❌ API Key Missing", "### 🤖 Reviewer\n*Inactive*", "### ⚖️ Auditor\n*Inactive*", "### 📋 Environment\n*Inactive*", "### 📊 Summary\n*Inactive*"
        return

    all_steps = []
    reward_total = 0.0
    sector_cfg = SECTOR_CONFIGS.get(sector, SECTOR_CONFIGS["tech"])
    
    # --- Agent 1: Problem Maker ---
    progress(0, desc="Generating Scenario...")
    scenario_info = f"### 🧩 Problem Maker\n"
    if mode == "Live Qwen mode":
        scenario_info += f"🔄 *Initializing {sector_cfg.name} environment...*\n"
        yield scenario_info, "### 🤖 Reviewer\n*Waiting...*", "### ⚖️ Auditor\n*Waiting...*", "### 📋 Environment\n*Initializing...*", ""
        
        try:
            gen_resp = requests.post(f"{BASE_URL}/generate_task", json={
                "difficulty": "medium",
                "num_violations": random.randint(1, 3)
            })
            if gen_resp.status_code == 200:
                task = gen_resp.json()
                scenario_info = f"""### 🧩 Problem Maker
<div style="background: #0f172a; padding: 12px; border-radius: 8px; border: 1px solid #334155;">
    <b style="color: #38bdf8;">🏢 Feature:</b> {task['feature_name']}<br>
    <b style="color: #38bdf8;">🏭 Sector:</b> {sector_cfg.name}<br>
    <b style="color: #38bdf8;">🎯 Target:</b> {task['domain']}<br>
    <hr style="border: 0; border-top: 1px solid #334155; margin: 8px 0;">
    <p style="margin: 0; font-size: 0.9em; color: #cbd5e1;">{task['feature_summary'][:200]}...</p>
</div>
"""
                yield scenario_info, "### 🤖 Reviewer\n*Booting...*", "### ⚖️ Auditor\n*Waiting...*", "### 📋 Environment\n*Ready*", ""
                reset_resp = requests.post(f"{BASE_URL}/reset", json={"use_generated_task": True})
                obs = reset_resp.json()["observation"]
            else:
                yield "❌ Generation failed", "", "", "", ""
                return
        except Exception as e:
            yield f"❌ Error: {e}", "", "", "", ""
            return
    else:
        scenario_info += f"📍 *Benchmark Task {task_id}*\n"
        reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
        obs = reset_resp.json()["observation"]
        yield scenario_info, "### 🤖 Reviewer\n*Ready*", "### ⚖️ Auditor\n*Waiting...*", format_observation_html(obs), ""

    # --- Agent 2: Reviewer ---
    progress(0.2, desc="RL Agent Reviewing...")
    
    for i in range(15):
        # Step metadata
        step_header = f"### 🤖 {sector_cfg.persona_name} (Step {i+1})"
        yield scenario_info, f"{step_header}\n🤖 *Thinking...*", "### ⚖️ Auditor\n*Waiting...*", format_observation_html(obs), ""

        if mode == "Pre-scripted Demo":
            time.sleep(0.6)
            # Find first uninspected artifact
            uninspected = [k for k in obs['visible_artifacts'].keys() if k not in obs['inspected_artifacts']]
            if uninspected:
                action = {"action_type": "inspect_artifact", "target": uninspected[0]}
            else:
                action = {"action_type": "approve", "note": "All clear."}
        else:
            # Live Qwen Logic
            from agents.reviewer import get_memory_context, REVIEWER_SYSTEM_TEMPLATE, _extract_json, _plan_actions
            from app.model_config import get_reviewer_client
            
            client = get_reviewer_client()
            memory_context = get_memory_context(n_episodes=3)
            system = REVIEWER_SYSTEM_TEMPLATE.format(
                memory_context=memory_context,
                persona_name=sector_cfg.persona_name,
                sector_name=sector_cfg.name,
                persona_style=sector_cfg.persona_style
            )
            user = f"State: {json.dumps(obs)}"
            
            raw_resp = client.complete(system, user)
            action = _extract_json(raw_resp)
            if not action:
                action = _plan_actions(obs, obs.get("inspected_artifacts", []), [], [])

        # Step Environment
        resp = requests.post(f"{BASE_URL}/step", json={"action": json.dumps(action)})
        result = resp.json()
        reward = result["reward"]
        reward_total += reward
        obs = result["observation"]

        # Action Rendering
        icon = "✅" if reward > 0 else "❌" if reward < 0 else "🔍"
        color = "#10b981" if reward > 0 else "#ef4444" if reward < 0 else "#38bdf8"
        
        log_entry = f"""
<div style="background: #0f172a; padding: 10px; border-radius: 8px; border-left: 4px solid {color}; margin-bottom: 10px;">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <b style="color: {color};">{icon} {action['action_type'].upper()}</b>
        <span style="font-family: monospace; font-size: 0.9em; color: {color};">{reward:+.1f}</span>
    </div>
    <div style="font-size: 0.85em; color: #94a3b8; margin-top: 4px;">
        {f"Target: <code>{action.get('target', '')}</code>" if action.get('target') else ""}
        {f" | Code: <code>{action.get('issue_code', '')}</code>" if action.get('issue_code') else ""}
    </div>
    {f"<p style='margin: 5px 0 0 0; font-size: 0.85em; color: #cbd5e1;'><i>{action.get('note', '')[:120]}</i></p>" if action.get('note') else ""}
</div>
"""
        all_steps.append(log_entry)
        
        yield scenario_info, f"{step_header}\n\n" + "".join(reversed(all_steps[-4:])), "### ⚖️ Auditor\n*Waiting...*", format_observation_html(obs), ""

        time.sleep(0.4)
        if result["done"]: break

    # --- Agent 3: Judge ---
    progress(0.9, desc="Principal Audit...")
    judge_info = "### ⚖️ Auditor (Judge)\n🔍 *Finalizing compliance audit...*"
    yield scenario_info, f"### 🤖 Reviewer\n\n" + "".join(reversed(all_steps)), judge_info, format_observation_html(obs), ""
    
    # Final Grader
    grader_resp = requests.get(f"{BASE_URL}/grader")
    grader = grader_resp.json()
    scores = grader["scores"]
    
    judge_info = f"""### ⚖️ Audit Verdict: {grader['grade']}
<div style="background: #1e293b; padding: 15px; border-radius: 10px; border: 1px solid #334155;">
    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 15px;">
        <div style="text-align: center; background: #0f172a; padding: 8px; border-radius: 6px;">
            <div style="font-size: 0.7em; color: #94a3b8;">SAFETY</div>
            <div style="font-size: 1.1em; font-weight: bold; color: #ef4444;">{scores['safety']:.2f}</div>
        </div>
        <div style="text-align: center; background: #0f172a; padding: 8px; border-radius: 6px;">
            <div style="font-size: 0.7em; color: #94a3b8;">COMPLIANCE</div>
            <div style="font-size: 1.1em; font-weight: bold; color: #10b981;">{scores['compliance']:.2f}</div>
        </div>
    </div>
    <p style="margin: 0; font-size: 0.85em; color: #cbd5e1; line-height: 1.4;">
        <b>Principal GRC Feedback:</b><br>
        The model demonstrated {sector_cfg.persona_name} behavioral patterns. Policy coverage was {grader['grade']}.
    </p>
</div>
"""
    
    # Save review
    os.makedirs("reviews", exist_ok=True)
    review_path = f"reviews/audit_{datetime.now().strftime('%H%M%S')}.json"
    with open(review_path, "w") as f:
        json.dump({"sector": sector, "steps": len(all_steps), "score": scores['overall']}, f)

    final_summary = f"""
<div style="background: #0f172a; padding: 20px; border-radius: 12px; border: 2px solid #38bdf8; text-align: center;">
    <h2 style="margin: 0; color: #38bdf8;">🏆 Training Session Complete</h2>
    <div style="font-size: 3em; font-weight: bold; color: white; margin: 10px 0;">{scores['overall']:.2f}</div>
    <div style="color: #94a3b8; font-size: 1.1em;">Global Compliance Score</div>
    <div style="margin-top: 15px; color: #38bdf8; font-family: monospace;">{review_path}</div>
</div>
"""
    yield scenario_info, f"### 🤖 Reviewer\n\n" + "".join(reversed(all_steps)), judge_info, format_observation_html(obs), final_summary


def create_demo():
    custom_css = """
    .agent-box { border-radius: 12px !important; background: #0f172a !important; border: 1px solid #1e293b !important; padding: 15px !important; }
    .gr-tabs { border: none !important; }
    .gr-tab-item { border: none !important; }
    #obs-col { border-left: 1px solid #334155; padding-left: 20px; }
    """
    
    with gr.Blocks(title="GovernanceReview-Gym Pro", theme=gr.themes.Soft(primary_hue="sky", secondary_hue="slate"), css=custom_css) as demo:
        gr.HTML("""
        <div style="display: flex; align-items: center; justify-content: center; gap: 20px; margin-bottom: 30px;">
            <div style="font-size: 3em;">🏛️</div>
            <div style="text-align: left;">
                <h1 style="margin: 0; color: #38bdf8; font-size: 2.2em;">GovernanceReview-Gym <span style="font-size: 0.4em; vertical-align: middle; background: #334155; padding: 4px 10px; border-radius: 4px; color: #94a3b8;">PRO v3.1</span></h1>
                <p style="margin: 0; color: #94a3b8;">High-Fidelity Multi-Agent Reinforcement Learning Dashboard</p>
            </div>
        </div>
        """)

        with gr.Row():
            # Sidebar
            with gr.Column(scale=1):
                with gr.Group(elem_classes="agent-box"):
                    gr.Markdown("### ⚙️ MISSION CONTROL")
                    sector_input = gr.Dropdown(
                        choices=[(v.name, k) for k, v in SECTOR_CONFIGS.items()],
                        value="tech",
                        label="Target Sector"
                    )
                    mode_input = gr.Dropdown(
                        choices=["Pre-scripted Demo", "Live Qwen mode"],
                        value="Live Qwen mode",
                        label="Agent Brain"
                    )
                    task_input = gr.Radio(
                        choices=[("Task 1", 1), ("Task 2", 2), ("Task 3", 3)],
                        value=1,
                        label="Benchmark Set"
                    )
                    
                    persona_display = gr.HTML(get_persona_card("tech"))
                    sector_input.change(fn=get_persona_card, inputs=sector_input, outputs=persona_display)
                    
                    run_btn = gr.Button("🚀 LAUNCH SIMULATION", variant="primary", size="lg")

            # Main Dashboard
            with gr.Column(scale=4):
                with gr.Row():
                    # Left: Agents
                    with gr.Column(scale=2):
                        with gr.Tabs():
                            with gr.Tab("🤖 Agent Live Feed"):
                                with gr.Row():
                                    scenario_display = gr.Markdown("### 🧩 Problem Maker\n*Waiting...*", elem_classes="agent-box")
                                    judge_display = gr.HTML("### ⚖️ Auditor\n*Waiting...*")
                                
                                reviewer_display = gr.HTML("<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: #475569;'><h3>Reviewer Action Log</h3></div>", elem_classes="agent-box")
                            
                            with gr.Tab("📈 Training Metrics"):
                                gr.Markdown("#### Real-time Performance tracking")
                                # Note: Plotly/LinePlot requires data state, keeping simple for now
                                gr.Markdown("*Metrics tracking initialized. Episode data will be plotted here.*")

                    # Right: Environment State
                    with gr.Column(scale=1, elem_id="obs-col"):
                        gr.Markdown("### 📋 SYSTEM OBSERVATION")
                        obs_display = gr.HTML("<div style='color: #475569; text-align: center; padding-top: 50px;'>Start simulation to view artifacts</div>")

                with gr.Row():
                    score_display = gr.HTML("")

        run_btn.click(
            fn=run_demo_episode,
            inputs=[task_input, mode_input, sector_input],
            outputs=[scenario_display, reviewer_display, judge_display, obs_display, score_display]
        )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)
