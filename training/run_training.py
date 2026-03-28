"""
GRPO Training v7 — Fixed JSON parsing, markdown stripping, prompt format.
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints")

VALID_ACTIONS = {
    "inspect_artifact", "flag_issue", "request_mitigation",
    "set_risk", "approve", "reject", "escalate",
}

VALID_CODES = {
    "PII-001", "ACCESS-002", "ESCALATION-003", "DOMAIN-004",
    "RETENTION-005", "TRAINING-006", "AUDIT-007", "EVAL-008",
}

PROMPT_SUFFIX = (
    '\nRespond with ONLY a JSON object. No explanation. No markdown. '
    'Use key "action_type" not "action". Example: '
    '{"action_type": "flag_issue", "issue_code": "PII-001", '
    '"severity": "critical", "target": "file.yaml", "note": "reason"}'
)


def build_prompt_dataset(n_prompts: int = 2000):
    """Load real-world examples from governance_examples.json."""
    from datasets import Dataset
    import random

    dataset_path = os.path.join(
        os.path.dirname(__file__), "dataset", "governance_examples.json"
    )

    with open(dataset_path, "r") as f:
        all_examples = json.load(f)

    print(f"Loaded {len(all_examples)} unique examples from dataset")
    random.shuffle(all_examples)

    entries = []
    for i in range(n_prompts):
        ex = all_examples[i % len(all_examples)]
        entries.append({
            "prompt": ex["prompt"] + PROMPT_SUFFIX,
            "correct_action": ex.get("correct_action", ""),
            "correct_code": ex.get("correct_code", ""),
            "correct_severity": ex.get("correct_severity", ""),
            "correct_target": ex.get("correct_target", ""),
        })

    return Dataset.from_list(entries)


def make_reward_fn():
    def reward_fn(completions, prompts=None, correct_action=None,
                  correct_code=None, correct_severity=None, correct_target=None, **kwargs):
        rewards = []

        for i, completion in enumerate(completions):
            text = completion if isinstance(completion, str) else str(completion)
            score = 0.0

            # Length + closing brace penalty
            char_len = len(text.strip())
            has_closing_brace = "}" in text

            if not has_closing_brace:
                score -= 3.0
            elif char_len <= 200:
                score += 1.5
            elif char_len <= 350:
                score += 0.0
            elif char_len <= 500:
                score -= 1.0
            else:
                score -= 2.5

            # Parse JSON — handle markdown blocks and key variants
            action = None
            try:
                clean = re.sub(r'```json|```', '', text).strip()
                json_match = re.search(r'\{[^{}]*\}', clean, re.DOTALL)
                if json_match:
                    candidate = json.loads(json_match.group())
                    if isinstance(candidate, dict):
                        if "action" in candidate and "action_type" not in candidate:
                            candidate["action_type"] = candidate["action"]
                        if "action_type" in candidate:
                            action = candidate
            except (json.JSONDecodeError, TypeError):
                pass

            if action is None:
                rewards.append(round(score - 3.0, 2))
                continue

            action_type = action.get("action_type", "")

            if action_type not in VALID_ACTIONS:
                rewards.append(round(score - 3.0, 2))
                continue

            exp_action = correct_action[i] if correct_action and i < len(correct_action) else None
            exp_code = correct_code[i] if correct_code and i < len(correct_code) else None
            exp_severity = correct_severity[i] if correct_severity and i < len(correct_severity) else None
            exp_target = correct_target[i] if correct_target and i < len(correct_target) else None

            if exp_action:
                if action_type == exp_action:
                    score += 3.0
                else:
                    score -= 3.0

            if exp_code:
                if action.get("issue_code") == exp_code:
                    score += 3.0
                elif action.get("issue_code") in VALID_CODES:
                    score -= 1.0
                else:
                    score -= 2.0

            if exp_severity:
                if action.get("severity") == exp_severity:
                    score += 1.5
                else:
                    score -= 1.5

            if exp_target:
                if action.get("target") == exp_target:
                    score += 1.0
                else:
                    score -= 0.5

            note = action.get("note", "")
            if len(note) > 20:
                score += 0.5
            elif len(note) < 5:
                score -= 0.5

            rewards.append(round(score, 2))

        return rewards

    return reward_fn


def run_training(
    total_steps: int = 1000,
    checkpoint_every: int = 100,
    lora_r: int = 16,
    batch_size: int = 4,
    learning_rate: float = 1e-4,
    resume_from: str = None,
):
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    log_path = os.path.join(LOG_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    def log(msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    log("=" * 60)
    log("GovernanceReview-Gym v7 — Fixed JSON Parsing")
    log(f"Steps: {total_steps} | LoRA r={lora_r} | Batch: {batch_size} | LR: {learning_rate}")
    log("=" * 60)

    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model
        log(f"torch {torch.__version__} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    except ImportError as e:
        log(f"Missing: {e}")
        sys.exit(1)

    model_id = "Qwen/Qwen2.5-0.5B-Instruct"
    log(f"Loading {model_id}...")
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_r * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.1,
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Ready in {time.time()-t0:.1f}s | Trainable: {trainable:,} params")

    if resume_from and os.path.exists(resume_from):
        log(f"Resuming: {resume_from}")
        model.load_adapter(resume_from)

    log("Building real-world prompt dataset...")
    train_dataset = build_prompt_dataset(n_prompts=2000)
    log(f"Dataset ready: {len(train_dataset)} prompts")

    try:
        from trl import GRPOTrainer, GRPOConfig

        is_ampere_or_later = False
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            is_ampere_or_later = cap[0] >= 8

        grpo_config = GRPOConfig(
            output_dir=CHECKPOINT_DIR,
            per_device_train_batch_size=4,
            num_generations=4,
            max_completion_length=256,
            learning_rate=learning_rate,
            max_steps=total_steps,
            save_steps=checkpoint_every,
            logging_steps=5,
            gradient_checkpointing=True,
            bf16=is_ampere_or_later,
            fp16=not is_ampere_or_later,
            report_to="none",
            remove_unused_columns=False,
            temperature=0.9,
            top_p=0.95,
            beta=0.01,
        )

        reward_fn = make_reward_fn()

        trainer = GRPOTrainer(
            model=model,
            args=grpo_config,
            processing_class=tokenizer,
            train_dataset=train_dataset,
            reward_funcs=[reward_fn],
        )

        log("Starting GRPO v7 training...")
        log("Watch: reward_std >0.5, grad_norm >0, reward improving")
        trainer.train()
        trainer.save_model(os.path.join(CHECKPOINT_DIR, "final"))
        log("Done! Saved to checkpoints/final/")

    except Exception as e:
        log(f"Training error: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume_from", type=str, default=None)
    args = parser.parse_args()

    run_training(
        total_steps=args.total_steps,
        checkpoint_every=args.checkpoint_every,
        lora_r=args.lora_r,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        resume_from=args.resume_from,
    )
