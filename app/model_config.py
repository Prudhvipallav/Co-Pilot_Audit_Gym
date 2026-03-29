"""
Flexible model configuration.

Provider routing:
  local      — Qwen2.5-0.5B-Instruct via transformers (no API key, CPU-friendly)
  openai     — OpenAI API (gpt-4o, gpt-4o-mini, etc.)
  anthropic  — Anthropic API (claude-haiku, claude-sonnet, etc.)
  huggingface — HF Inference API (Phi-3, Gemma, Mistral, etc.)
  ollama     — Local Ollama server

Edit config.yaml to swap models. Never touch agent code.
"""

import os
import json
import yaml
from typing import Optional

# ─── Model cache (local models stay loaded in memory) ───────────
_local_pipelines = {}


def load_config(path: str = "config.yaml") -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return yaml.safe_load(f)
    return _default_config()


def _default_config() -> dict:
    """Fallback if config.yaml is missing."""
    return {
        "reviewer": {
            "provider": "local",
            "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
            "max_new_tokens": 512,
            "temperature": 0.1,
        },
        "problem_maker": {
            "provider": "openai",
            "model_id": "gpt-4o-mini",
            "max_tokens": 2048,
            "temperature": 0.7,
            "api_key_env": "OPENAI_API_KEY",
        },
        "judge": {
            "provider": "openai",
            "model_id": "gpt-4o",
            "max_tokens": 1024,
            "temperature": 0.1,
            "api_key_env": "OPENAI_API_KEY",
        }
    }


class ModelClient:
    """Unified client that routes to the right provider."""

    def __init__(self, config: dict):
        self.provider = config["provider"]
        self.model_id = config["model_id"]
        self.max_tokens = config.get("max_tokens", config.get("max_new_tokens", 512))
        self.temperature = config.get("temperature", 0.1)
        self.api_key = os.getenv(config.get("api_key_env", ""), "")

    def complete(self, system: str, user: str) -> str:
        """Single-turn completion. Returns response text."""
        if self.provider == "local":
            return self._local(system, user)
        elif self.provider == "openai":
            return self._openai(system, user)
        elif self.provider == "google":
            return self._google(system, user)
        elif self.provider == "anthropic":
            return self._anthropic(system, user)
        elif self.provider == "huggingface":
            return self._huggingface(system, user)
        elif self.provider == "ollama":
            return self._ollama(system, user)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _google(self, system: str, user: str) -> str:
        """Google Gemini API."""
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model_id)
        
        # Format as chat for system instruction support
        chat = model.start_chat(history=[])
        response = chat.send_message(f"{system}\n\n{user}")
        return response.text.strip()

    def _local(self, system: str, user: str) -> str:
        """
        Local inference via transformers pipeline.
        Model is cached after first load — fast on subsequent calls.
        Uses Qwen2.5-0.5B-Instruct chat template.
        Supports LoRA adapters if lora_path is set in config.
        """
        global _local_pipelines

        cache_key = self.model_id
        lora_path = load_config().get("reviewer", {}).get("lora_path")
        if lora_path:
            cache_key = f"{self.model_id}@{lora_path}"

        if cache_key not in _local_pipelines:
            try:
                from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
                import torch

                print(f"[ModelClient] Loading {self.model_id} locally...")
                
                # Use lora_path for tokenizer too if it has the config files
                tokenizer_id = self.model_id
                if lora_path and os.path.exists(os.path.join(lora_path, "tokenizer_config.json")):
                    print(f"[ModelClient] Loading tokenizer from {lora_path}...")
                    tokenizer_id = lora_path

                tokenizer = AutoTokenizer.from_pretrained(
                    tokenizer_id, 
                    trust_remote_code=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    torch_dtype=torch.float32,  # CPU-safe
                    device_map="auto",
                    trust_remote_code=True
                )

                if lora_path and os.path.exists(lora_path):
                    from peft import PeftModel
                    print(f"[ModelClient] Applying LoRA adapter from {lora_path}...")
                    model = PeftModel.from_pretrained(model, lora_path)
                    model = model.merge_and_unload() # Merge for inference speed

                pipe = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    max_new_tokens=self.max_tokens,
                    temperature=self.temperature,
                    do_sample=(self.temperature > 0),
                    return_full_text=False
                )
                _local_pipelines[cache_key] = {"pipe": pipe, "tokenizer": tokenizer}
                print(f"[ModelClient] ✅ {cache_key} loaded and cached.")
            except ImportError:
                raise RuntimeError(
                    "transformers, torch, and peft required for local inference.\n"
                    "Run: pip install transformers torch peft"
                )

        cached = _local_pipelines[cache_key]
        pipe = cached["pipe"]
        tokenizer = cached["tokenizer"]

        # Build chat-format prompt using Qwen's template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        # Apply chat template if available
        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = pipe(prompt)
        except Exception:
            # Fallback: raw concatenation
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            outputs = pipe(prompt)

        return outputs[0]["generated_text"].strip()

    def _openai(self, system: str, user: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content

    def _anthropic(self, system: str, user: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        msg = client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}]
        )
        return msg.content[0].text

    def _huggingface(self, system: str, user: str) -> str:
        """HuggingFace Inference API."""
        import requests as req
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "inputs": f"<s>[INST] {system}\n\n{user} [/INST]",
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature,
                "return_full_text": False
            }
        }
        resp = req.post(
            f"https://api-inference.huggingface.co/models/{self.model_id}",
            headers=headers, json=payload, timeout=60
        )
        resp.raise_for_status()
        result = resp.json()
        if isinstance(result, list):
            return result[0].get("generated_text", "")
        return result.get("generated_text", "")

    def _ollama(self, system: str, user: str) -> str:
        """Local Ollama server."""
        import requests as req
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        resp = req.post(
            f"{base_url}/api/generate",
            json={
                "model": self.model_id,
                "prompt": f"{system}\n\n{user}",
                "stream": False,
                "options": {"temperature": self.temperature, "num_predict": self.max_tokens}
            },
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()["response"]


# ─── Singleton accessors ─────────────────────────────────────────
_cfg = None


def _get_cfg():
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg


def get_reviewer_client() -> ModelClient:
    return ModelClient(_get_cfg()["reviewer"])


def get_problem_maker_client() -> ModelClient:
    return ModelClient(_get_cfg()["problem_maker"])


def get_judge_client() -> ModelClient:
    return ModelClient(_get_cfg()["judge"])
