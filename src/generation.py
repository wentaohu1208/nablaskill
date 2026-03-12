"""Response generation backends (vLLM and HuggingFace).

Extracted from ttso.py to keep file sizes manageable.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generate LM responses with skill context via vLLM or HuggingFace.

    Args:
        lm_model: Language model (used for HF generation).
        lm_tokenizer: Tokenizer with chat template.
        device: Computation device.
        vllm_url: Optional vLLM server URL.
        vllm_model_name: Model name for vLLM API.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_generation_len: Maximum new tokens to generate.
    """

    def __init__(
        self,
        lm_model: nn.Module,
        lm_tokenizer,
        device: torch.device,
        vllm_url: Optional[str] = None,
        vllm_model_name: Optional[str] = None,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_generation_len: int = 1024,
    ) -> None:
        self.lm_model = lm_model
        self.lm_tokenizer = lm_tokenizer
        self.device = device
        self.vllm_url = vllm_url
        self.vllm_model_name = vllm_model_name
        self.temperature = temperature
        self.top_p = top_p
        self.max_generation_len = max_generation_len

    def generate(
        self,
        query: str,
        skill_text: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate a response using the LM with skill context.

        Args:
            query: User query.
            skill_text: Skill instructions.
            system_prompt: Optional system prompt.
            seed: Random seed.

        Returns:
            Generated response text.
        """
        prompt = (
            f"Use the following skill to solve the problem:\n"
            f"{skill_text}\n\n"
            f"Problem: {query}"
        )

        if self.vllm_url:
            return self._generate_vllm(prompt, system_prompt, seed)
        return self._generate_hf(prompt, system_prompt, seed)

    def _generate_vllm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate via vLLM server."""
        import requests

        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        url = self.vllm_url.rstrip("/") + "/v1/chat/completions"
        body = {
            "model": self.vllm_model_name or "default",
            "messages": messages,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_generation_len,
            "stream": False,
        }
        if seed is not None:
            body["seed"] = seed

        try:
            resp = requests.post(url, json=body, timeout=600)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            raise RuntimeError(f"vLLM request failed: {e!r}") from e

    def _generate_hf(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Generate via HuggingFace model.generate()."""
        messages = [{"role": "user", "content": prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        text = self.lm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        input_ids = self.lm_tokenizer.encode(
            text, add_special_tokens=False, return_tensors="pt"
        ).to(self.device)

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        with torch.no_grad():
            outputs = self.lm_model.generate(
                input_ids,
                max_new_tokens=self.max_generation_len,
                do_sample=self.temperature > 0,
                temperature=self.temperature if self.temperature > 0 else None,
                top_p=self.top_p,
                pad_token_id=self.lm_tokenizer.eos_token_id,
            )

        new_tokens = outputs[0, input_ids.shape[1] :]
        return self.lm_tokenizer.decode(new_tokens, skip_special_tokens=True)
