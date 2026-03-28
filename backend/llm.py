"""
Central LLM — uses HuggingFace Router with OpenAI-compatible API.
Provider: featherless-ai
Model: HuggingFaceH4/zephyr-7b-beta
"""

import os
from typing import Optional, List, Any
from langchain_core.language_models.llms import LLM


class ZephyrLLM(LLM):
    temperature: float = 0.4
    max_new_tokens: int = 512
    api_token: str = ""

    @property
    def _llm_type(self) -> str:
        return "zephyr_hf_router"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_token,
        )
        completion = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        return completion.choices[0].message.content.strip()


def get_llm(temperature: float = 0.4, max_new_tokens: int = 512) -> ZephyrLLM:
    token = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise EnvironmentError("HUGGINGFACEHUB_API_TOKEN not set.")
    return ZephyrLLM(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        api_token=token,
    )
