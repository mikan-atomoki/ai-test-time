"""Baseline: 単純な1回生成."""

from __future__ import annotations

from src.client import VLLMClient

from .base import Algorithm, AlgorithmResult


class Baseline(Algorithm):
    """温度サンプリングで1回だけ生成するベースライン."""

    name = "Baseline"

    def __init__(
        self,
        client: VLLMClient,
        *,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ):
        super().__init__(client)
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, prompt: str) -> AlgorithmResult:
        results = self.client.generate(
            prompt,
            n=1,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            logprobs=True,
        )
        gen = results[0]
        avg_logprob = (
            sum(gen.logprobs) / len(gen.logprobs) if gen.logprobs else 0.0
        )
        return AlgorithmResult(
            algorithm_name=self.name,
            answer=gen.text,
            total_tokens=len(gen.logprobs) if gen.logprobs else len(gen.text.split()),
            metadata={"avg_logprob": avg_logprob},
        )
