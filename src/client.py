"""vLLM OpenAI互換APIクライアント."""

from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI


@dataclass
class GenerationResult:
    """1つの生成結果."""

    text: str
    logprobs: list[float] = field(default_factory=list)
    finish_reason: str | None = None


class VLLMClient:
    """vLLMサーバーへのOpenAI互換クライアント."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "default"):
        self.client = OpenAI(base_url=base_url, api_key="dummy")
        self.model = model

    def generate(
        self,
        prompt: str,
        *,
        n: int = 1,
        temperature: float = 0.7,
        top_p: float = 1.0,
        max_tokens: int = 512,
        logprobs: bool = False,
        top_logprobs: int | None = None,
        stop: list[str] | None = None,
    ) -> list[GenerationResult]:
        """テキスト生成を実行し結果を返す."""
        response = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            n=n,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            logprobs=1 if logprobs else None,
            stop=stop,
        )
        results: list[GenerationResult] = []
        for choice in response.choices:
            token_logprobs: list[float] = []
            if logprobs and choice.logprobs and choice.logprobs.token_logprobs:
                token_logprobs = [
                    lp for lp in choice.logprobs.token_logprobs if lp is not None
                ]
            results.append(
                GenerationResult(
                    text=choice.text,
                    logprobs=token_logprobs,
                    finish_reason=choice.finish_reason,
                )
            )
        return results

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        """Chat completions API でメッセージを送り応答テキストを返す（スコアリング用）."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
