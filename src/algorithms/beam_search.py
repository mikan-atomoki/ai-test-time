"""Beam Search: ステップ単位で複数候補を維持しながら生成."""

from __future__ import annotations

from dataclasses import dataclass

from src.client import VLLMClient
from src.scoring import logprob_score

from .base import Algorithm, AlgorithmResult


@dataclass
class Beam:
    """ビーム候補."""

    text: str
    logprobs: list[float]
    score: float = 0.0


class BeamSearch(Algorithm):
    """Beam Searchアルゴリズム."""

    name = "Beam Search"

    def __init__(
        self,
        client: VLLMClient,
        *,
        beam_width: int = 3,
        max_steps: int = 3,
        step_tokens: int = 50,
        temperature: float = 0.7,
    ):
        super().__init__(client)
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.step_tokens = step_tokens
        self.temperature = temperature

    def run(self, prompt: str) -> AlgorithmResult:
        # 初期ビーム: プロンプトからbeam_width個の候補を生成
        initial = self.client.generate(
            prompt,
            n=self.beam_width,
            temperature=self.temperature,
            max_tokens=self.step_tokens,
            logprobs=True,
        )
        beams = [
            Beam(
                text=gen.text,
                logprobs=gen.logprobs,
                score=logprob_score(gen.logprobs),
            )
            for gen in initial
        ]

        total_tokens = sum(len(b.logprobs) for b in beams)

        for _step in range(1, self.max_steps):
            all_candidates: list[Beam] = []

            for beam in beams:
                # 各ビームから次ステップを展開
                continuations = self.client.generate(
                    prompt + beam.text,
                    n=self.beam_width,
                    temperature=self.temperature,
                    max_tokens=self.step_tokens,
                    logprobs=True,
                )
                for gen in continuations:
                    combined_logprobs = beam.logprobs + gen.logprobs
                    all_candidates.append(
                        Beam(
                            text=beam.text + gen.text,
                            logprobs=combined_logprobs,
                            score=logprob_score(combined_logprobs),
                        )
                    )
                    total_tokens += len(gen.logprobs)

            # 上位beam_width個を保持
            all_candidates.sort(key=lambda b: b.score, reverse=True)
            beams = all_candidates[: self.beam_width]

            # 全候補が完了していれば終了
            if all(b.text.rstrip().endswith((".", "。", "\n")) for b in beams):
                break

        # 最終候補からベストを選択
        best = max(beams, key=lambda b: b.score)

        return AlgorithmResult(
            algorithm_name=self.name,
            answer=best.text,
            score=best.score,
            total_tokens=total_tokens,
            metadata={
                "avg_logprob": (
                    sum(best.logprobs) / len(best.logprobs)
                    if best.logprobs
                    else 0.0
                ),
                "num_beams_explored": len(beams),
            },
        )
