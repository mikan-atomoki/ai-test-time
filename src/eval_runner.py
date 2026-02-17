"""評価実行エンジン."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.algorithms.base import Algorithm, AlgorithmResult
from src.benchmarks.datasets import BenchmarkProblem
from src.client import VLLMClient
from src.scoring import combined_score


@dataclass
class EvalResult:
    """1つの問題に対する全アルゴリズムの評価結果."""

    problem: BenchmarkProblem
    results: dict[str, AlgorithmResult] = field(default_factory=dict)


@dataclass
class EvalSummary:
    """全問題の評価サマリー."""

    eval_results: list[EvalResult] = field(default_factory=list)
    total_latency_sec: float = 0.0


class EvalRunner:
    """評価実行エンジン: 各アルゴリズムでベンチマーク問題を実行し比較."""

    def __init__(self, client: VLLMClient, algorithms: list[Algorithm]):
        self.client = client
        self.algorithms = algorithms

    def run_single(self, prompt: str) -> dict[str, AlgorithmResult]:
        """単一プロンプトに対して全アルゴリズムを実行."""
        results: dict[str, AlgorithmResult] = {}
        for algo in self.algorithms:
            start = time.perf_counter()
            result = algo.run(prompt)
            result.latency_sec = time.perf_counter() - start
            # スコアリング
            result.score = combined_score(
                self.client, prompt, result.answer, result.metadata.get("logprobs", []),
            )
            results[algo.name] = result
        return results

    def run_benchmark(self, problems: list[BenchmarkProblem]) -> EvalSummary:
        """ベンチマーク問題セットに対して全アルゴリズムを実行."""
        summary = EvalSummary()
        total_start = time.perf_counter()

        for problem in problems:
            eval_result = EvalResult(problem=problem)
            for algo in self.algorithms:
                start = time.perf_counter()
                result = algo.run(problem.prompt)
                result.latency_sec = time.perf_counter() - start
                result.score = combined_score(
                    self.client,
                    problem.prompt,
                    result.answer,
                    result.metadata.get("logprobs", []),
                )
                eval_result.results[algo.name] = result
            summary.eval_results.append(eval_result)

        summary.total_latency_sec = time.perf_counter() - total_start
        return summary
