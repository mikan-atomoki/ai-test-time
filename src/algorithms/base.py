"""アルゴリズム共通インターフェース."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.client import VLLMClient


@dataclass
class AlgorithmResult:
    """アルゴリズム実行結果."""

    algorithm_name: str
    answer: str
    score: float = 0.0
    total_tokens: int = 0
    latency_sec: float = 0.0
    metadata: dict = field(default_factory=dict)


class Algorithm(ABC):
    """全アルゴリズムの基底クラス."""

    name: str = "base"

    def __init__(self, client: VLLMClient):
        self.client = client

    @abstractmethod
    def run(self, prompt: str) -> AlgorithmResult:
        """プロンプトに対してアルゴリズムを実行し結果を返す."""
        ...

    def _timed_run(self, prompt: str) -> AlgorithmResult:
        """実行時間を計測しながらrunを呼ぶユーティリティ."""
        start = time.perf_counter()
        result = self.run(prompt)
        result.latency_sec = time.perf_counter() - start
        return result
