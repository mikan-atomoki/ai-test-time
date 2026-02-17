"""MCTS (Monte Carlo Tree Search): 木構造で部分回答を探索."""

from __future__ import annotations

import math

from src.client import VLLMClient
from src.scoring import logprob_score

from .base import Algorithm, AlgorithmResult


class MCTSNode:
    """MCTSの探索ノード."""

    def __init__(self, text: str, logprobs: list[float], parent: MCTSNode | None = None):
        self.text = text
        self.logprobs = logprobs
        self.parent = parent
        self.children: list[MCTSNode] = []
        self.visits: int = 0
        self.total_value: float = 0.0

    @property
    def value(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.total_value / self.visits

    def ucb1(self, c: float = 1.414) -> float:
        """UCB1スコアを計算."""
        if self.visits == 0:
            return float("inf")
        parent_visits = self.parent.visits if self.parent else self.visits
        exploitation = self.value
        exploration = c * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def depth(self) -> int:
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d


class MCTS(Algorithm):
    """Monte Carlo Tree Search アルゴリズム."""

    name = "MCTS"

    def __init__(
        self,
        client: VLLMClient,
        *,
        num_iterations: int = 10,
        expansion_width: int = 3,
        c: float = 1.414,
        max_depth: int = 5,
        step_tokens: int = 50,
        rollout_tokens: int = 200,
        temperature: float = 0.8,
    ):
        super().__init__(client)
        self.num_iterations = num_iterations
        self.expansion_width = expansion_width
        self.c = c
        self.max_depth = max_depth
        self.step_tokens = step_tokens
        self.rollout_tokens = rollout_tokens
        self.temperature = temperature

    def run(self, prompt: str) -> AlgorithmResult:
        root = MCTSNode(text="", logprobs=[])
        total_tokens = 0

        for _ in range(self.num_iterations):
            # 1. Selection
            node = self._select(root)

            # 2. Expansion
            if node.depth() < self.max_depth:
                children, tokens = self._expand(prompt, node)
                total_tokens += tokens
                if children:
                    node = children[0]  # 最初の子ノードでシミュレーション

            # 3. Simulation
            rollout_score, tokens = self._simulate(prompt, node)
            total_tokens += tokens

            # 4. Backpropagation
            self._backpropagate(node, rollout_score)

        # 最良ノードを選択（最も訪問回数が多いルート直下の子）
        best = self._best_child(root)
        if best is None:
            # フォールバック: 通常の生成
            gen = self.client.generate(
                prompt, n=1, temperature=self.temperature, max_tokens=512, logprobs=True,
            )
            return AlgorithmResult(
                algorithm_name=self.name,
                answer=gen[0].text,
                total_tokens=total_tokens + len(gen[0].logprobs),
            )

        return AlgorithmResult(
            algorithm_name=self.name,
            answer=best.text,
            score=best.value,
            total_tokens=total_tokens,
            metadata={
                "iterations": self.num_iterations,
                "tree_size": self._count_nodes(root),
                "best_visits": best.visits,
                "best_value": best.value,
            },
        )

    def _select(self, node: MCTSNode) -> MCTSNode:
        """UCB1スコアで探索ノードを選択."""
        while not node.is_leaf():
            node = max(node.children, key=lambda n: n.ucb1(self.c))
        return node

    def _expand(self, prompt: str, node: MCTSNode) -> tuple[list[MCTSNode], int]:
        """選択ノードから複数の継続テキストを生成."""
        full_text = prompt + node.text
        results = self.client.generate(
            full_text,
            n=self.expansion_width,
            temperature=self.temperature,
            max_tokens=self.step_tokens,
            logprobs=True,
        )
        tokens_used = 0
        children: list[MCTSNode] = []
        for gen in results:
            child = MCTSNode(
                text=node.text + gen.text,
                logprobs=node.logprobs + gen.logprobs,
                parent=node,
            )
            node.children.append(child)
            children.append(child)
            tokens_used += len(gen.logprobs)
        return children, tokens_used

    def _simulate(self, prompt: str, node: MCTSNode) -> tuple[float, int]:
        """展開ノードから1回の完全生成でロールアウトし、スコアを返す."""
        full_text = prompt + node.text
        results = self.client.generate(
            full_text,
            n=1,
            temperature=self.temperature,
            max_tokens=self.rollout_tokens,
            logprobs=True,
        )
        gen = results[0]
        all_logprobs = node.logprobs + gen.logprobs
        score = logprob_score(all_logprobs)
        return score, len(gen.logprobs)

    def _backpropagate(self, node: MCTSNode, score: float) -> None:
        """スコアを親ノードまで伝播."""
        current: MCTSNode | None = node
        while current is not None:
            current.visits += 1
            current.total_value += score
            current = current.parent

    def _best_child(self, root: MCTSNode) -> MCTSNode | None:
        """ルート直下で最も訪問回数が多い子ノードを返す."""
        if not root.children:
            return None
        return max(root.children, key=lambda n: n.visits)

    def _count_nodes(self, node: MCTSNode) -> int:
        """木のノード数をカウント."""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
