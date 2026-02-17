"""アルゴリズムのユニットテスト."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.client import GenerationResult, VLLMClient
from src.algorithms.baseline import Baseline
from src.algorithms.beam_search import BeamSearch
from src.algorithms.mcts import MCTS, MCTSNode
from src.scoring import (
    logprob_score,
    length_penalty_score,
    extract_choice,
    accuracy_score,
)


def make_mock_client() -> VLLMClient:
    """モックvLLMクライアントを作成."""
    client = MagicMock(spec=VLLMClient)
    client.generate.return_value = [
        GenerationResult(
            text="テスト回答です。",
            logprobs=[-0.5, -0.3, -0.2, -0.1, -0.4],
            finish_reason="stop",
        )
    ]
    client.chat.return_value = "7"
    return client


class TestBaseline:
    def test_run_returns_result(self):
        client = make_mock_client()
        algo = Baseline(client, temperature=0.7, max_tokens=100)
        result = algo.run("テストプロンプト")

        assert result.algorithm_name == "Baseline"
        assert result.answer == "テスト回答です。"
        assert result.total_tokens == 5
        client.generate.assert_called_once()

    def test_run_with_empty_logprobs(self):
        client = make_mock_client()
        client.generate.return_value = [
            GenerationResult(text="回答", logprobs=[], finish_reason="stop")
        ]
        algo = Baseline(client)
        result = algo.run("テスト")

        assert result.metadata["avg_logprob"] == 0.0


class TestBeamSearch:
    def test_run_returns_result(self):
        client = make_mock_client()
        # beam_width=2なので初回は2個返す
        client.generate.return_value = [
            GenerationResult(text="候補A。", logprobs=[-0.2, -0.1], finish_reason="stop"),
            GenerationResult(text="候補B。", logprobs=[-0.5, -0.3], finish_reason="stop"),
        ]
        algo = BeamSearch(client, beam_width=2, max_steps=2, step_tokens=30)
        result = algo.run("テスト")

        assert result.algorithm_name == "Beam Search"
        assert result.answer is not None
        assert result.total_tokens > 0

    def test_beam_width_preserved(self):
        client = make_mock_client()
        client.generate.return_value = [
            GenerationResult(text="A", logprobs=[-0.1], finish_reason=None),
            GenerationResult(text="B", logprobs=[-0.2], finish_reason=None),
            GenerationResult(text="C", logprobs=[-0.3], finish_reason=None),
        ]
        algo = BeamSearch(client, beam_width=3, max_steps=1, step_tokens=30)
        result = algo.run("テスト")
        assert result.answer is not None


class TestMCTS:
    def test_run_returns_result(self):
        client = make_mock_client()
        client.generate.return_value = [
            GenerationResult(text="MCTS回答。", logprobs=[-0.3, -0.2], finish_reason="stop"),
        ]
        algo = MCTS(client, num_iterations=3, expansion_width=1, max_depth=2)
        result = algo.run("テスト")

        assert result.algorithm_name == "MCTS"
        assert result.answer is not None

    def test_node_ucb1(self):
        parent = MCTSNode(text="", logprobs=[])
        parent.visits = 10
        parent.total_value = 5.0

        child = MCTSNode(text="child", logprobs=[], parent=parent)
        child.visits = 3
        child.total_value = 1.5

        ucb = child.ucb1(c=1.414)
        assert ucb > child.value  # 探索項が加算されるのでvalueより大きい

    def test_node_unvisited_has_inf_ucb(self):
        node = MCTSNode(text="", logprobs=[])
        assert node.ucb1() == float("inf")

    def test_node_depth(self):
        root = MCTSNode(text="", logprobs=[])
        child = MCTSNode(text="a", logprobs=[], parent=root)
        grandchild = MCTSNode(text="ab", logprobs=[], parent=child)

        assert root.depth() == 0
        assert child.depth() == 1
        assert grandchild.depth() == 2


class TestScoring:
    def test_logprob_score_empty(self):
        assert logprob_score([]) == 0.0

    def test_logprob_score_values(self):
        score = logprob_score([-0.1, -0.2, -0.3])
        assert 0.0 < score < 1.0

    def test_logprob_score_perfect(self):
        # logprob=0 は確率1（完全な自信）
        score = logprob_score([0.0, 0.0, 0.0])
        assert score == pytest.approx(1.0)

    def test_length_penalty_at_target(self):
        # ちょうどtarget_lengthの文字数 → スコア最大
        text = " ".join(["word"] * 200)
        score = length_penalty_score(text, target_length=200)
        assert score == pytest.approx(1.0)

    def test_length_penalty_empty(self):
        assert length_penalty_score("") == 0.0

    def test_length_penalty_far_from_target(self):
        text = " ".join(["word"] * 10)
        score = length_penalty_score(text, target_length=200)
        assert score < 0.5  # targetから遠いのでスコア低い


class TestExtractChoice:
    """extract_choice() のテスト."""

    def test_single_letter(self):
        assert extract_choice("A") == "A"
        assert extract_choice("B") == "B"
        assert extract_choice("C") == "C"
        assert extract_choice("D") == "D"

    def test_single_letter_lowercase(self):
        assert extract_choice("a") == "A"
        assert extract_choice("b") == "B"

    def test_letter_with_period(self):
        assert extract_choice("B. 正解は5050です") == "B"

    def test_letter_with_explanation(self):
        assert extract_choice("C です。二酸化炭素を吸収します。") == "C"

    def test_keyword_pattern(self):
        assert extract_choice("答えはBです") == "B"
        assert extract_choice("正解はC") == "C"
        assert extract_choice("回答はA") == "A"

    def test_embedded_letter(self):
        assert extract_choice("私の回答はDだと思います") == "D"

    def test_empty_input(self):
        assert extract_choice("") is None
        assert extract_choice("   ") is None

    def test_no_valid_choice(self):
        # E以降は選択肢にないので抽出されない
        assert extract_choice("Eが正解") is None

    def test_does_not_match_mid_word(self):
        # "ABC" のような連続英字の中のB/Cを誤抽出しない
        assert extract_choice("ABCコーポレーション") is None


class TestAccuracyScore:
    """accuracy_score() のテスト."""

    def test_correct(self):
        assert accuracy_score("B", "B") == 1.0

    def test_correct_with_explanation(self):
        assert accuracy_score("答えはBです。理由は...", "B") == 1.0

    def test_incorrect(self):
        assert accuracy_score("A", "B") == 0.0

    def test_no_choice_extracted(self):
        assert accuracy_score("わかりません", "B") == 0.0

    def test_case_insensitive_correct_answer(self):
        assert accuracy_score("b", "B") == 1.0
        assert accuracy_score("B", "b") == 1.0
