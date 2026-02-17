"""ベンチマーク問題セット."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BenchmarkProblem:
    """ベンチマーク問題."""

    id: str
    category: str
    prompt: str
    reference_answer: str | None = None


# 推論・知識を問うベンチマーク問題集
BENCHMARK_PROBLEMS: list[BenchmarkProblem] = [
    # 論理推論
    BenchmarkProblem(
        id="logic_01",
        category="論理推論",
        prompt="AはBより背が高い。CはBより背が低い。AとCではどちらが背が高いですか？ステップバイステップで考えてください。",
        reference_answer="Aの方が背が高い",
    ),
    BenchmarkProblem(
        id="logic_02",
        category="論理推論",
        prompt="3つの箱があります。1つには金貨、1つには銀貨、1つは空です。箱Aには「金貨はここにない」と書かれています。箱Bには「この箱は空です」と書かれています。箱Cには「金貨は箱Aにある」と書かれています。ラベルが全て嘘の場合、金貨はどこにありますか？",
        reference_answer="金貨は箱Aにある",
    ),
    # 数学
    BenchmarkProblem(
        id="math_01",
        category="数学",
        prompt="りんごが5個あります。3人で均等に分けると、1人何個で何個余りますか？",
        reference_answer="1人1個で2個余る",
    ),
    BenchmarkProblem(
        id="math_02",
        category="数学",
        prompt="1から100までの整数の合計はいくつですか？計算過程も示してください。",
        reference_answer="5050",
    ),
    # 知識
    BenchmarkProblem(
        id="knowledge_01",
        category="知識",
        prompt="光合成の仕組みを簡潔に説明してください。",
        reference_answer=None,
    ),
    BenchmarkProblem(
        id="knowledge_02",
        category="知識",
        prompt="HTTPとHTTPSの違いを説明してください。",
        reference_answer=None,
    ),
    # コード生成
    BenchmarkProblem(
        id="code_01",
        category="コード生成",
        prompt="Pythonでフィボナッチ数列の最初の10個を出力するコードを書いてください。",
        reference_answer=None,
    ),
    BenchmarkProblem(
        id="code_02",
        category="コード生成",
        prompt="Pythonでリストの重複要素を除去する関数を書いてください。順序は保持してください。",
        reference_answer=None,
    ),
    # 要約・文章力
    BenchmarkProblem(
        id="summary_01",
        category="要約",
        prompt="機械学習とは何かを、小学生にもわかるように3文以内で説明してください。",
        reference_answer=None,
    ),
    BenchmarkProblem(
        id="creative_01",
        category="創作",
        prompt="「時間」をテーマにした俳句を1つ詠んでください。",
        reference_answer=None,
    ),
]


def get_problems(category: str | None = None) -> list[BenchmarkProblem]:
    """カテゴリでフィルタしてベンチマーク問題を取得."""
    if category is None:
        return BENCHMARK_PROBLEMS
    return [p for p in BENCHMARK_PROBLEMS if p.category == category]


def get_categories() -> list[str]:
    """利用可能なカテゴリ一覧を取得."""
    return sorted(set(p.category for p in BENCHMARK_PROBLEMS))
