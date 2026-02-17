"""ベンチマーク問題セット（多肢選択式）."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class BenchmarkProblem:
    """ベンチマーク問題."""

    id: str
    category: str
    prompt: str
    reference_answer: str | None = None
    choices: list[str] = field(default_factory=list)
    correct_answer: str | None = None


def _format_mc_prompt(question: str, choices: list[str]) -> str:
    """多肢選択式のプロンプトを生成."""
    lines = [question, ""]
    for label, choice in zip("ABCD", choices):
        lines.append(f"{label}. {choice}")
    lines.append("")
    lines.append("A/B/C/Dから1つ選んで回答してください。")
    return "\n".join(lines)


# 多肢選択式ベンチマーク問題集（20問）
BENCHMARK_PROBLEMS: list[BenchmarkProblem] = [
    # ===== 論理推論（5問） =====
    BenchmarkProblem(
        id="logic_01",
        category="論理推論",
        prompt=_format_mc_prompt(
            "AはBより背が高い。CはBより背が低い。AとCではどちらが背が高いですか？",
            ["Cの方が背が高い", "同じ背の高さ", "Aの方が背が高い", "情報が不足している"],
        ),
        choices=["Cの方が背が高い", "同じ背の高さ", "Aの方が背が高い", "情報が不足している"],
        correct_answer="C",
        reference_answer="Aの方が背が高い",
    ),
    BenchmarkProblem(
        id="logic_02",
        category="論理推論",
        prompt=_format_mc_prompt(
            "3つの箱があります。1つには金貨、1つには銀貨、1つは空です。"
            "箱Aには「金貨はここにない」、箱Bには「この箱は空です」、箱Cには「金貨は箱Aにある」と書かれています。"
            "ラベルが全て嘘の場合、金貨はどこにありますか？",
            ["箱B", "箱C", "箱A", "判別不能"],
        ),
        choices=["箱B", "箱C", "箱A", "判別不能"],
        correct_answer="C",
        reference_answer="金貨は箱Aにある",
    ),
    BenchmarkProblem(
        id="logic_03",
        category="論理推論",
        prompt=_format_mc_prompt(
            "「全ての犬は動物である」「ポチは犬である」この2つの前提から確実に言えることは？",
            ["ポチは猫ではない", "ポチは動物である", "全ての動物は犬である", "ポチは哺乳類である"],
        ),
        choices=["ポチは猫ではない", "ポチは動物である", "全ての動物は犬である", "ポチは哺乳類である"],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="logic_04",
        category="論理推論",
        prompt=_format_mc_prompt(
            "A→BかつB→Cが成り立つとき、確実に言えるのはどれですか？",
            ["C→A", "A→C", "B→A", "C→B"],
        ),
        choices=["C→A", "A→C", "B→A", "C→B"],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="logic_05",
        category="論理推論",
        prompt=_format_mc_prompt(
            "5人が一列に並んでいます。太郎は左から2番目、花子は右から2番目にいます。"
            "5人の列で太郎と花子の間には何人いますか？",
            ["0人", "1人", "2人", "3人"],
        ),
        choices=["0人", "1人", "2人", "3人"],
        correct_answer="B",
    ),
    # ===== 数学（5問） =====
    BenchmarkProblem(
        id="math_01",
        category="数学",
        prompt=_format_mc_prompt(
            "りんごが5個あります。3人で均等に分けると、1人何個で何個余りますか？",
            ["1人2個で1個余る", "1人1個で2個余る", "1人1個で3個余る", "1人2個で余りなし"],
        ),
        choices=["1人2個で1個余る", "1人1個で2個余る", "1人1個で3個余る", "1人2個で余りなし"],
        correct_answer="B",
        reference_answer="1人1個で2個余る",
    ),
    BenchmarkProblem(
        id="math_02",
        category="数学",
        prompt=_format_mc_prompt(
            "1から100までの整数の合計はいくつですか？",
            ["4950", "5050", "5150", "10000"],
        ),
        choices=["4950", "5050", "5150", "10000"],
        correct_answer="B",
        reference_answer="5050",
    ),
    BenchmarkProblem(
        id="math_03",
        category="数学",
        prompt=_format_mc_prompt(
            "2の10乗はいくつですか？",
            ["512", "1000", "1024", "2048"],
        ),
        choices=["512", "1000", "1024", "2048"],
        correct_answer="C",
    ),
    BenchmarkProblem(
        id="math_04",
        category="数学",
        prompt=_format_mc_prompt(
            "三角形の内角の和は何度ですか？",
            ["90度", "180度", "270度", "360度"],
        ),
        choices=["90度", "180度", "270度", "360度"],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="math_05",
        category="数学",
        prompt=_format_mc_prompt(
            "円の面積の公式はどれですか？（rは半径）",
            ["2πr", "πr²", "πd", "4πr²"],
        ),
        choices=["2πr", "πr²", "πd", "4πr²"],
        correct_answer="B",
    ),
    # ===== 知識（5問） =====
    BenchmarkProblem(
        id="knowledge_01",
        category="知識",
        prompt=_format_mc_prompt(
            "光合成で植物が吸収する気体はどれですか？",
            ["酸素", "窒素", "二酸化炭素", "水素"],
        ),
        choices=["酸素", "窒素", "二酸化炭素", "水素"],
        correct_answer="C",
    ),
    BenchmarkProblem(
        id="knowledge_02",
        category="知識",
        prompt=_format_mc_prompt(
            "HTTPSの「S」が意味するものはどれですか？",
            ["Speed", "Secure", "Server", "Standard"],
        ),
        choices=["Speed", "Secure", "Server", "Standard"],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="knowledge_03",
        category="知識",
        prompt=_format_mc_prompt(
            "地球から最も近い恒星は何ですか？",
            ["シリウス", "プロキシマ・ケンタウリ", "太陽", "ベテルギウス"],
        ),
        choices=["シリウス", "プロキシマ・ケンタウリ", "太陽", "ベテルギウス"],
        correct_answer="C",
    ),
    BenchmarkProblem(
        id="knowledge_04",
        category="知識",
        prompt=_format_mc_prompt(
            "DNAの二重らせん構造を発見した科学者は誰ですか？",
            [
                "アインシュタインとボーア",
                "ワトソンとクリック",
                "ニュートンとライプニッツ",
                "ダーウィンとメンデル",
            ],
        ),
        choices=[
            "アインシュタインとボーア",
            "ワトソンとクリック",
            "ニュートンとライプニッツ",
            "ダーウィンとメンデル",
        ],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="knowledge_05",
        category="知識",
        prompt=_format_mc_prompt(
            "世界で最も深い海溝はどれですか？",
            ["トンガ海溝", "マリアナ海溝", "日本海溝", "プエルトリコ海溝"],
        ),
        choices=["トンガ海溝", "マリアナ海溝", "日本海溝", "プエルトリコ海溝"],
        correct_answer="B",
    ),
    # ===== 語彙（5問） =====
    BenchmarkProblem(
        id="vocab_01",
        category="語彙",
        prompt=_format_mc_prompt(
            "「忖度」の意味として最も適切なものはどれですか？",
            [
                "正確に計算すること",
                "他人の気持ちを推し量ること",
                "物事を忘れること",
                "命令に従うこと",
            ],
        ),
        choices=[
            "正確に計算すること",
            "他人の気持ちを推し量ること",
            "物事を忘れること",
            "命令に従うこと",
        ],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="vocab_02",
        category="語彙",
        prompt=_format_mc_prompt(
            "「矛盾」の語源となった故事で、商人が売っていたものは何ですか？",
            ["刀と鎧", "矛と盾", "弓と矢", "剣と兜"],
        ),
        choices=["刀と鎧", "矛と盾", "弓と矢", "剣と兜"],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="vocab_03",
        category="語彙",
        prompt=_format_mc_prompt(
            "「塞翁が馬」のことわざの意味として正しいものはどれですか？",
            [
                "馬は大切にすべき",
                "人生の幸不幸は予測できない",
                "年寄りの知恵は尊い",
                "急いては事を仕損じる",
            ],
        ),
        choices=[
            "馬は大切にすべき",
            "人生の幸不幸は予測できない",
            "年寄りの知恵は尊い",
            "急いては事を仕損じる",
        ],
        correct_answer="B",
    ),
    BenchmarkProblem(
        id="vocab_04",
        category="語彙",
        prompt=_format_mc_prompt(
            "「推敲」という言葉の由来に関係する動作はどれですか？",
            ["門を押すか敲くか迷った", "石を推すか投げるか迷った", "筆を持つか置くか迷った", "道を進むか戻るか迷った"],
        ),
        choices=["門を押すか敲くか迷った", "石を推すか投げるか迷った", "筆を持つか置くか迷った", "道を進むか戻るか迷った"],
        correct_answer="A",
    ),
    BenchmarkProblem(
        id="vocab_05",
        category="語彙",
        prompt=_format_mc_prompt(
            "「蛇足」の意味として正しいものはどれですか？",
            [
                "余計な付け足し",
                "非常に長いもの",
                "素早い行動",
                "不気味なもの",
            ],
        ),
        choices=[
            "余計な付け足し",
            "非常に長いもの",
            "素早い行動",
            "不気味なもの",
        ],
        correct_answer="A",
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
