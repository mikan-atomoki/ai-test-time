"""スコアリング関数: Self-evaluation / Logprob平均 / Length penalty / 多肢選択正解判定."""

from __future__ import annotations

import math
import re

from src.client import VLLMClient

SELF_EVAL_PROMPT = """以下の回答を1〜10のスコアで評価してください。
スコアのみを整数で回答してください。他のテキストは不要です。

質問: {question}
回答: {answer}

スコア:"""


def self_evaluation_score(
    client: VLLMClient, question: str, answer: str
) -> float:
    """モデル自身に回答を1-10で評価させる。0-1に正規化して返す."""
    response = client.chat(
        [
            {"role": "user", "content": SELF_EVAL_PROMPT.format(question=question, answer=answer)},
        ],
        temperature=0.0,
        max_tokens=16,
    )
    match = re.search(r"(\d+)", response)
    if match:
        score = int(match.group(1))
        return max(0.0, min(1.0, score / 10.0))
    return 0.5


def logprob_score(logprobs: list[float]) -> float:
    """log確率の平均を0-1のスコアに変換。値が高いほどモデルの自信度が高い."""
    if not logprobs:
        return 0.0
    avg = sum(logprobs) / len(logprobs)
    # logprobは負の値。0に近いほど自信が高い。
    # exp(avg_logprob) で0-1の確率に変換
    return math.exp(avg)


def length_penalty_score(
    text: str, target_length: int = 200, tolerance: float = 0.5
) -> float:
    """テキストの長さに基づくペナルティ。target_length付近で最大1.0を返す."""
    length = len(text.split())
    if length == 0:
        return 0.0
    ratio = length / target_length
    # ガウシアン風のペナルティ: target_lengthから離れるほどスコア低下
    return math.exp(-((ratio - 1.0) ** 2) / (2 * tolerance**2))


def extract_choice(text: str) -> str | None:
    """回答テキストからA/B/C/Dの選択肢を抽出する.

    以下のパターンを優先順に試行:
    1. 先頭付近の単独の A/B/C/D
    2. テキスト中の明示的なパターン (例: "答えはB", "回答: C")
    3. テキスト中の最初の単独 A/B/C/D
    """
    if not text or not text.strip():
        return None

    normalized = text.strip()

    # パターン1: 先頭が A/B/C/D で始まる（オプションでピリオドや括弧付き）
    m = re.match(r"^([A-Da-d])\s*[.。)）:：]?\s", normalized)
    if m:
        return m.group(1).upper()

    # テキスト全体が A/B/C/D 1文字のみ
    if re.fullmatch(r"[A-Da-d]", normalized):
        return normalized.upper()

    # パターン2: 明示的なキーワード付きパターン
    m = re.search(r"(?:答え|回答|正解|選択)[はがをの：:]\s*([A-Da-d])", normalized)
    if m:
        return m.group(1).upper()

    # パターン3: テキスト中の最初の単独 A/B/C/D（前後が非アルファベット）
    m = re.search(r"(?<![A-Za-z])([A-Da-d])(?![A-Za-z])", normalized)
    if m:
        return m.group(1).upper()

    return None


def accuracy_score(answer: str, correct_answer: str) -> float:
    """多肢選択の正解判定。正解なら1.0、不正解なら0.0."""
    extracted = extract_choice(answer)
    if extracted is None:
        return 0.0
    return 1.0 if extracted == correct_answer.upper() else 0.0


def combined_score(
    client: VLLMClient,
    question: str,
    answer: str,
    logprobs: list[float],
    *,
    correct_answer: str | None = None,
    w_self_eval: float = 0.5,
    w_logprob: float = 0.3,
    w_length: float = 0.2,
    target_length: int = 200,
) -> float:
    """スコアを計算。正解がある場合はaccuracy重視の重み配分に切替."""
    if correct_answer is not None:
        # 多肢選択式: accuracy 70%, logprob 20%, self_eval 10%
        s_accuracy = accuracy_score(answer, correct_answer)
        s_logprob = logprob_score(logprobs)
        s_eval = self_evaluation_score(client, question, answer)
        return 0.7 * s_accuracy + 0.2 * s_logprob + 0.1 * s_eval

    # 自由記述式: 従来の重み配分
    s_eval = self_evaluation_score(client, question, answer)
    s_logprob = logprob_score(logprobs)
    s_length = length_penalty_score(answer, target_length=target_length)
    return w_self_eval * s_eval + w_logprob * s_logprob + w_length * s_length
