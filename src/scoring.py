"""スコアリング関数: Self-evaluation / Logprob平均 / Length penalty."""

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


def combined_score(
    client: VLLMClient,
    question: str,
    answer: str,
    logprobs: list[float],
    *,
    w_self_eval: float = 0.5,
    w_logprob: float = 0.3,
    w_length: float = 0.2,
    target_length: int = 200,
) -> float:
    """3つの指標の重み付き合計スコアを計算."""
    s_eval = self_evaluation_score(client, question, answer)
    s_logprob = logprob_score(logprobs)
    s_length = length_penalty_score(answer, target_length=target_length)
    return w_self_eval * s_eval + w_logprob * s_logprob + w_length * s_length
