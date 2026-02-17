"""Streamlit UI: AI Test-Time Compute 評価システム."""

from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go

from src.client import VLLMClient
from src.algorithms.baseline import Baseline
from src.algorithms.beam_search import BeamSearch
from src.algorithms.mcts import MCTS
from src.algorithms.base import Algorithm, AlgorithmResult
from src.benchmarks.datasets import get_problems, get_categories
from src.eval_runner import EvalRunner

st.set_page_config(page_title="AI Test-Time Compute 評価", layout="wide")
st.title("AI Test-Time Compute 評価システム")

# --- サイドバー ---
with st.sidebar:
    st.header("設定")
    vllm_url = st.text_input("vLLM サーバーURL", value="http://localhost:8000/v1")
    model_name = st.text_input("モデル名", value="Qwen/Qwen2.5-0.5B-Instruct")

    st.subheader("アルゴリズム選択")
    use_baseline = st.checkbox("Baseline", value=True)
    use_beam = st.checkbox("Beam Search", value=True)
    use_mcts = st.checkbox("MCTS", value=True)

    st.subheader("Beam Search パラメータ")
    beam_width = st.slider("beam_width", 2, 10, 3)
    beam_steps = st.slider("max_steps", 1, 10, 3)
    beam_step_tokens = st.slider("step_tokens (Beam)", 20, 200, 50)

    st.subheader("MCTS パラメータ")
    mcts_iterations = st.slider("num_iterations", 5, 50, 10)
    mcts_expansion = st.slider("expansion_width", 2, 5, 3)
    mcts_c = st.slider("探索係数 C", 0.5, 3.0, 1.414, step=0.1)
    mcts_max_depth = st.slider("max_depth", 2, 10, 5)
    mcts_step_tokens = st.slider("step_tokens (MCTS)", 20, 200, 50)

    st.subheader("共通パラメータ")
    temperature = st.slider("temperature", 0.0, 2.0, 0.7, step=0.1)
    max_tokens = st.slider("max_tokens", 64, 1024, 512)


def build_algorithms(client: VLLMClient) -> list[Algorithm]:
    """UIの設定に基づいてアルゴリズムインスタンスを構築."""
    algos: list[Algorithm] = []
    if use_baseline:
        algos.append(Baseline(client, temperature=temperature, max_tokens=max_tokens))
    if use_beam:
        algos.append(
            BeamSearch(
                client,
                beam_width=beam_width,
                max_steps=beam_steps,
                step_tokens=beam_step_tokens,
                temperature=temperature,
            )
        )
    if use_mcts:
        algos.append(
            MCTS(
                client,
                num_iterations=mcts_iterations,
                expansion_width=mcts_expansion,
                c=mcts_c,
                max_depth=mcts_max_depth,
                step_tokens=mcts_step_tokens,
                temperature=temperature,
            )
        )
    return algos


# --- 入力モード切替 ---
input_mode = st.radio("入力モード", ["自由入力", "ベンチマーク"], horizontal=True)

if input_mode == "自由入力":
    prompt = st.text_area("プロンプトを入力", height=120, placeholder="質問や指示を入力してください...")
    run_button = st.button("実行", type="primary", use_container_width=True)

    if run_button and prompt:
        client = VLLMClient(base_url=vllm_url, model=model_name)
        algos = build_algorithms(client)

        if not algos:
            st.warning("アルゴリズムを1つ以上選択してください。")
        else:
            runner = EvalRunner(client, algos)
            with st.spinner("生成中..."):
                results = runner.run_single(prompt)

            # 結果表示
            cols = st.columns(len(results))
            for col, (name, result) in zip(cols, results.items()):
                with col:
                    st.subheader(name)
                    st.markdown("**回答:**")
                    st.text_area(f"{name} 回答", result.answer, height=200, disabled=True, label_visibility="collapsed")
                    st.metric("スコア", f"{result.score:.3f}")
                    st.metric("レイテンシ", f"{result.latency_sec:.2f}s")
                    st.metric("トークン数", result.total_tokens)

            # メトリクス比較チャート
            st.subheader("メトリクス比較")
            names = list(results.keys())
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="スコア",
                x=names,
                y=[r.score for r in results.values()],
            ))
            fig.add_trace(go.Bar(
                name="レイテンシ (s)",
                x=names,
                y=[r.latency_sec for r in results.values()],
            ))
            fig.add_trace(go.Bar(
                name="トークン数 (÷100)",
                x=names,
                y=[r.total_tokens / 100 for r in results.values()],
            ))
            fig.update_layout(barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)

elif input_mode == "ベンチマーク":
    categories = ["全て"] + get_categories()
    selected_cat = st.selectbox("カテゴリ", categories)
    category_filter = None if selected_cat == "全て" else selected_cat
    problems = get_problems(category_filter)

    st.info(f"{len(problems)} 問のベンチマーク問題が見つかりました")

    if st.button("ベンチマーク実行", type="primary", use_container_width=True):
        client = VLLMClient(base_url=vllm_url, model=model_name)
        algos = build_algorithms(client)

        if not algos:
            st.warning("アルゴリズムを1つ以上選択してください。")
        else:
            runner = EvalRunner(client, algos)
            progress = st.progress(0)

            all_results = []
            for i, problem in enumerate(problems):
                with st.spinner(f"[{i + 1}/{len(problems)}] {problem.id} を評価中..."):
                    results = runner.run_single(problem.prompt)
                    all_results.append((problem, results))
                progress.progress((i + 1) / len(problems))

            # 結果一覧テーブル
            st.subheader("結果一覧")
            for problem, results in all_results:
                with st.expander(f"[{problem.category}] {problem.id}: {problem.prompt[:60]}..."):
                    cols = st.columns(len(results))
                    for col, (name, result) in zip(cols, results.items()):
                        with col:
                            st.markdown(f"**{name}**")
                            st.text_area(
                                f"{problem.id}_{name}",
                                result.answer,
                                height=150,
                                disabled=True,
                                label_visibility="collapsed",
                            )
                            st.caption(
                                f"スコア: {result.score:.3f} | "
                                f"レイテンシ: {result.latency_sec:.2f}s | "
                                f"トークン: {result.total_tokens}"
                            )

            # 集計チャート
            st.subheader("アルゴリズム別 平均スコア")
            algo_names = [a.name for a in algos]
            avg_scores = {}
            avg_latencies = {}
            for algo_name in algo_names:
                scores = [r[algo_name].score for _, r in all_results if algo_name in r]
                latencies = [r[algo_name].latency_sec for _, r in all_results if algo_name in r]
                avg_scores[algo_name] = sum(scores) / len(scores) if scores else 0
                avg_latencies[algo_name] = sum(latencies) / len(latencies) if latencies else 0

            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="平均スコア",
                x=list(avg_scores.keys()),
                y=list(avg_scores.values()),
            ))
            fig.add_trace(go.Bar(
                name="平均レイテンシ (s)",
                x=list(avg_latencies.keys()),
                y=list(avg_latencies.values()),
            ))
            fig.update_layout(barmode="group", height=400)
            st.plotly_chart(fig, use_container_width=True)
