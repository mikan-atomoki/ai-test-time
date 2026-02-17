"""Streamlit UI: AI Test-Time Compute 評価システム."""

from __future__ import annotations

from collections import defaultdict

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

    st.info(f"{len(problems)} 問の多肢選択式ベンチマーク問題が見つかりました")

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
                    results = runner.run_single(
                        problem.prompt, correct_answer=problem.correct_answer
                    )
                    all_results.append((problem, results))
                progress.progress((i + 1) / len(problems))

            # --- 結果一覧テーブル ---
            st.subheader("結果一覧")
            for problem, results in all_results:
                # 正解/不正解のサマリーをタイトルに含める
                has_correct = problem.correct_answer is not None
                with st.expander(f"[{problem.category}] {problem.id}: {problem.prompt.splitlines()[0][:60]}"):
                    if has_correct:
                        st.markdown(f"**正解: {problem.correct_answer}**")

                    cols = st.columns(len(results))
                    for col, (name, result) in zip(cols, results.items()):
                        with col:
                            st.markdown(f"**{name}**")

                            # 正解/不正解の表示
                            if has_correct:
                                is_correct = result.metadata.get("is_correct", False)
                                extracted = result.metadata.get("extracted_choice", "?")
                                if is_correct:
                                    st.success(f"正解 (選択: {extracted})")
                                else:
                                    st.error(f"不正解 (選択: {extracted})")

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

            # --- アルゴリズム別 正答率チャート ---
            algo_names = [a.name for a in algos]

            # 正答率・平均スコア・平均レイテンシの集計
            algo_correct: dict[str, int] = defaultdict(int)
            algo_total: dict[str, int] = defaultdict(int)
            algo_scores: dict[str, list[float]] = defaultdict(list)
            algo_latencies: dict[str, list[float]] = defaultdict(list)

            for problem, results in all_results:
                for algo_name in algo_names:
                    if algo_name not in results:
                        continue
                    r = results[algo_name]
                    if problem.correct_answer is not None:
                        algo_total[algo_name] += 1
                        if r.metadata.get("is_correct", False):
                            algo_correct[algo_name] += 1
                    algo_scores[algo_name].append(r.score)
                    algo_latencies[algo_name].append(r.latency_sec)

            st.subheader("アルゴリズム別 正答率")
            accuracy_values = []
            for name in algo_names:
                total = algo_total.get(name, 0)
                correct = algo_correct.get(name, 0)
                accuracy_values.append(correct / total if total > 0 else 0.0)

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Bar(
                name="正答率",
                x=algo_names,
                y=accuracy_values,
                text=[f"{v:.0%}" for v in accuracy_values],
                textposition="auto",
            ))
            fig_acc.update_layout(
                yaxis=dict(range=[0, 1], tickformat=".0%"),
                height=400,
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # 正答率の数値表示
            acc_cols = st.columns(len(algo_names))
            for col, name, acc in zip(acc_cols, algo_names, accuracy_values):
                with col:
                    total = algo_total.get(name, 0)
                    correct = algo_correct.get(name, 0)
                    st.metric(f"{name} 正答率", f"{acc:.0%} ({correct}/{total})")

            # --- カテゴリ別正答率 ---
            st.subheader("カテゴリ別 正答率")
            cat_correct: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
            cat_total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

            for problem, results in all_results:
                if problem.correct_answer is None:
                    continue
                for algo_name in algo_names:
                    if algo_name not in results:
                        continue
                    cat_total[problem.category][algo_name] += 1
                    if results[algo_name].metadata.get("is_correct", False):
                        cat_correct[problem.category][algo_name] += 1

            if cat_total:
                cat_names = sorted(cat_total.keys())
                fig_cat = go.Figure()
                for algo_name in algo_names:
                    cat_accs = []
                    for cat in cat_names:
                        t = cat_total[cat].get(algo_name, 0)
                        c = cat_correct[cat].get(algo_name, 0)
                        cat_accs.append(c / t if t > 0 else 0.0)
                    fig_cat.add_trace(go.Bar(
                        name=algo_name,
                        x=cat_names,
                        y=cat_accs,
                        text=[f"{v:.0%}" for v in cat_accs],
                        textposition="auto",
                    ))
                fig_cat.update_layout(
                    barmode="group",
                    yaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=400,
                )
                st.plotly_chart(fig_cat, use_container_width=True)

            # --- 平均スコア・レイテンシ ---
            st.subheader("アルゴリズム別 平均スコア・レイテンシ")
            avg_scores = {}
            avg_latencies = {}
            for name in algo_names:
                scores = algo_scores.get(name, [])
                latencies = algo_latencies.get(name, [])
                avg_scores[name] = sum(scores) / len(scores) if scores else 0
                avg_latencies[name] = sum(latencies) / len(latencies) if latencies else 0

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
