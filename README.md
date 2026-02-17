# AI Test-Time Compute 評価システム

軽量ローカル LLM の「素の出力」と、テスト時計算（Test-Time Compute）アルゴリズムを比較評価するシステムです。
vLLM をバックエンドに使用し、Streamlit の Web UI から対話的にベンチマークを実行できます。

## 概要

推論時にどれだけ計算リソースを投入すれば回答品質が向上するかを、以下の 3 つのアルゴリズムで比較します。

| アルゴリズム | 説明 |
|---|---|
| **Baseline** | 温度サンプリングによる 1 回生成 |
| **Beam Search** | ステップ単位で複数候補を維持しながら最良経路を選択 |
| **MCTS** | Monte Carlo Tree Search で部分回答を木構造探索 |

## 前提条件

- Python 3.11 以上
- Docker Desktop（vLLM サーバー用）
- NVIDIA GPU + ドライバ（CUDA 対応）

## セットアップ

### ワンクリック（Windows）

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

このスクリプトは以下を自動で行います:

1. 前提条件（Docker / GPU / Python）の確認
2. `.env` ファイルの作成
3. Python 仮想環境の構築と依存パッケージのインストール
4. vLLM サーバーの起動（Docker Compose）
5. Streamlit UI の起動

### 手動セットアップ

```bash
# 1. .env を作成
cp .env.example .env
# 必要に応じて HF_TOKEN, VLLM_MODEL を編集

# 2. Python 仮想環境
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# 3. vLLM サーバーを起動
docker compose up -d

# 4. Streamlit を起動
streamlit run app.py
```

## 使い方

ブラウザで Streamlit UI（デフォルト: http://localhost:8501）を開き、2 つのモードから選択できます。

### 自由入力モード

任意のプロンプトを入力し、各アルゴリズムの回答・スコア・レイテンシ・トークン数を比較できます。

### ベンチマークモード

組み込みの多肢選択式ベンチマーク（20 問 / 4 カテゴリ）を実行し、アルゴリズム別・カテゴリ別の正答率やスコアを可視化します。

**ベンチマークカテゴリ:**

- 論理推論（5 問）
- 数学（5 問）
- 知識（5 問）
- 語彙（5 問）

## プロジェクト構成

```
ai-test-time/
├── app.py                      # Streamlit UI
├── setup.ps1                   # ワンクリックセットアップスクリプト
├── docker-compose.yml          # vLLM サーバー定義
├── pyproject.toml              # プロジェクト設定・依存関係
├── .env.example                # 環境変数テンプレート
├── src/
│   ├── client.py               # vLLM OpenAI 互換 API クライアント
│   ├── scoring.py              # スコアリング関数群
│   ├── eval_runner.py          # 評価実行エンジン
│   ├── algorithms/
│   │   ├── base.py             # アルゴリズム共通インターフェース
│   │   ├── baseline.py         # Baseline（1 回生成）
│   │   ├── beam_search.py      # Beam Search
│   │   └── mcts.py             # Monte Carlo Tree Search
│   └── benchmarks/
│       └── datasets.py         # ベンチマーク問題セット
└── tests/
    └── test_algorithms.py      # アルゴリズムのテスト
```

## スコアリング

回答の品質は複数の指標を組み合わせて評価されます。

**自由記述式:**

| 指標 | 重み | 説明 |
|---|---|---|
| Self-Evaluation | 50% | モデル自身による 1-10 点評価 |
| Logprob | 30% | トークン log 確率の平均（モデルの自信度） |
| Length Penalty | 20% | 目標長に対するガウシアンペナルティ |

**多肢選択式:**

| 指標 | 重み | 説明 |
|---|---|---|
| Accuracy | 70% | 正解との一致判定 |
| Logprob | 20% | トークン log 確率の平均 |
| Self-Evaluation | 10% | モデル自身による評価 |

## 設定

サイドバーから以下のパラメータを調整できます。

- **vLLM サーバー URL / モデル名**
- **Beam Search:** beam_width, max_steps, step_tokens
- **MCTS:** num_iterations, expansion_width, 探索係数 C, max_depth, step_tokens
- **共通:** temperature, max_tokens

## テスト

```bash
pytest
```

## ライセンス

<!-- ライセンスをここに記載してください -->
