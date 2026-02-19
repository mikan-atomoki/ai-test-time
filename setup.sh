#!/usr/bin/env bash
# AI Test-Time Compute 評価システム — ワンクリックセットアップ
# Linux / macOS + NVIDIA GPU 環境で vLLM サーバー (Docker) と Streamlit UI を起動します。
#
# 使い方:
#   chmod +x setup.sh && ./setup.sh

set -euo pipefail

# ─── ヘルパー ─────────────────────────────────────────────────────
step()  { printf '\n\033[36m==> %s\033[0m\n' "$1"; }
ok()    { printf '    \033[32m[OK] %s\033[0m\n' "$1"; }
fail()  { printf '    \033[31m[NG] %s\033[0m\n' "$1"; }
warn()  { printf '    \033[33m%s\033[0m\n' "$1"; }
gray()  { printf '    \033[90m%s\033[0m\n' "$1"; }

# ─── 1. 前提チェック ─────────────────────────────────────────────
step "前提条件をチェックしています..."

# Docker
if ! command -v docker &>/dev/null; then
    fail "docker コマンドが見つかりません。Docker をインストールしてください。"
    warn "https://docs.docker.com/engine/install/"
    exit 1
fi
ok "docker が見つかりました"

# Docker デーモン
if ! docker info &>/dev/null; then
    fail "Docker デーモンに接続できません。Docker を起動してください。"
    exit 1
fi
ok "Docker デーモンが起動しています"

# NVIDIA GPU
if ! command -v nvidia-smi &>/dev/null; then
    fail "nvidia-smi が見つかりません。NVIDIA GPU ドライバがインストールされているか確認してください。"
    exit 1
fi
if ! nvidia-smi &>/dev/null; then
    fail "nvidia-smi の実行に失敗しました。GPU ドライバを確認してください。"
    exit 1
fi
ok "NVIDIA GPU を検出しました"

# Python
if ! command -v python3 &>/dev/null; then
    fail "python3 コマンドが見つかりません。Python 3.11 以上をインストールしてください。"
    warn "https://www.python.org/downloads/"
    exit 1
fi
py_version=$(python3 --version 2>&1)
ok "Python が見つかりました ($py_version)"

# ─── 2. .env 作成 ────────────────────────────────────────────────
step ".env ファイルを確認しています..."

if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        ok ".env.example から .env を作成しました"
        warn "必要に応じて .env を編集してください（HF_TOKEN, VLLM_MODEL）"
    else
        fail ".env.example が見つかりません。リポジトリが正しいか確認してください。"
        exit 1
    fi
else
    ok ".env は既に存在します"
fi

# ─── 3. Python 仮想環境 ─────────────────────────────────────────
step "Python 仮想環境をセットアップしています..."

if [ ! -d .venv ]; then
    python3 -m venv .venv
    ok ".venv を作成しました"
else
    ok ".venv は既に存在します"
fi

# shellcheck disable=SC1091
source .venv/bin/activate
ok "仮想環境を有効化しました"

warn "pip install -e '.[dev]' を実行しています..."
pip install -e ".[dev]"
ok "依存パッケージをインストールしました"

# ─── 4. vLLM サーバー起動 ────────────────────────────────────────
step "vLLM サーバーを起動しています..."

docker compose up -d
ok "docker compose up -d を実行しました"

# ヘルスチェック: /v1/models をポーリング
warn "vLLM サーバーの起動を待機しています..."
max_retries=60
retry_interval=5
ready=false

for i in $(seq 1 "$max_retries"); do
    if curl -sf --max-time 3 "http://localhost:8000/v1/models" >/dev/null 2>&1; then
        ready=true
        break
    fi
    gray "[$i/$max_retries] 起動待ち... (${retry_interval}秒後にリトライ)"
    sleep "$retry_interval"
done

if $ready; then
    ok "vLLM サーバーが起動しました (http://localhost:8000)"
else
    fail "vLLM サーバーの起動がタイムアウトしました。"
    warn "'docker compose logs vllm' でログを確認してください。"
    warn "GPU メモリ不足の場合はモデルサイズの小さいものに変更してください。"
    exit 1
fi

# ─── 5. Streamlit 起動 ──────────────────────────────────────────
step "Streamlit を起動しています..."
warn "ブラウザで UI が自動的に開きます。"
warn "終了するには Ctrl+C を押してください。"
echo ""

streamlit run app.py
