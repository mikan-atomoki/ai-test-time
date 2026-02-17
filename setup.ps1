<#
.SYNOPSIS
    AI Test-Time Compute 評価システム — ワンクリックセットアップ
.DESCRIPTION
    Windows + NVIDIA GPU 環境で vLLM サーバー (Docker) と Streamlit UI を起動します。
.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "    [OK] $msg" -ForegroundColor Green }
function Write-Fail($msg) { Write-Host "    [NG] $msg" -ForegroundColor Red }

# ─── 1. 前提チェック ─────────────────────────────────────────────
Write-Step "前提条件をチェックしています..."

# Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Fail "docker コマンドが見つかりません。Docker Desktop をインストールしてください。"
    Write-Host "    https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}
Write-Ok "docker が見つかりました"

# Docker デーモンが起動しているか
try {
    docker info *>$null
    Write-Ok "Docker デーモンが起動しています"
} catch {
    Write-Fail "Docker デーモンに接続できません。Docker Desktop を起動してください。"
    exit 1
}

# NVIDIA GPU
if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    Write-Fail "nvidia-smi が見つかりません。NVIDIA GPU ドライバがインストールされているか確認してください。"
    exit 1
}
try {
    nvidia-smi *>$null
    Write-Ok "NVIDIA GPU を検出しました"
} catch {
    Write-Fail "nvidia-smi の実行に失敗しました。GPU ドライバを確認してください。"
    exit 1
}

# Python
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Fail "python コマンドが見つかりません。Python 3.11 以上をインストールしてください。"
    Write-Host "    https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}
$pyVersion = python --version 2>&1
Write-Ok "Python が見つかりました ($pyVersion)"

# ─── 2. .env 作成 ────────────────────────────────────────────────
Write-Step ".env ファイルを確認しています..."

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Copy-Item ".env.example" ".env"
        Write-Ok ".env.example から .env を作成しました"
        Write-Host "    必要に応じて .env を編集してください（HF_TOKEN, VLLM_MODEL）" -ForegroundColor Yellow
    } else {
        Write-Fail ".env.example が見つかりません。リポジトリが正しいか確認してください。"
        exit 1
    }
} else {
    Write-Ok ".env は既に存在します"
}

# ─── 3. Python 仮想環境 ─────────────────────────────────────────
Write-Step "Python 仮想環境をセットアップしています..."

if (-not (Test-Path ".venv")) {
    python -m venv .venv
    Write-Ok ".venv を作成しました"
} else {
    Write-Ok ".venv は既に存在します"
}

# venv を有効化して依存パッケージをインストール
& .venv\Scripts\Activate.ps1
Write-Ok "仮想環境を有効化しました"

Write-Host "    pip install -e '.[dev]' を実行しています..." -ForegroundColor Yellow
pip install -e ".[dev]"
Write-Ok "依存パッケージをインストールしました"

# ─── 4. vLLM サーバー起動 ────────────────────────────────────────
Write-Step "vLLM サーバーを起動しています..."

docker compose up -d
Write-Ok "docker compose up -d を実行しました"

# ヘルスチェック: /v1/models をポーリング
Write-Host "    vLLM サーバーの起動を待機しています..." -ForegroundColor Yellow
$maxRetries = 60
$retryInterval = 5
$ready = $false

for ($i = 1; $i -le $maxRetries; $i++) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/v1/models" -Method Get -TimeoutSec 3
        if ($response) {
            $ready = $true
            break
        }
    } catch {
        # まだ起動中
    }
    Write-Host "    [$i/$maxRetries] 起動待ち... (${retryInterval}秒後にリトライ)" -ForegroundColor Gray
    Start-Sleep -Seconds $retryInterval
}

if ($ready) {
    Write-Ok "vLLM サーバーが起動しました (http://localhost:8000)"
} else {
    Write-Fail "vLLM サーバーの起動がタイムアウトしました。"
    Write-Host "    'docker compose logs vllm' でログを確認してください。" -ForegroundColor Yellow
    Write-Host "    GPU メモリ不足の場合はモデルサイズの小さいものに変更してください。" -ForegroundColor Yellow
    exit 1
}

# ─── 5. Streamlit 起動 ──────────────────────────────────────────
Write-Step "Streamlit を起動しています..."
Write-Host "    ブラウザで UI が自動的に開きます。" -ForegroundColor Yellow
Write-Host "    終了するには Ctrl+C を押してください。" -ForegroundColor Yellow
Write-Host ""

streamlit run app.py
