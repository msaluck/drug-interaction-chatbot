param(
  [string]$EnvName = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "[setup] Working directory: $PSScriptRoot\.." -ForegroundColor Cyan
Set-Location "$PSScriptRoot\.."

# 1) Ensure venv exists
if (-not (Test-Path $EnvName)) {
  Write-Host "[setup] Creating venv at $EnvName" -ForegroundColor Cyan
  python -m venv $EnvName
}

# 2) Activate venv
$activatePath = Join-Path $EnvName "Scripts/Activate.ps1"
if (-not (Test-Path $activatePath)) {
  throw "[setup] Activation script not found at $activatePath"
}
. $activatePath

# 3) Ensure pip up to date and install deps
Write-Host "[setup] Upgrading pip and installing requirements" -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) Verify model path from .env
if (-not (Test-Path ".env")) {
  Write-Host "[warn] .env not found. Create it before running the server." -ForegroundColor Yellow
} else {
  $modelLine = (Get-Content .env) | Where-Object { $_ -match '^LLAMA_CPP_MODEL=' } | Select-Object -First 1
  if ($null -eq $modelLine) {
    Write-Host "[warn] LLAMA_CPP_MODEL not set in .env" -ForegroundColor Yellow
  } else {
    $modelPath = $modelLine.Split('=')[1]
    if (-not (Test-Path $modelPath)) {
      Write-Host "[warn] Model file not found at '$modelPath' (from .env)." -ForegroundColor Yellow
    } else {
      Write-Host "[ok] Model found at '$modelPath'" -ForegroundColor Green
    }
  }
}

# 5) Run server
Write-Host "[run] Starting FastAPI with Uvicorn on http://0.0.0.0:8000" -ForegroundColor Green
uvicorn src.main:app --reload --host 0.0.0.0
