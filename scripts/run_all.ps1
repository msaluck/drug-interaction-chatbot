# Master script to launch Server + Tunnel
$ErrorActionPreference = "Stop"

# 1. Start the FastAPI Server in a separate window
Write-Host "Launching API Server..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-File", "$PSScriptRoot\start.ps1"

# 2. Wait a few seconds for the server to initialize
Write-Host "Waiting 5 seconds for server to warm up..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# 3. Start ngrok Tunnel in the current window (or a new one)
Write-Host "Launching ngrok Tunnel..." -ForegroundColor Cyan
# We use start.ps1's logic context, but we can just call tunnel.ps1 directly
& "$PSScriptRoot\tunnel.ps1"
