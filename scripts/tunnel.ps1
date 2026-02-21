# Script to start ngrok tunnel
# Requirement: ngrok must be installed and authenticated

$Port = 8000

# 1. Check if ngrok is installed
if (-not (Get-Command "ngrok" -ErrorAction SilentlyContinue)) {
    Write-Error "ngrok is not installed or not in your PATH."
    Write-Host "Please install it via: winget install ngrok" -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting ngrok on port $Port..." -ForegroundColor Cyan
Write-Host "IMPORTANT: If using a FREE ngrok account, your API calls might fail with a HTML warning page." -ForegroundColor Yellow
Write-Host "To fix this, add this header to your Android App's HTTP client:" -ForegroundColor Cyan
Write-Host "Key: ngrok-skip-browser-warning" -ForegroundColor Green
Write-Host "Value: true" -ForegroundColor Green

# 2. Run ngrok
# We use Start-Process to run it in a new window so the UI renders correctly, 
# or just run it here if preferred. Let's run it directly.
ngrok http $Port
