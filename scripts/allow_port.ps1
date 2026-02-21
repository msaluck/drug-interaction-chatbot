# Check if running as Administrator
if (-not ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole] "Administrator")) {
    Write-Warning "This script requires Administrator privileges to manage firewall rules."
    Write-Warning "Please right-click on PowerShell and select 'Run as Administrator', then run this script again."
    exit
}

$Port = 8000
$RuleName = "Allow FastAPI Port $Port"

# Check if rule exists
$existingRule = Get-NetFirewallRule -DisplayName $RuleName -ErrorAction SilentlyContinue

if ($existingRule) {
    Write-Host "Firewall rule '$RuleName' already exists." -ForegroundColor Green
    # Ensure it allows Inbound traffic
    Set-NetFirewallRule -DisplayName $RuleName -Action Allow -Direction Inbound -Enabled True
}
else {
    Write-Host "Creating firewall rule '$RuleName'..." -ForegroundColor Cyan
    New-NetFirewallRule -DisplayName $RuleName `
        -Direction Inbound `
        -LocalPort $Port `
        -Protocol TCP `
        -Action Allow `
        -Profile Any
    Write-Host "Rule created successfully." -ForegroundColor Green
}

Write-Host "Port $Port is now open." -ForegroundColor Green
Write-Host "You can access the server at http://$(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias Wi-Fi | Select-Object -ExpandProperty IPAddress):$Port"
