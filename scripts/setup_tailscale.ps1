# Tailscale setup and verification script for B.A.Y.M.A.X.
$TAILSCALE_IP = "100.89.251.123"

Write-Host "Checking Tailscale Status..." -ForegroundColor Cyan
try {
    $status = tailscale status
    Write-Host $status
} catch {
    Write-Host "Tailscale command line not found or inactive. Ensure Tailscale is installed and running." -ForegroundColor Yellow
}

Write-Host "Starting Baymax server bound to 0.0.0.0 (Reachable at http://$($TAILSCALE_IP):8000)..." -ForegroundColor Green
.\venv\Scripts\uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
