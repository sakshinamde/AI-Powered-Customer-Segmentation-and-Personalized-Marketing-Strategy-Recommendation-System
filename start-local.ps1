$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host 'Checking for stale Node processes on port 3000...'
$ports = @(3000)
foreach ($port in $ports) {
    $processIds = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique

    foreach ($procId in $processIds) {
        if ($null -ne $procId -and $procId -gt 0) {
            try {
                Stop-Process -Id $procId -Force -ErrorAction Stop
                Write-Host "Stopped stale process $procId on port $port"
            }
            catch {
                Write-Host "Could not stop PID $procId on port $port; it may already be cleaned up."
            }
        }
    }
}

$env:NODE_ENV = 'development'
Write-Host 'Starting SmartSeg AI on http://localhost:3000'
npm run dev
