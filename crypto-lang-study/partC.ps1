# Part C, automatic: install Ollama, download the study models (with retries), write config.yaml,
# and check the prompts (static checks + machine back-translation).
#
# Run from the project folder in PowerShell:
#     powershell -ExecutionPolicy Bypass -File .\partC.ps1
# Options:
#     -SkipInstall            Ollama is already installed
#     -SkipPull               models are already downloaded
#     -MaxTries 50            retries per model download (default 30)
#     -BtModel <tag>          model used for back-translation (default llama3.1:8b-instruct-q4_K_M)
#     -NoBackTranslation      only the static prompt checks
# Safe to run again: finished downloads are skipped, broken downloads continue where they stopped.
# A log is written to partC_log.txt.

param(
    [switch]$SkipInstall,
    [switch]$SkipPull,
    [switch]$NoBackTranslation,
    [int]$MaxTries = 30,
    [string]$BtModel = "llama3.1:8b-instruct-q4_K_M",
    [string[]]$Models = @("qwen2.5-coder:7b-instruct-q4_K_M", "llama3.1:8b-instruct-q4_K_M", "deepseek-coder:6.7b-instruct-q4_K_M")
)

$ErrorActionPreference = "Continue"
Set-Location $PSScriptRoot
Start-Transcript -Path (Join-Path $PSScriptRoot "partC_log.txt") -Append | Out-Null

function Say($msg, $color = "Cyan") { Write-Host ""; Write-Host "=== $msg" -ForegroundColor $color }

# ---------------------------------------------------------------- Python
$py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Say "Python: $py"

# ---------------------------------------------------------------- 1. Ollama installed?
function Find-Ollama {
    $c = Get-Command ollama -ErrorAction SilentlyContinue
    if ($c) { return $c.Source }
    $p = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $p) { return $p }
    return $null
}

$ollama = Find-Ollama
if (-not $ollama -and -not $SkipInstall) {
    Say "Step 1/5: installing Ollama"
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        winget install --id Ollama.Ollama -e --accept-source-agreements --accept-package-agreements
    }
    $ollama = Find-Ollama
    if (-not $ollama) {
        Write-Host "winget did not work, downloading the installer (resumes if the connection drops)..."
        $setup = Join-Path $env:TEMP "OllamaSetup.exe"
        $ok = $false
        for ($i = 1; $i -le $MaxTries -and -not $ok; $i++) {
            try {
                Start-BitsTransfer -Source "https://ollama.com/download/OllamaSetup.exe" -Destination $setup -ErrorAction Stop
                $ok = $true
            } catch {
                Write-Host "  download attempt $i failed: $($_.Exception.Message). Retrying in 30 s..." -ForegroundColor Yellow
                Start-Sleep -Seconds 30
            }
        }
        if (-not $ok) { Say "Could not download Ollama. Install it by hand from https://ollama.com and run this script again with -SkipInstall" "Red"; Stop-Transcript | Out-Null; exit 1 }
        Write-Host "Starting the Ollama installer - click Install if it asks."
        Start-Process -FilePath $setup -Wait
        $ollama = Find-Ollama
    }
}
if (-not $ollama) { Say "Ollama not found. Install it from https://ollama.com, close and reopen PowerShell, then run this again." "Red"; Stop-Transcript | Out-Null; exit 1 }
Say "Step 1/5: Ollama found at $ollama" "Green"

# ---------------------------------------------------------------- 2. Ollama server running?
function Test-Server {
    try { Invoke-RestMethod -Uri "http://localhost:11434/api/version" -TimeoutSec 5 | Out-Null; return $true } catch { return $false }
}
if (-not (Test-Server)) {
    Say "Step 2/5: starting the Ollama server"
    Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 30 -and -not (Test-Server); $i++) { Start-Sleep -Seconds 2 }
}
if (-not (Test-Server)) { Say "The Ollama server does not answer. Open the Ollama app from the Start menu and run this again." "Red"; Stop-Transcript | Out-Null; exit 1 }
Say "Step 2/5: Ollama server is running" "Green"

# ---------------------------------------------------------------- 3. download models with retries
function Get-Installed {
    try { return (Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -TimeoutSec 10).models.name } catch { return @() }
}
$failed = @()
if (-not $SkipPull) {
    foreach ($m in $Models) {
        if ((Get-Installed) -contains $m) { Say "Step 3/5: $m already downloaded" "Green"; continue }
        Say "Step 3/5: downloading $m (about 4-5 GB; it resumes after a drop)"
        $done = $false
        for ($i = 1; $i -le $MaxTries -and -not $done; $i++) {
            & $ollama pull $m
            if ($LASTEXITCODE -eq 0) { $done = $true }
            else {
                Write-Host "  attempt $i of $MaxTries stopped. Waiting 30 s and continuing..." -ForegroundColor Yellow
                Start-Sleep -Seconds 30
                if (-not (Test-Server)) { Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden; Start-Sleep -Seconds 5 }
            }
        }
        if ($done) { Say "$m downloaded" "Green" } else { $failed += $m; Say "$m did NOT finish after $MaxTries tries - run this script again later" "Red" }
    }
}

# ---------------------------------------------------------------- 4. config.yaml
Say "Step 4/5: writing config.yaml from the installed models"
& $py src\setup_config.py
$cfgExit = $LASTEXITCODE

# ---------------------------------------------------------------- 5. prompt checks
Say "Step 5/5: checking the prompts"
$installed = Get-Installed
if (-not $NoBackTranslation -and ($installed -contains $BtModel)) {
    & $py src\check_prompts.py --backtranslate $BtModel
} else {
    if (-not $NoBackTranslation) { Write-Host "Back-translation model $BtModel is not installed yet - doing the static checks only." -ForegroundColor Yellow }
    & $py src\check_prompts.py
}
$chkExit = $LASTEXITCODE

# ---------------------------------------------------------------- summary
Say "SUMMARY" "Magenta"
if ($failed.Count -gt 0) { Write-Host "Models not finished: $($failed -join ', ')  -> run .\partC.ps1 again" -ForegroundColor Red }
elseif ($cfgExit -ne 0) { Write-Host "config.yaml: some study models are missing (see above)" -ForegroundColor Red }
else { Write-Host "All models downloaded and config.yaml is ready (mock model removed)." -ForegroundColor Green }
if ($chkExit -ne 0) { Write-Host "Prompt check found HARD problems - fix data\prompts.json and run: python src\check_prompts.py" -ForegroundColor Red }
else { Write-Host "Prompt check: no hard problems." -ForegroundColor Green }
Write-Host ""
Write-Host "Next:"
Write-Host "  1. Open results\prompt_check\prompt_check.csv in Excel, filter needs_human_look = TRUE,"
Write-Host "     and have a native speaker fill human_ok / human_comment for those rows."
Write-Host "  2. Pilot:     python src\generate.py --models qwen2.5-coder-7b --tasks T01 T10 --conditions baseline"
Write-Host "  3. Full run:  python src\generate.py"
Stop-Transcript | Out-Null
