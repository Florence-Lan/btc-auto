[CmdletBinding()]
param([switch]$Start)
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExecutable = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot '.venv\Scripts\python.exe')).Path
$runnerScript = Join-Path $workspaceRoot 'scripts\run_stock_early_recovery_shadow.py'
$configPath = Join-Path $workspaceRoot 'config\stock_early_recovery_shadow_20261028.json'
$experimentDirectory = Join-Path $workspaceRoot 'data\paper_trading\stock_early_recovery_shadow_20261028'
$launcherPath = Join-Path $experimentDirectory 'start_early_recovery.ps1'
$registrationPath = Join-Path $experimentDirectory 'startup_registration.json'

# Validation must be read-only. Creating a new epoch is a separate explicit --once action.
& $pythonExecutable '-X' 'utf8' $runnerScript '--config' $configPath '--check'
if ($LASTEXITCODE -ne 0) { throw 'Read-only early-recovery validation failed.' }
$manifestPath = Join-Path $experimentDirectory 'manifest.json'
if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
    throw 'Initialize this isolated experiment explicitly with --once before registering recovery.'
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($manifest.mode -ne 'SIMULATION' -or $manifest.places_orders -ne $false) {
    throw 'The existing manifest is not an isolated simulation experiment.'
}
$hashAlgorithm = [Security.Cryptography.SHA256]::Create()
try { $suffix = ([BitConverter]::ToString($hashAlgorithm.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspaceRoot.ToLowerInvariant())))).Replace('-', '').Substring(0,8) }
finally { $hashAlgorithm.Dispose() }
$taskName = "btc-auto-early-recovery-$suffix"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and @($existing.Actions | Where-Object { $_.Arguments -like "*$launcherPath*" }).Count -ne 1) {
    throw 'The existing task belongs to a different launcher; refusing replacement.'
}
$wasDisabled = $existing -and -not $existing.Settings.Enabled
function Literal([string]$value) { return "'" + $value.Replace("'", "''") + "'" }
$rootLiteral = Literal $workspaceRoot
$pythonLiteral = Literal $pythonExecutable
$scriptLiteral = Literal $runnerScript
$configLiteral = Literal $configPath
$directoryLiteral = Literal $experimentDirectory
$launcher = @"
`$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $rootLiteral
`$env:PYTHONUTF8 = '1'
`$env:PYTHONIOENCODING = 'utf-8'
`$attempt = 0
while (`$true) {
    `$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')
    `$log = Join-Path $directoryLiteral ('early_recovery_' + `$stamp + '_' + `$PID + '.log')
    & $pythonLiteral '-X' 'utf8' '-u' $scriptLiteral '--config' $configLiteral '--require-existing' *>> `$log
    `$exitCode = `$LASTEXITCODE
    if (`$exitCode -in @(0,73,78)) { exit 0 }
    `$attempt++
    `$delay = [Math]::Min(300,10 * [Math]::Pow(2,[Math]::Min(`$attempt - 1,5)))
    Start-Sleep -Seconds `$delay
}
"@
Set-Content -LiteralPath $launcherPath -Value $launcher -Encoding UTF8
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$powershellExecutable = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $launcherPath + '"'
$action = New-ScheduledTaskAction -Execute $powershellExecutable -Argument $arguments -WorkingDirectory $workspaceRoot
$logon = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$recovery = New-ScheduledTaskTrigger -Once -At ((Get-Date).AddMinutes(1)) -RepetitionInterval (New-TimeSpan -Minutes 1)
$recovery.Repetition.StopAtDurationEnd = $false
$principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) -Disable:$wasDisabled
$task = New-ScheduledTask -Action $action -Trigger @($logon,$recovery) -Principal $principal -Settings $settings
Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
[ordered]@{
    task_name = $taskName
    registered_at_utc = [DateTime]::UtcNow.ToString('o')
    launcher_path = $launcherPath
    config_path = $configPath
    enabled = -not $wasDisabled
    mode = 'SIMULATION'
    places_orders = $false
    require_existing = $true
    recovery_interval_seconds = 60
    coverage = 'after_current_user_logon; not_before_login'
    original_services_modified = $false
} | ConvertTo-Json | Set-Content -LiteralPath $registrationPath -Encoding UTF8
if ($Start -and -not $wasDisabled) { Start-ScheduledTask -TaskName $taskName }
Get-Content -LiteralPath $registrationPath -Raw
