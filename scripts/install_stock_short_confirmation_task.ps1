[CmdletBinding()]
param([switch]$Start)
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$pythonExecutable = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot '.venv\Scripts\python.exe')).Path
$pythonWindowless = (Resolve-Path -LiteralPath (Join-Path $workspaceRoot '.venv\Scripts\pythonw.exe')).Path
$runnerScript = Join-Path $workspaceRoot 'scripts\run_stock_short_confirmation_trial.py'
$configPath = Join-Path $workspaceRoot 'config\stock_short_confirmation_trial_20261030.json'
$experimentDirectory = Join-Path $workspaceRoot 'data\paper_trading\stock_short_confirmation_trial_20261030'
$launcherPath = Join-Path $experimentDirectory 'start_short_confirmation.ps1'
$registrationPath = Join-Path $experimentDirectory 'startup_registration.json'

# Validation must be read-only. Creating a new epoch is a separate explicit --once action.
& $pythonExecutable '-X' 'utf8' $runnerScript '--config' $configPath '--check'
if ($LASTEXITCODE -ne 0) { throw 'Read-only short_confirmation validation failed.' }
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
$taskName = "btc-auto-short-confirmation-$suffix"
$existing = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($existing -and (@($existing.Actions).Count -ne 1 -or
    ($existing.Actions[0].Arguments -notlike "*$launcherPath*" -and $existing.Actions[0].Arguments -notlike "*$runnerScript*"))) {
    throw 'The existing task belongs to a different launcher; refusing replacement.'
}
$wasDisabled = $existing -and -not $existing.Settings.Enabled
$currentUser = [Security.Principal.WindowsIdentity]::GetCurrent().Name
# A console-free Python action avoids an unnecessary interpreter wrapper.
# Scheduler minute recovery and IgnoreNew replace the wrapper's retry loop.
# No principal, ACL, execution policy, or system security setting is relaxed.
$arguments = '-B -X utf8 -u "' + $runnerScript + '" --config "' + $configPath + '" --require-existing'
$action = New-ScheduledTaskAction -Execute $pythonWindowless -Argument $arguments -WorkingDirectory $workspaceRoot
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
    action_kind = 'direct_windowless_python'
    executable = $pythonWindowless
    launcher_path = $runnerScript
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
