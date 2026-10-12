[CmdletBinding()]
param(
    [string]$PythonExecutable,
    [switch]$Uninstall
)
$ErrorActionPreference = 'Stop'
$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$servicesDirectory = Join-Path $workspaceRoot 'data\paper_services'
$supervisorScript = Join-Path $workspaceRoot 'scripts\run_paper_services.py'
if (-not $PythonExecutable) { $PythonExecutable = Join-Path $workspaceRoot '.venv\Scripts\python.exe' }
$PythonExecutable = (Resolve-Path -LiteralPath $PythonExecutable).Path
if (-not (Test-Path -LiteralPath $supervisorScript -PathType Leaf)) { throw 'Paper supervisor is unavailable.' }
$sha = [Security.Cryptography.SHA256]::Create()
try { $suffix = ([BitConverter]::ToString($sha.ComputeHash([Text.Encoding]::UTF8.GetBytes($workspaceRoot.ToLowerInvariant())))).Replace('-', '').Substring(0, 8) }
finally { $sha.Dispose() }
$taskName = "btc-auto-paper-services-$suffix"
$launcherPath = Join-Path $servicesDirectory 'start_paper_services.ps1'
$startupDirectory = [Environment]::GetFolderPath('Startup')
$startupPath = Join-Path $startupDirectory "$taskName.vbs"
$registrationPath = Join-Path $servicesDirectory 'startup_registration.json'

function Quote-PowerShellLiteral([string]$value) { return "'" + $value.Replace("'", "''") + "'" }
function Assert-OwnedTask($task) {
    if ($task -and @($task.Actions | Where-Object { $_.Arguments -like "*$launcherPath*" }).Count -ne 1) {
        throw 'A task with this name has a different action; refusing to replace it.'
    }
}

$existingTask = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
Assert-OwnedTask $existingTask
if ($Uninstall) {
    if ($existingTask) { Unregister-ScheduledTask -TaskName $taskName -Confirm:$false }
    if (Test-Path -LiteralPath $startupPath) {
        $existingStartup = Get-Content -LiteralPath $startupPath -Raw
        if (-not $existingStartup.Contains($launcherPath)) { throw 'Startup helper has a different target; refusing to remove it.' }
        Remove-Item -LiteralPath $startupPath
    }
    if (Test-Path -LiteralPath $registrationPath) { Remove-Item -LiteralPath $registrationPath }
    Write-Output 'Removed current-user startup registration. Existing running processes were not stopped.'
    exit 0
}

# The check is deliberately read-only. Registration cannot create new ledgers.
& $PythonExecutable '-u' $supervisorScript '--check'
if ($LASTEXITCODE -ne 0) { throw 'Paper guard validation failed; startup was not registered.' }
New-Item -ItemType Directory -Path $servicesDirectory -Force | Out-Null
$rootLiteral = Quote-PowerShellLiteral $workspaceRoot
$pythonLiteral = Quote-PowerShellLiteral $PythonExecutable
$scriptLiteral = Quote-PowerShellLiteral $supervisorScript
$directoryLiteral = Quote-PowerShellLiteral $servicesDirectory
$launcher = @"
`$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $rootLiteral
`$env:PYTHONUTF8 = '1'
`$env:PYTHONIOENCODING = 'utf-8'
[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding
`$attempt = 0
while (`$true) {
    `$stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmss')
    `$log = Join-Path $directoryLiteral ('supervisor_' + `$stamp + '_' + `$PID + '.log')
    & $pythonLiteral '-u' $scriptLiteral *>> `$log
    `$exitCode = `$LASTEXITCODE
    if (`$exitCode -in @(0, 73, 78)) { exit 0 }
    `$attempt++
    `$delay = [Math]::Min(300, 10 * [Math]::Pow(2, [Math]::Min(`$attempt - 1, 5)))
    Start-Sleep -Seconds `$delay
}
"@
Set-Content -LiteralPath $launcherPath -Value $launcher -Encoding UTF8
$currentIdentity = [Security.Principal.WindowsIdentity]::GetCurrent()
$currentUser = $currentIdentity.Name
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$arguments = '-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File "' + $launcherPath + '"'
$method = 'scheduled_task'
$taskFailure = $null
try {
    $action = New-ScheduledTaskAction -Execute $powershellPath -Argument $arguments -WorkingDirectory $workspaceRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
    $principal = New-ScheduledTaskPrincipal -UserId $currentUser -LogonType Interactive -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet -MultipleInstances IgnoreNew -StartWhenAvailable `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1)
    $task = New-ScheduledTask -Action $action -Trigger $trigger -Principal $principal -Settings $settings
    Register-ScheduledTask -TaskName $taskName -InputObject $task -Force | Out-Null
    if (Test-Path -LiteralPath $startupPath) {
        $existingStartup = Get-Content -LiteralPath $startupPath -Raw
        if (-not $existingStartup.Contains($launcherPath)) { throw 'Startup helper has a different target.' }
        Remove-Item -LiteralPath $startupPath
    }
}
catch {
    # Non-admin current-user task registration is restricted on some machines.
    # A hidden per-user Startup helper covers the same current-user login event.
    $method = 'startup_folder'
    $taskFailure = $_.Exception.Message
    if (Test-Path -LiteralPath $startupPath) {
        $existingStartup = Get-Content -LiteralPath $startupPath -Raw
        if (-not $existingStartup.Contains($launcherPath)) { throw 'Startup helper has a different target; refusing to replace it.' }
    }
    $hiddenCommand = '"' + $powershellPath + '" ' + $arguments
    $vbs = 'CreateObject("WScript.Shell").Run "' + $hiddenCommand.Replace('"', '""') + '", 0, False'
    Set-Content -LiteralPath $startupPath -Value $vbs -Encoding Unicode
}
$registration = [ordered]@{
    method = $method
    task_name = $taskName
    user = $currentUser
    launcher_path = $launcherPath
    startup_path = $(if ($method -eq 'startup_folder') { $startupPath } else { $null })
    registered_at_utc = [DateTime]::UtcNow.ToString('o')
    coverage = 'after_this_user_logs_in; not_before_login'
    places_orders = $false
    task_registration_error = $taskFailure
}
$registration | ConvertTo-Json | Set-Content -LiteralPath $registrationPath -Encoding UTF8
$registration | ConvertTo-Json
# Registration only: the caller starts the concrete launch helper after review.
# Start-Process powershell.exe -WindowStyle Hidden -ArgumentList $arguments -WorkingDirectory $workspaceRoot
