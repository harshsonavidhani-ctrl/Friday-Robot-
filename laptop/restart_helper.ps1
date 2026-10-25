# Friday's laptop helper: start the CURRENT copy, in Harsh's own desktop
# session (2026-10-25: "make the laptop helper work on the first try").
#
# Harsh runs this ONCE himself, in his own (non-admin) PowerShell:
#     powershell -NoProfile -ExecutionPolicy Bypass -File "$env:USERPROFILE\Friday\restart_helper.ps1"
# It (1) points the Startup-folder entry at Friday\friday_laptop_agent.py run
# by FridayML's pythonw (the old entry started an old copy from his home
# folder, so every reboot brought back helper 9); (2) makes an on-demand
# scheduled task "Friday laptop helper" (no trigger, runs only while he is
# logged on, in his session) so the Pi can restart the helper after an
# update; (3) stops every running copy and starts the current one.
#
# The Pi runs it with -RestartOnly (scripts/laptop_update.py): then it
# creates and changes NOTHING, only stops the old copy and runs the task he
# made. Undo: Unregister-ScheduledTask "Friday laptop helper"; the old
# Startup line is kept in Friday\friday_laptop_helper.cmd.old.
param([switch]$RestartOnly)
$ErrorActionPreference = 'Stop'
$task = 'Friday laptop helper'
$pythonw = Join-Path $env:USERPROFILE 'FridayML\py\pythonw.exe'
$script = Join-Path $env:USERPROFILE 'Friday\friday_laptop_agent.py'
if (-not (Test-Path $pythonw)) { $pythonw = 'pythonw' }      # no FridayML: the Store Python

if (-not $RestartOnly) {
    $startup = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs\Startup\friday_laptop_helper.cmd'
    $line = "@start `"`" `"$pythonw`" `"$script`""
    $old = if (Test-Path $startup) { (Get-Content $startup -Raw).Trim() } else { '' }
    if ($old -ne $line) {
        if ($old) { Set-Content (Join-Path $env:USERPROFILE 'Friday\friday_laptop_helper.cmd.old') $old }
        Set-Content -Path $startup -Value $line -Encoding ASCII
        "startup entry now: $line"
    }
    if (-not (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue)) {
        $action = New-ScheduledTaskAction -Execute $pythonw -Argument "`"$script`"" -WorkingDirectory (Split-Path $script)
        $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
        $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero)
        Register-ScheduledTask -TaskName $task -Action $action -Principal $principal -Settings $settings | Out-Null
        "scheduled task made: $task (no trigger: it runs only when started)"
    }
} elseif (-not (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue)) {
    'NO TASK: Harsh has not run restart_helper.ps1 himself yet'
    exit 2
}

# one port, one helper: stop every running copy, from any folder, any Python
Get-CimInstance Win32_Process |
    Where-Object { $_.CommandLine -like '*friday_laptop_agent*' -and $_.ProcessId -ne $PID } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.Name) $($_.ProcessId)" }
Start-Sleep -Milliseconds 800
Start-ScheduledTask -TaskName $task
Start-Sleep -Seconds 3
$running = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*Friday\friday_laptop_agent*' }
if ($running) { "running: $($running.Name) pid $($running.ProcessId) session $($running.SessionId)" }
else { 'NOT RUNNING'; exit 1 }
