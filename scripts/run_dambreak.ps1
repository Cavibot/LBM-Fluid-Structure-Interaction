# SPDX-FileCopyrightText: Copyright (c) 2025 WanPhys Developers
# SPDX-License-Identifier: Apache-2.0
#
# Dam-break three-axis launcher (Encoding / Collision / Force)
#
# Usage:
#   .\scripts\run_dambreak.ps1              # interactive menu
#   .\scripts\run_dambreak.ps1 help         # print flag cheat-sheet
#   .\scripts\run_dambreak.ps1 1            # preset #1 (FullF + SRT + g+SC, N=64)
#   .\scripts\run_dambreak.ps1 -e f -c t -f gsc -n 64
#   .\scripts\run_dambreak.ps1 f t gsc 64   # positional short form: enc col force [N]
#
# Flags (short):
#   -e  encoding   f|fullf   h|home
#   -c  collision  s|srt     t|trt     r|raw|raw_mrt     n|nocm|nocm_mrt
#   -f  force      0|none    g|gravity sc|shan_chen      gsc|gs|gravity+shan_chen
#   -n  resolution N^3 (default 128)
#   --viewer gl|null   --device cuda:0|cpu   --headless

$ErrorActionPreference = "Stop"
$Module = "wanphys.examples.lbm.fluid_grid_lbm_dambreak_trt"
$DefaultViewer = "gl"
$DefaultN = 128

# All presets: force = g+SC; matrix = {f,h} x {s,r,n,t} x {64,128}
# Note: HOME + RawMRT is fail-fast (still listed for completeness).
$Presets = @()
$id = 1
foreach ($e in @("f", "h")) {
    foreach ($c in @("s", "r", "n", "t")) {
        foreach ($n in @(64, 128)) {
            $eName = if ($e -eq "f") { "FullF" } else { "HOME" }
            $cName = switch ($c) {
                "s" { "SRT" }
                "r" { "RawMRT" }
                "n" { "NOCM" }
                "t" { "TRT" }
            }
            $note = if ($e -eq "h" -and $c -eq "r") { "  [illegal]" } else { "" }
            $Presets += @{
                Id   = $id
                E    = $e
                C    = $c
                F    = "gsc"
                N    = $n
                Desc = "$eName + $cName + g+SC  N=$n$note"
            }
            $id++
        }
    }
}

function Show-CheatSheet {
    Write-Host ""
    Write-Host "=== Dam-Break 3-axis flags ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  -e / --enc     " -NoNewline -ForegroundColor Yellow
    Write-Host "f|fullf   h|home"
    Write-Host "  -c / --col     " -NoNewline -ForegroundColor Yellow
    Write-Host "s|srt   t|trt   r|raw|raw_mrt   n|nocm|nocm_mrt"
    Write-Host "  -f / --force   " -NoNewline -ForegroundColor Yellow
    Write-Host "0|none   g|gravity   sc|shan_chen   gsc|gs"
    Write-Host "  -n / --res     " -NoNewline -ForegroundColor Yellow
    Write-Host "grid N^3  (default $DefaultN)"
    Write-Host ""
    Write-Host "  illegal:  -e h -c r   (HOME + RawMRT fail-fast)" -ForegroundColor DarkRed
    Write-Host "  research: SC on non-(FullF SRT/TRT) paths" -ForegroundColor DarkYellow
    Write-Host ""
    Write-Host "Examples:" -ForegroundColor Cyan
    Write-Host "  .\scripts\run_dambreak.ps1 1"
    Write-Host "  .\scripts\run_dambreak.ps1 -e f -c t -f gsc -n 64"
    Write-Host "  .\scripts\run_dambreak.ps1 f t gsc 64"
    Write-Host "  .\scripts\run_dambreak.ps1 -e h -c n -f g --viewer null -n 32"
    Write-Host ""
}

function Show-Menu {
    Write-Host ""
    Write-Host "=== Dam-Break Launcher (enc / col / force=g+SC) ===" -ForegroundColor Cyan
    Write-Host ""
    Write-Host ("  {0,-4} {1,-3} {2,-3} {3,-4} {4,-5}  {5}" -f `
        "#", "E", "C", "F", "N", "description") -ForegroundColor DarkGray
    Write-Host ("  {0}" -f ("-" * 64)) -ForegroundColor DarkGray
    foreach ($p in $Presets) {
        $color = if ($p.E -eq "h" -and $p.C -eq "r") { "DarkRed" } else { "White" }
        Write-Host ("  [{0,2}] " -f $p.Id) -NoNewline -ForegroundColor Yellow
        Write-Host ("{0,-3} {1,-3} {2,-4} {3,-5}  {4}" -f `
            $p.E, $p.C, $p.F, $p.N, $p.Desc) -ForegroundColor $color
    }
    Write-Host ""
    Write-Host "  [ 0] Quit" -ForegroundColor DarkGray
    Write-Host "  [ h] Help / flag cheat-sheet" -ForegroundColor DarkGray
    Write-Host "  [ c] Custom  (prompt -e -c -f -n)" -ForegroundColor DarkGray
    Write-Host ""
}

function Invoke-DamBreak {
    param(
        [Parameter(Mandatory = $true)][string]$Enc,
        [Parameter(Mandatory = $true)][string]$Col,
        [Parameter(Mandatory = $true)][string]$Force,
        [int]$N = $DefaultN,
        [string]$Viewer = $DefaultViewer,
        [string]$Device = "",
        [switch]$Headless,
        [string[]]$ExtraArgs = @()
    )

    if ($Enc -match '^(h|home)$' -and $Col -match '^(r|raw|raw_mrt)$') {
        Write-Host "ERROR: HOME + RawMRT is fail-fast (unsupported)." -ForegroundColor Red
        return 2
    }

    $cmdArgs = @(
        "run", "--extra", "examples",
        "python", "-m", $Module,
        "--viewer", $Viewer,
        "-e", $Enc,
        "-c", $Col,
        "-f", $Force,
        "-n", "$N"
    )
    if ($Device -ne "") {
        $cmdArgs += @("--device", $Device)
    }
    if ($Headless) {
        $cmdArgs += "--headless"
    }
    if ($ExtraArgs.Count -gt 0) {
        $cmdArgs += $ExtraArgs
    }

    $pretty = "uv " + ($cmdArgs -join " ")
    Write-Host ""
    Write-Host ">> $pretty" -ForegroundColor Green
    Write-Host ""
    & uv @cmdArgs
    return $LASTEXITCODE
}

function Read-CustomAndRun {
    $enc = Read-Host "encoding  [-e] f|h  (default f)"
    if ([string]::IsNullOrWhiteSpace($enc)) { $enc = "f" }
    $col = Read-Host "collision [-c] s|t|r|n  (default t)"
    if ([string]::IsNullOrWhiteSpace($col)) { $col = "t" }
    $frc = Read-Host "force     [-f] 0|g|sc|gsc  (default gsc)"
    if ([string]::IsNullOrWhiteSpace($frc)) { $frc = "gsc" }
    $nIn = Read-Host "resolution [-n]  (default $DefaultN)"
    $n = $DefaultN
    if (-not [string]::IsNullOrWhiteSpace($nIn)) { $n = [int]$nIn }
    Invoke-DamBreak -Enc $enc -Col $col -Force $frc -N $n | Out-Null
}

function Parse-And-Run {
    param([string[]]$ArgList)

    if ($ArgList.Count -eq 0) { return $false }

    $first = $ArgList[0].ToLowerInvariant()
    if ($first -in @("help", "-h", "--help", "?")) {
        Show-CheatSheet
        return $true
    }
    if ($first -eq "list") {
        Show-Menu
        return $true
    }

    # preset number
    $presetId = 0
    if ([int]::TryParse($ArgList[0], [ref]$presetId)) {
        $p = $Presets | Where-Object { $_.Id -eq $presetId } | Select-Object -First 1
        if ($null -eq $p) {
            Write-Host "Unknown preset: $presetId" -ForegroundColor Red
            return $true
        }
        Invoke-DamBreak -Enc $p.E -Col $p.C -Force $p.F -N $p.N | Out-Null
        return $true
    }

    # positional: e c f [n]
    $isFlag = $ArgList[0].StartsWith("-")
    if (-not $isFlag -and $ArgList.Count -ge 3) {
        $enc = $ArgList[0]
        $col = $ArgList[1]
        $frc = $ArgList[2]
        $n = $DefaultN
        $viewer = $DefaultViewer
        $device = ""
        $headless = $false
        $i = 3
        if ($ArgList.Count -gt 3 -and $ArgList[3] -match '^\d+$') {
            $n = [int]$ArgList[3]
            $i = 4
        }
        while ($i -lt $ArgList.Count) {
            switch -Regex ($ArgList[$i]) {
                '^--viewer$' { $viewer = $ArgList[$i + 1]; $i += 2; continue }
                '^--device$'  { $device = $ArgList[$i + 1]; $i += 2; continue }
                '^--headless$' { $headless = $true; $i += 1; continue }
                default { $i += 1 }
            }
        }
        Invoke-DamBreak -Enc $enc -Col $col -Force $frc -N $n -Viewer $viewer -Device $device -Headless:$headless | Out-Null
        return $true
    }

    # flag form: -e f -c t -f gsc -n 64 ...
    $enc = "f"
    $col = "t"
    $frc = "gsc"
    $n = $DefaultN
    $viewer = $DefaultViewer
    $device = ""
    $headless = $false
    $extra = New-Object System.Collections.Generic.List[string]
    $i = 0
    while ($i -lt $ArgList.Count) {
        $a = $ArgList[$i]
        switch -Regex ($a) {
            '^-e$|^--enc$' {
                $enc = $ArgList[$i + 1]; $i += 2; continue
            }
            '^-c$|^--col$' {
                $col = $ArgList[$i + 1]; $i += 2; continue
            }
            '^-f$|^--force$' {
                $frc = $ArgList[$i + 1]; $i += 2; continue
            }
            '^-n$|^--res$' {
                $n = [int]$ArgList[$i + 1]; $i += 2; continue
            }
            '^--viewer$' {
                $viewer = $ArgList[$i + 1]; $i += 2; continue
            }
            '^--device$' {
                $device = $ArgList[$i + 1]; $i += 2; continue
            }
            '^--headless$' {
                $headless = $true; $i += 1; continue
            }
            default {
                $extra.Add($a) | Out-Null
                $i += 1
            }
        }
    }
    Invoke-DamBreak -Enc $enc -Col $col -Force $frc -N $n -Viewer $viewer -Device $device -Headless:$headless -ExtraArgs $extra.ToArray() | Out-Null
    return $true
}

# --- main ---
$raw = @($args)
if (Parse-And-Run -ArgList $raw) {
    return
}

while ($true) {
    Show-Menu
    $choice = Read-Host "Pick preset # / h / c / 0"
    if ([string]::IsNullOrWhiteSpace($choice)) { continue }
    $c = $choice.Trim().ToLowerInvariant()
    if ($c -eq "0" -or $c -eq "q" -or $c -eq "quit") { break }
    if ($c -eq "h" -or $c -eq "help") { Show-CheatSheet; continue }
    if ($c -eq "c" -or $c -eq "custom") { Read-CustomAndRun; Read-Host "Press Enter..."; continue }

    $id = 0
    if ([int]::TryParse($c, [ref]$id)) {
        $p = $Presets | Where-Object { $_.Id -eq $id } | Select-Object -First 1
        if ($null -eq $p) {
            Write-Host "Unknown preset: $id" -ForegroundColor Red
            continue
        }
        Invoke-DamBreak -Enc $p.E -Col $p.C -Force $p.F -N $p.N | Out-Null
        Write-Host ""
        Read-Host "Press Enter to continue..."
        continue
    }
    Write-Host "Unrecognized input: $choice" -ForegroundColor Red
}
