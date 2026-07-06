# LBM Examples Launcher
# Usage: .\run_lbm.ps1          -> show menu
#        .\run_lbm.ps1 3        -> run #3 directly
#        .\run_lbm.ps1 list     -> show menu

$Examples = @(
    @{ Name = "fluid_grid_lbm_cavity";              Desc = "顶盖驱动腔体流 (BGK单相, 密度场)" }
    @{ Name = "fluid_grid_lbm_container";            Desc = "刚体容器单向耦合 (BGK单相, 5面墙挡流体)" }
    @{ Name = "fluid_grid_lbm_cylinder";             Desc = "圆柱绕流 (BGK单相, 速度场+SDF障碍)" }
    @{ Name = "fluid_grid_lbm_dambreak";             Desc = "单相溃坝 (BGK, 被动标记场平流)" }
    @{ Name = "fluid_grid_lbm_dambreak_reg";         Desc = "正则化SC溃坝 (BGK+Reg, 强重力)" }
    @{ Name = "fluid_grid_lbm_dambreak_trt";         Desc = "TRT SC溃坝 (TRT+Reg, 高精度)" }
    @{ Name = "fluid_grid_lbm_dambreak_sc";          Desc = "纯SC溃坝 (BGK两相, 弱重力)" }
    @{ Name = "fluid_grid_lbm_dambreak_sc_rigid";    Desc = "SC溃坝+刚体球 (BGK, 掩膜密度场)" }
    @{ Name = "fluid_grid_lbm_debug";                Desc = "静态调试 (16^3, 不跑仿真)" }
    @{ Name = "fluid_grid_lbm_fsi";                  Desc = "运动球FSI (BGK单相, 速度场)" }
    @{ Name = "fluid_grid_lbm_oneway_moving_rigid_visual";     Desc = "单向振荡球BGK (SC两相, 密度/速度可切)" }
    @{ Name = "fluid_grid_lbm_oneway_moving_rigid_visual_trt"; Desc = "单向振荡球TRT (可切换碰撞模型)" }
    @{ Name = "fluid_grid_lbm_twophase";             Desc = "两相液滴平衡 (SC, 表面张力变圆)" }
    @{ Name = "fluid_grid_lbm_twophase_gravity";     Desc = "液滴重力下落 (SC, 撞击铺展)" }
    @{ Name = "fluid_grid_lbm_twoway_fsi_visual";    Desc = "双向FSI (示踪粒子+力箭头)" }
)

function Show-Menu {
    Write-Host ""
    Write-Host "=== LBM Examples Launcher ===" -ForegroundColor Cyan
    Write-Host ""
    for ($i = 0; $i -lt $Examples.Count; $i++) {
        $num = ($i + 1).ToString().PadLeft(2)
        Write-Host "  [$num] " -ForegroundColor Yellow -NoNewline
        Write-Host "$($Examples[$i].Name)" -ForegroundColor White -NoNewline
        Write-Host "  --  $($Examples[$i].Desc)" -ForegroundColor DarkGray
    }
    Write-Host ""
    Write-Host "  [ 0] Quit" -ForegroundColor DarkGray
    Write-Host ""
}

function Run-Example([int]$idx) {
    if ($idx -lt 1 -or $idx -gt $Examples.Count) {
        Write-Host "Invalid number: $idx" -ForegroundColor Red
        return
    }
    $name = $Examples[$idx - 1].Name
    Write-Host ""
    Write-Host ">> uv run --python 3.11 --extra examples python -m wanphys.examples.lbm.$name --viewer gl" -ForegroundColor Green
    Write-Host ""
    uv run --python 3.11 --extra examples python -m "wanphys.examples.lbm.$name" --viewer gl
}

# --- main ---
$target = $args[0]

if ($null -ne $target -and $target -ne "list") {
    $n = 0
    if ([int]::TryParse($target, [ref]$n)) {
        Run-Example $n
        return
    }
}

while ($true) {
    Show-Menu
    $input = Read-Host "Pick a number"
    $n = 0
    if ([int]::TryParse($input, [ref]$n)) {
        if ($n -eq 0) { break }
        Run-Example $n
        Write-Host ""
        Read-Host "Press Enter to continue..."
    }
}
