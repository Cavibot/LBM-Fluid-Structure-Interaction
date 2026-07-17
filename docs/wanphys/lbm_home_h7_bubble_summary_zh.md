# H7 成果总结：气泡压力、GPU CCL 与 Home-FSLBM 对齐

> 文档版本：2026-07-17  
> 适用范围：`wanphys` `lbm_backend='home_fp32'`（`home_fp32_ref`）  
> 对照实现：`Home-FSLBM`（`mrLbmSolverGpu3D.cu` / `tDCCL`）  
> 论文锚点：锐界面自由表面流与泡沫 §4.2（封闭气泡 p–V）；§4.4（溶解气 / 泡沫）仍延期

---

## 1. 一句话结论

在 wanphys 的 **HOME-FREE GPU VOF** 上落地了论文 **§4.2 最小可用版**：对 G∪I 做连通标记、理想气体 \(\rho=V_0/V\)、大气腔强制 \(\rho=1\)，并接入 FS。  
日常体积更新与 Home 对齐为 **Δφ 原子累加**；拓扑变化用 **Warp GPU CCL**（扮演 Home `tDCCL` 角色）。  
**Dam-break 验证了大气路径与性能可用性**；封闭泡压缩（\(\rho\neq1\)）需倒液 / 人工气腔算例，不是静置水池的默认现象。

---

## 2. 本阶段做了什么

### 2.1 §4.2 气泡压力（默认关）

| 能力 | 说明 |
|------|------|
| 缓冲 | `bubble_tag` / `bubble_tag_prev` / `gas_rho` / `delta_phi` / 气泡表（V、V₀、ρ） |
| FS | \(\rho_g = \rho_b(\mathbf{x}) - 6\gamma\kappa\)（可再叠分离压） |
| 大气 | 触顶 `z=nz-1` 或 \(V > \min(V_{\mathrm{atm}}, 0.25N^3)\) → \(\rho=1\)，\(V_0=V\) |
| 封闭泡 | \(\rho=\mathrm{clip}(V_0/V,\,0.2,\,1.8)\) |
| 开关 | `LbmModel.vof_bubble_pressure=False`；算例 `--bubble-pressure` |

### 2.2 体积更新（与 Home 同构）

| 路径 | 行为 |
|------|------|
| 日常 | `surface_3` / `seal` 写 \(\Delta\phi\)；`volume[tag] \mathrel{-}= \Delta\phi`（无 tag 用 previous） |
| 拓扑变化 | GPU CCL 后按标签 reduce \(V\)、\(V_0\)（Home `reduce_label_rho` 意图） |
| 曾用方案 | 每步 \(\sum(1-\phi)\) 重求和：更稳但更重；已改为 Δφ |

### 2.3 GPU CCL（替换早期 host BFS）

| 项 | 内容 |
|----|------|
| 算法 | 设备端 6-连通 union-find → 稠密标签 `1..N` |
| 触发 | 初始化 / merge / split（`IF→F`） |
| 对照 | Home：`tDCCL`（YACCLAB）；wanphys：Warp 自研 CCL（角色对齐，实现不同） |
| 教训 | 早期「每步 host 全量 CCL」在 64³ + 12 substeps 下会假死；GPU 后 `sim≈60–150ms` 可交互 |

### 2.4 可选：分离压 / 小泡 σ / 近泡涡黏

| 项 | Home | wanphys | 算例默认（`--bubble-pressure`） |
|----|------|---------|--------------------------------|
| 分离压 disjoint | 有 | 有（φ 梯度射线，简化版 PLIC） | **开** |
| 小泡 \(6\sigma\to2\times10^{-4}\) | 有 | 有 | **关** |
| 近泡涡黏 | 写死开启 | 有（可关；仅界面格扫描） | **关** |

**为何涡黏默认关：** dam-break 早期大量碎 tag 会使 eddy 大面积抬高 \(\tau\)，池子过黏且 6³ 扫描过慢；Home 倒液基线 \(\nu=10^{-4}\) 且 eddy 更局部，观感不黏。

---

## 3. 与 Home-FSLBM 对齐表

### 3.1 已对齐（可用）

- D3Q27 + 矩量 + Hermite 重建 + NOCM 系碰撞  
- 融合 stream + 质量交换 + FS + collide → `surface_1/2/3`  
- Guo 半力气体平衡速度；\|u\| clamp  
- PLIC κ → Laplace；可选 fill-empty / wall \(f^{eq}\)（`--home-faithful`）  
- §4.2：tag、\(V_0/V\)、大气、Δφ 体积、merge/split 触发重标  

### 3.2 部分对齐

| 项 | 差距 |
|----|------|
| Hermite | Home 4 阶；wanphys 3 阶 |
| CCL | Home YACCLAB；wanphys Warp UF |
| Disjoint 法向 | Home PLIC；wanphys ∇φ |
| Zou–He / 内部固体 | GPU VOF 路径简化 |
| 工程外挂 | seal、quiet level、orphan、topup（非 Home；`--home-faithful` 可关） |

### 3.3 未做（明确延期）

- §4.4 **溶解气 D3Q7**（亨利定律、与气泡体积耦合）  
- 泡沫固化黏度（Home 3D 也基本无完整泡沫）  
- 原生 **tDCCL / YACCLAB** 移植  
- **倒液 / pouring** 主测算例（Home 主测；dam-break 不是）  
- **量化** `home_quant`  

---

## 4. Dam-break 验收观察（`--bubble-pressure`）

典型日志字段：

- **`sim=…ms`**：一帧墙钟（含 12 substeps）← **性能指标**  
- **`bub=` / `trap=` / `ρmax=` / `ccl=`**：气泡诊断  

观察摘要：

| 现象 | 解读 |
|------|------|
| `bub=1, trap=0, ρmax=1, ccl=*`（晚池） | 顶空大气腔，§4.2 大气分支正确 |
| 早期 `trap` 高但 `ρmax=1` | 新生小气团 \(V_0=V\)，尚未体现压缩 |
| `sim` 约 60–150ms（64³） | GPU CCL 后可交互；host CCL 时代会卡死 |
| 开全场 eddy 时过黏过慢 | 工况不适配；已默认关闭 |

**不承诺：** 开放自由面贴壁 O(Δx) 找平由 §4.2 改善——开放面不是封闭气泡。

---

## 5. 关键代码与开关

### 5.1 文件

- `wanphys/.../home_fp32_ref/vof_warp.py` — 缓冲、Δφ、GPU CCL、disjoint/eddy、步进  
- `wanphys/.../home_fp32_ref/bridge.py` — 模型 → GPU 步进  
- `wanphys/.../lbm/model.py` — `vof_bubble_*` 标志  
- `wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof.py` — `--bubble-pressure`  

### 5.2 主要标志

| 标志 | 默认 | 含义 |
|------|------|------|
| `vof_bubble_pressure` | False | 总开关 |
| `vof_bubble_atm_volume` | 1e6 | 大气体积阈值 |
| `vof_bubble_disjoint` | False | 分离压（算例开 bubble 时默认 True） |
| `vof_bubble_small_sigma` | False | 小泡 σ |
| `vof_bubble_eddy` | False | 近泡涡黏（dam-break 勿默认开） |

### 5.3 运行

```bash
python wanphys/examples/lbm/fluid_grid_lbm_dambreak_vof.py --backend home_fp32 --bubble-pressure
```

默认网格 **64³**，`home_fp32` 下约 **12 substeps / 帧**。

---

## 6. 性能与实现教训

1. **Host 每步全量 CCL 不可接受**（顶空 ~10⁵ 格 × 多 substep）。  
2. **Home 也不每步全量重标**：日常 Δφ + ρ；仅 merge/split 跑 GPU CCL。  
3. **IF→F 很密** 时 CCL 仍勤；GPU 可承受，host 不能。  
4. **涡黏公式照搬 + dam-break 碎泡** → 大面积高 τ；与倒液观感不可直接类比。  
5. Warp 内核注意：避免嵌套 `for`+`break`、避免同名 float/int 变量冲突。

---

## 7. 溶解气（§4.4）对 dam-break 的预估

溶解气 = 在 D3Q27 流体上 **再挂一套 D3Q7** 浓度场，不是替换 27 速。

对 **dam-break**：预期 **几乎无观感收益**（开放大气主导）；代价为每步额外 D3Q7。  
应用场景：倒液夹气、封闭泡溶气、泡沫相关；不宜作为 dam-break 找平手段。

---

## 8. 建议下一步

| 优先级 | 项 | 理由 |
|--------|----|------|
| 高 | **倒液 / pouring 算例** | 对齐 Home 主测；才能看到 `trap≥1` 且 `ρ≠1` |
| 中 | 收紧 eddy（仅小封闭泡、弱强度） | 保留稳泡能力，避免 dam-break 糖浆化 |
| 中 | Disjoint 用法向 PLIC 细化 | 更接近 Home 抑并泡 |
| 低 | D3Q7 溶解气 | H8；非 dam-break 刚需 |
| 低 | 移植 YACCLAB tDCCL | 现有 Warp CCL 已可用 |

---

## 9. 相关文档

- `docs/wanphys/lbm_home_roadmap_zh.md` — 总路线图（G3/G4）  
- `docs/wanphys/lbm_home_fslbm_one_cell_limit_zh.md` — 一格极限 / 贴壁找平  
- `docs/wanphys/lbm_home_vof_quiet_level_summary_zh.md` — quiet level / topup  
- 后端增量注释：`home_fp32_ref/__init__.py`（H0–H7 / NEXT=H8）

---

## 10. 状态标签

| 项 | 状态 |
|----|------|
| H6 GPU HOME-FREE VOF | 完成 |
| H7 §4.2 气泡压力 + GPU CCL + Δφ | 完成（默认关） |
| Disjoint | 可选，算例随 bubble 默认开 |
| 小泡 σ / 涡黏 | 实现保留，算例默认关 |
| H8 溶解气 / 泡沫 | 未开始 |
| 量化 | 延期 |
