# LBM CUDA 验收记录

> 模板 + 本机运行结果。每完成一次 GPU 验收，在下方追加一节。

## 记录模板

```text
日期:
GPU 型号:
CUDA Driver / Toolkit:
Warp 版本:
测试网格:
六种组合一步:
HOME reconstruction/NOCM 编译:
FullF Shan-Chen smoke:
小规模多步有限性:
HOME-NOCM population scratch:
显存占用:
结论: pass / fail
备注:
```

## 验收脚本

项目根目录：

```bash
uv run python scripts/lbm_cuda_acceptance.py
```

## 2026-07-16 本机运行

```text
日期: 2026-07-16
GPU 型号: NVIDIA GeForce RTX 4070 SUPER (12 GiB, sm_89)
CUDA Driver / Toolkit: Driver 13.1 / Toolkit 12.9 (Warp report)
Warp 版本: 1.12.0
测试网格:
  - 六种组合一步: 8^3
  - FullF SC smoke: 12^3, 5 steps
  - shear-wave: 8x16x4, 20 steps
六种组合一步: PASS
HOME reconstruction/NOCM 编译: PASS (encoding module compiled on cuda:0)
FullF Shan-Chen smoke: PASS
小规模多步有限性: PASS (shear-wave amp=0.007267 theory=0.007346 rel_err=0.011)
HOME-NOCM population scratch: PASS (no population scratch)
home_nocm+SC fail-fast: PASS
显存占用: Warp Device API 无 get_memory_info；nvidia-smi 可另记
结论: pass
备注: overall PASS from scripts/lbm_cuda_acceptance.py
```
