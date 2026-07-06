# WanPhys LBM 刚体耦合操作观察实验手册

本文和 `lbm_rigid_coupling_guide_zh.md` 配套。前一份文档回答“这套实现是什么、为什么这样写”；本文回答“我怎样亲手观察它、验证它、拆开它、提出问题、再回到源码里找到答案”。

这不是单纯的测试清单，而是一组由浅入深的实验。每个实验都尽量包含：

- 观察：先看什么现象或源码。
- 问题：带着什么问题去跑。
- 运行：执行什么命令或写什么最小脚本。
- 结果：应该看到什么。
- 思考：这个结果说明了哪条代码路径。
- 回到源码：下一步应该读哪个文件、哪段函数。

你可以按顺序做，也可以挑自己不理解的部分做。建议先读一遍 `docs/wanphys/lbm_rigid_coupling_guide_zh.md`，再做本文实验。

## 0. 实验前准备

### 0.1 工作目录

本文所有命令默认在仓库根目录执行：

```powershell
D:\W_LBM - Copy
```

Windows 下建议使用仓库约定的 `pwsh` wrapper，例如：

```powershell
powershell.exe -Command "& pwsh -NoLogo -NoProfile -Command '<你的命令>'"
```

如果命令里有复杂正则、`$`、`|`、中文路径或多层引号，优先写一个临时 Python 文件再运行，别在 shell 里硬转义。

### 0.2 建议的临时实验目录

本文会多次让你“新建一个空文件做小实验”。建议建一个不提交的临时目录：

```powershell
mkdir scratch
```

实验脚本可以放在：

```text
scratch/lbm_exp_*.py
```

这些文件只是学习用，不需要提交。

### 0.3 最小验证命令

先确认当前实现能跑：

```powershell
uv run --python 3.11 --extra dev -m newton.tests -p 'test_lbm*.py'
```

期望：

```text
Running 1 test suites (4 total tests)
...
OK
```

这条命令会跑四个 focused tests：

- 静态刚体 rasterize。
- moving-wall 速度 stamp 和流体扰动。
- two-way force。
- off-center torque。

## 1. 实验一：先画出源码地图

### 观察

这次实现分布在几个文件里，不要一上来就钻进 900 行 kernel。先用搜索确认“关键词在哪”。

### 问题

`vel_solid_u/v/w`、`solid_phi`、two-way feedback 到底在哪些文件出现？

### 运行

```powershell
rg -n --encoding utf-8 -e "vel_solid" -e "solid_phi" -e "set_two_way" -e "accumulate_lbm" wanphys/_src/fluid/fluid_grid newton/tests/test_lbm_rigid_coupling.py wanphys/examples/fluid_grid_lbm_twoway_fsi.py
```

### 结果

你会看到大致分布：

- `lbm/state.py`：定义 `vel_solid_u/v/w` 和 `solid_phi`。
- `lbm/solver.py`：复制并传给 kernel。
- `lbm/kernels.py`：moving-wall bounce-back 使用它们。
- `coupling/grid_lbm_rigid_coupling.py`：每步清空、rasterize、stamp velocity、two-way feedback。
- `coupling/coupling_kernels.py`：rasterize、embed velocity、accumulate feedback。
- `newton/tests/test_lbm_rigid_coupling.py`：四个 focused tests。
- `wanphys/examples/fluid_grid_lbm_twoway_fsi.py`：headless two-way 示例。

### 思考

这一步的目的不是理解细节，而是确认职责边界：

```text
State 保存字段
Solver 消费字段
Coupling 准备字段
Kernel 计算字段
Test 证明字段
Example 演示字段
```

### 回到源码

先打开：

```text
wanphys/_src/fluid/fluid_grid/lbm/state.py
```

只读 `LbmState.__init__`、`clear()`、`clone()` 三段。

## 2. 实验二：空文件 import 实验

### 观察

以前 `GridLbmRigidCoupling` 没有从 package export，示例从 coupling 包导入会失败。本次修复后，导入应该成功。

### 问题

导出是否真的通了？如果只新建一个空 Python 文件，最小 import 能不能跑？

### 运行

新建空文件：

```text
scratch/lbm_exp_01_import.py
```

填入：

```python
from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling

print(GridLbmRigidCoupling.__name__)
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_01_import.py
```

### 结果

期望输出：

```text
GridLbmRigidCoupling
```

### 思考

这证明：

- `coupling/__init__.py` 已 import `GridLbmRigidCoupling`。
- `__all__` 中有 `"GridLbmRigidCoupling"`。
- 示例不需要再从深层文件导入。

### 反向小实验

只读源码，不要改文件。假设 `coupling/__init__.py` 没有这行：

```python
from .grid_lbm_rigid_coupling import GridLbmRigidCoupling
```

那么这个最小 import 会失败。你就能理解为什么“导出”虽然小，却是 Phase 1 的第一块拼图。

## 3. 实验三：观察 LbmState 字段形状

### 观察

`vel_solid_u/v/w` 是 MAC face velocity，不是 cell-centered velocity，所以 shape 不一样。

### 问题

为什么 `vel_solid_u` 是 `(nx + 1, ny, nz)`，而不是 `(nx, ny, nz)`？

### 运行

新建：

```text
scratch/lbm_exp_02_state_shapes.py
```

填入：

```python
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel

model = LbmModel(fluid_grid_res=(4, 3, 2), fluid_grid_cell_size=0.1, tau=0.55)
domain = LbmDomain(model)
domain.create_state()
state = domain.state

print("density", state.density.shape)
print("velocity_x", state.velocity_x.shape)
print("vel_u", state.vel_u.shape)
print("vel_v", state.vel_v.shape)
print("vel_w", state.vel_w.shape)
print("vel_solid_u", state.vel_solid_u.shape)
print("vel_solid_v", state.vel_solid_v.shape)
print("vel_solid_w", state.vel_solid_w.shape)
print("solid_phi", state.solid_phi.shape)
print("solid_body_id", state.solid_body_id.shape)
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_02_state_shapes.py
```

### 结果

期望类似：

```text
density (4, 3, 2)
velocity_x (4, 3, 2)
vel_u (5, 3, 2)
vel_v (4, 4, 2)
vel_w (4, 3, 3)
vel_solid_u (5, 3, 2)
vel_solid_v (4, 4, 2)
vel_solid_w (4, 3, 3)
solid_phi (4, 3, 2)
solid_body_id (4, 3, 2)
```

### 思考

`density` 和 `velocity_x/y/z` 是 cell-centered。

`vel_u/v/w` 和 `vel_solid_u/v/w` 是 MAC staggered face fields：

- x 方向 face 比 cell 多一个，所以 `u` 是 `(nx + 1, ny, nz)`。
- y 方向 face 比 cell 多一个，所以 `v` 是 `(nx, ny + 1, nz)`。
- z 方向 face 比 cell 多一个，所以 `w` 是 `(nx, ny, nz + 1)`。

移动刚体表面速度必须放在 face 上，因为 bounce-back 发生在 cell 和 solid neighbor 之间的半链接位置，本质上更接近面速度，而不是 cell center 速度。

### 回到源码

读：

```text
wanphys/_src/fluid/fluid_grid/lbm/state.py
```

重点找：

```python
self.vel_solid_u
self.vel_solid_v
self.vel_solid_w
```

## 4. 实验四：观察 clear 和 clone

### 观察

新增字段如果只在 `__init__` 里分配，却忘了 `clear()` 或 `clone()`，后续会出现很隐蔽的问题。

### 问题

`clear()` 是否真的清掉 solid velocity？`clone()` 是否真的复制 solid velocity？

### 运行

新建：

```text
scratch/lbm_exp_03_clear_clone.py
```

填入：

```python
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel

model = LbmModel(fluid_grid_res=(4, 3, 2), fluid_grid_cell_size=0.1, tau=0.55)
domain = LbmDomain(model)
domain.create_state()
state = domain.state

state.vel_solid_u.fill_(7.0)
state.vel_solid_v.fill_(8.0)
state.vel_solid_w.fill_(9.0)
wp.synchronize_device(model._device)

clone = state.clone()
wp.synchronize_device(model._device)
print("clone max u", float(clone.vel_solid_u.numpy().max()))
print("clone max v", float(clone.vel_solid_v.numpy().max()))
print("clone max w", float(clone.vel_solid_w.numpy().max()))

state.clear()
wp.synchronize_device(model._device)
print("after clear max u", float(state.vel_solid_u.numpy().max()))
print("after clear max v", float(state.vel_solid_v.numpy().max()))
print("after clear max w", float(state.vel_solid_w.numpy().max()))
print("after clear solid_phi min", float(state.solid_phi.numpy().min()))
print("after clear solid_body_id max", int(state.solid_body_id.numpy().max()))
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_03_clear_clone.py
```

### 结果

期望：

```text
clone max u 7.0
clone max v 8.0
clone max w 9.0
after clear max u 0.0
after clear max v 0.0
after clear max w 0.0
after clear solid_phi min 1000.0
after clear solid_body_id max -1
```

### 思考

这证明两件事：

- `clone()` 不会丢掉刚体表面速度。
- `clear()` 会把边界字段恢复到“没有固体”的状态。

如果没有这两步，双缓冲或 reset 后可能残留旧刚体边界，流体会被一个已经不存在的刚体影响。

### 回到源码

读：

```text
LbmState.clear()
LbmState.clone()
```

## 5. 实验五：双缓冲 persistent copy

### 观察

LBM 是双缓冲：`state_in` 和 `state_out` 每步交换。边界字段必须从 input state 复制到 output state。

### 问题

如果 solver 不复制 `vel_solid_u/v/w`，会发生什么？

### 运行

先看源码：

```powershell
rg -n --encoding utf-8 -e "Copy persistent fields" -e "vel_solid" wanphys/_src/fluid/fluid_grid/lbm/solver.py
```

再新建：

```text
scratch/lbm_exp_04_double_buffer.py
```

填入：

```python
import warp as wp

from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel

model = LbmModel(fluid_grid_res=(6, 6, 6), fluid_grid_cell_size=0.1, tau=0.55)
domain = LbmDomain(model)
domain.create_state()
domain.solver.initialize_equilibrium(domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

state_before = domain.state
state_before.vel_solid_u.fill_(0.123)
wp.synchronize_device(model._device)
print("before step max solid u", float(state_before.vel_solid_u.numpy().max()))

domain.step(1.0 / 60.0)
wp.synchronize_device(model._device)
state_after = domain.state
print("after step max solid u", float(state_after.vel_solid_u.numpy().max()))
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_04_double_buffer.py
```

### 结果

期望：

```text
before step max solid u 0.123...
after step max solid u 0.123...
```

### 思考

`state_before` 和 `state_after` 不是同一个缓冲。`after step` 还能看到 `0.123`，说明 `solver.py` 开头的 persistent copy 生效。

如果去掉这段：

```python
wp.copy(state_out.vel_solid_u, state_in.vel_solid_u)
wp.copy(state_out.vel_solid_v, state_in.vel_solid_v)
wp.copy(state_out.vel_solid_w, state_in.vel_solid_w)
```

`after step` 就可能变成 0，moving-wall 信息会在 step 后丢失。

### 回到源码

读：

```text
wanphys/_src/fluid/fluid_grid/lbm/solver.py
```

重点看 `step()` 一开始的 persistent copy。

## 6. 实验六：静态球 rasterize 成固体

### 观察

`GridLbmRigidCoupling` 每步会把刚体几何 rasterize 到 LBM grid。

### 问题

一个静止球真的会让 `solid_phi < 0` 吗？`solid_body_id` 真的会写成刚体编号吗？

### 运行

可以直接跑 focused test：

```powershell
uv run --python 3.11 --extra dev python -m unittest newton.tests.test_lbm_rigid_coupling.TestLbmRigidCoupling.test_static_sphere_rasterizes_into_lbm_solid_fields -v
```

也可以写观察脚本：

```text
scratch/lbm_exp_05_static_rasterize.py
```

填入：

```python
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys.rigid import RigidDomain, RigidModelBuilder

nx = ny = nz = 12
dh = 0.1
body_id = 0

fluid_model = LbmModel(fluid_grid_res=(nx, ny, nz), fluid_grid_cell_size=dh, tau=0.55)
fluid_domain = LbmDomain(fluid_model)
fluid_domain.create_state()
fluid_domain.solver.initialize_equilibrium(fluid_domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

rigid_builder = RigidModelBuilder(gravity=0.0)
rigid_builder.add_body(position=(0.6, 0.6, 0.6), label="static_sphere")
rigid_builder.add_shape_sphere(body_id, radius=0.26)
rigid_domain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
rigid_domain.create_state()
rigid_domain.state.body_qd.zero_()

coupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
coupling.add_body_sphere(body_idx=body_id, radius=0.26)
coupling.set_rigid_dynamics_enabled(False)
coupling.step(dt=1.0 / 60.0)
wp.synchronize_device(fluid_model._device)

solid_phi = fluid_domain.state.solid_phi.numpy()
solid_body_id = fluid_domain.state.solid_body_id.numpy()
print("solid cell count", int(np.sum(solid_phi < 0.0)))
print("body id cell count", int(np.sum(solid_body_id == body_id)))
print("solid_phi min", float(solid_phi.min()))
print("solid_phi max", float(solid_phi.max()))
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_05_static_rasterize.py
```

### 结果

你应该看到：

```text
solid cell count > 0
body id cell count > 0
solid_phi min < 0
solid_phi max 1000 或其他正值
```

### 思考

这说明 rasterize kernel 已经把球投到了 LBM grid 上。

`solid_phi < 0` 是 LBM bounce-back 的触发条件。`solid_body_id` 暂时对静态 one-way 不重要，但 two-way feedback 要靠它把流体力映射回正确刚体。

### 回到源码

读：

```text
GridLbmRigidCoupling.step()
ck.rasterize_all_body_sdf_warp
```

## 7. 实验七：ASCII 切片可视化 solid_phi

### 观察

只看 `solid cell count` 还不够直观。我们可以把中间 z 切片打印成 ASCII 图。

### 问题

球 rasterize 到网格上以后，从中间切片看起来像不像一个圆形截面？

### 运行

在实验六脚本末尾追加：

```python
mid_k = nz // 2
print("solid slice at k =", mid_k)
for j in range(ny - 1, -1, -1):
    row = []
    for i in range(nx):
        row.append("#" if solid_phi[i, j, mid_k] < 0.0 else ".")
    print("".join(row))
```

重新运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_05_static_rasterize.py
```

### 结果

你应该看到类似：

```text
............
............
.....##.....
....####....
....####....
.....##.....
............
............
```

实际形状会随半径、网格分辨率、SDF 离散化略有变化。

### 思考

这一步把抽象字段变成了可观察形状：

- `#` 表示固体。
- `.` 表示流体或空白。

如果你把球半径从 `0.26` 改成 `0.16`，`#` 会减少。改成 `0.36`，`#` 会增多。

### 回到源码

重点读 `coupling_kernels.py` 里 rasterize 相关逻辑，观察它如何根据 shape type 算 SDF。

## 8. 实验八：PNG 可视化 solid_phi

### 观察

ASCII 图足够轻量，但 PNG 更适合保存、比较、放进报告。

### 问题

能否把 `solid_phi` 的中间切片保存成图片，直观看出 solid mask？

### 运行

新建：

```text
scratch/lbm_exp_06_solid_slice_png.py
```

填入：

```python
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys.rigid import RigidDomain, RigidModelBuilder

nx = ny = nz = 32
dh = 0.05
body_id = 0

fluid_model = LbmModel(fluid_grid_res=(nx, ny, nz), fluid_grid_cell_size=dh, tau=0.55)
fluid_domain = LbmDomain(fluid_model)
fluid_domain.create_state()
fluid_domain.solver.initialize_equilibrium(fluid_domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

rigid_builder = RigidModelBuilder(gravity=0.0)
rigid_builder.add_body(position=(0.8, 0.8, 0.8), label="static_sphere")
rigid_builder.add_shape_sphere(body_id, radius=0.35)
rigid_domain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
rigid_domain.create_state()

coupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
coupling.add_body_sphere(body_idx=body_id, radius=0.35)
coupling.set_rigid_dynamics_enabled(False)
coupling.step(dt=1.0 / 60.0)
wp.synchronize_device(fluid_model._device)

solid = fluid_domain.state.solid_phi.numpy() < 0.0
slice_img = solid[:, :, nz // 2].T

plt.figure(figsize=(5, 5))
plt.imshow(slice_img, origin="lower", cmap="gray_r")
plt.title("solid_phi < 0, middle z slice")
plt.xlabel("i")
plt.ylabel("j")
plt.tight_layout()
plt.savefig("scratch/lbm_solid_slice.png", dpi=160)
print("saved scratch/lbm_solid_slice.png")
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_06_solid_slice_png.py
```

### 结果

打开：

```text
scratch/lbm_solid_slice.png
```

你应该看到一个二维圆形 solid mask。

### 思考

这类可视化适合排查：

- sphere 位置是否对。
- radius 是否符合预期。
- `body_idx` 是否映射正确。
- 网格分辨率是否太粗。

如果图上没有圆，优先检查：

- `coupling.add_body_sphere(body_idx=..., radius=...)` 是否调用。
- 刚体 `add_body` 的 position 是否在 fluid domain 内。
- radius 是否小到没有覆盖任何 cell。

## 9. 实验九：移动球 stamp 固体表面速度

### 观察

Moving-wall 的第一步不是看流体，而是看 `vel_solid_u/v/w` 有没有被刚体速度 stamp。

### 问题

给刚体球一个 x 向速度，`vel_solid_u` 会不会变成非零？

### 运行

直接跑 focused test：

```powershell
uv run --python 3.11 --extra dev python -m unittest newton.tests.test_lbm_rigid_coupling.TestLbmRigidCoupling.test_moving_sphere_stamps_surface_velocity_and_moves_fluid -v
```

也可以写脚本观察数值：

```text
scratch/lbm_exp_07_moving_wall.py
```

填入：

```python
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys.rigid import RigidDomain, RigidModelBuilder

nx = ny = nz = 12
dh = 0.1
body_id = 0
sphere_speed = 0.04

fluid_model = LbmModel(fluid_grid_res=(nx, ny, nz), fluid_grid_cell_size=dh, tau=0.55)
fluid_domain = LbmDomain(fluid_model)
fluid_domain.create_state()
fluid_domain.solver.initialize_equilibrium(fluid_domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

rigid_builder = RigidModelBuilder(gravity=0.0)
rigid_builder.add_body(position=(0.6, 0.6, 0.6), label="moving_sphere")
rigid_builder.add_shape_sphere(body_id, radius=0.26)
rigid_domain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
rigid_domain.create_state()
rigid_domain.state._body_qd = wp.array(
    [[sphere_speed, 0.0, 0.0, 0.0, 0.0, 0.0]],
    dtype=wp.spatial_vector,
    device=fluid_model._device,
)

coupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
coupling.add_body_sphere(body_idx=body_id, radius=0.26)
coupling.set_rigid_dynamics_enabled(False)

for step in range(3):
    coupling.step(dt=1.0 / 60.0)
    wp.synchronize_device(fluid_model._device)
    solid_u = fluid_domain.state.vel_solid_u.numpy()
    fluid_ux = fluid_domain.state.velocity_x.numpy()
    print(step, "max |solid_u|", float(np.max(np.abs(solid_u))), "max |fluid_ux|", float(np.max(np.abs(fluid_ux))))
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_07_moving_wall.py
```

### 结果

你应该看到：

```text
max |solid_u| > 0
max |fluid_ux| > 0
```

第一步可能 `fluid_ux` 很小，后面会逐步被扰动。

### 思考

这里有两层观察：

- `solid_u` 非零证明 rigid -> fluid velocity stamping 成功。
- `fluid_ux` 非零证明 LBM bounce-back 使用了 moving-wall correction，而不是仍然当静态墙。

### 回到源码

读：

```text
grid_lbm_rigid_coupling.py
```

找：

```python
ck.embed_all_solid_velocity_u
ck.embed_all_solid_velocity_v
ck.embed_all_solid_velocity_w
```

再读：

```text
kernels.py
```

找：

```python
_moving_wall_correction
```

## 10. 实验十：moving-wall 修正项的纯数学小实验

### 观察

`_moving_wall_correction` 的公式是：

```text
2 * w_i * rho * (c_i dot u_wall) / c_s^2
```

D3Q19 的 `c_s^2 = 1/3`，所以除以 `c_s^2` 相当于乘 3。

### 问题

如果 `w_i = 1/18`、`rho = 1`、`c_i dot u_wall = 0.06`，修正量是多少？

### 运行

新建：

```text
scratch/lbm_exp_08_correction_math.py
```

填入：

```python
w_i = 1.0 / 18.0
rho = 1.0
wall_dot = 0.06
inv_cs2 = 3.0
correction = 2.0 * w_i * rho * inv_cs2 * wall_dot
print(correction)
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_08_correction_math.py
```

### 结果

期望：

```text
0.02
```

### 思考

这个小实验帮你建立量级感：

- correction 不是魔法常数。
- 它随墙面速度线性变化。
- 墙速度为 0 时 correction 为 0。

这也解释为什么 moving-wall 测试不要求精确速度值，而只检查方向和非零扰动：真实 LBM step 里还有碰撞、streaming、边界条件、密度变化。

## 11. 实验十一：看 diagonal MAC face 平均规则

### 观察

轴向方向只穿过一个 face；对角方向穿过两个方向分量，采样更复杂。

### 问题

为什么 `_solid_wall_velocity_dot()` 里有多次 `0.5 * (...)`？

### 运行

定位代码：

```powershell
rg -n --encoding utf-8 -e "_solid_wall_velocity_dot" -e "0.5" wanphys/_src/fluid/fluid_grid/lbm/kernels.py
```

然后打开函数，重点看：

```python
if cy != 0:
    u_wall = 0.5 * (...)
```

### 结果

你会看到：

- x component 采样 `vel_solid_u`。
- 如果 direction 同时有 y component，就平均另一个相邻 u face。
- 如果 direction 同时有 z component，也类似。

### 思考

当前实现不是 SDF 真实交点插值，而是一个局部 MAC-face 平均规则。

它的优点：

- 本地。
- 稳定。
- 不需要额外几何信息。
- 容易测试。

它的限制：

- 曲面边界上不如高阶插值精确。
- diagonal link 的真实半链接位置可能和平均 face 不完全一致。

这就是为什么主 guide 里说：这是 moving-wall 的可运行起点，不是最终几何精度上限。

## 12. 实验十二：two-way force 的 headless 示例

### 观察

Two-way feedback 最直观的输出是 `body_f`。

### 问题

开 two-way 后，一个处在均匀流里的球会不会得到非零流体力？

### 运行

```powershell
uv run --python 3.11 python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi --steps 2
```

### 结果

期望类似：

```text
LBM two-way feedback body_f=[0.0016, ..., ..., ..., ..., ...] |force|=0.001600
```

数值可能有微小 GPU/backend 浮动，但 `|force|` 应该非零。

### 思考

这个例子证明：

```text
uniform fluid velocity
-> LBM step
-> boundary feedback scan
-> rigid_state.body_f
-> print force norm
```

这不是完整物理 benchmark，只是 headless smoke。它要证明的是接口路径通了。

### 回到源码

读：

```text
wanphys/examples/fluid_grid_lbm_twoway_fsi.py
```

重点看：

```python
coupling.set_two_way_feedback_enabled(True, force_scale=force_scale)
body_force = rigid_domain.state.get_body_force(0)
```

## 13. 实验十三：two-way on/off 对照

### 观察

默认行为必须仍然是一向耦合。如果不开 two-way，`body_f` 不应被 LBM feedback 写入。

### 问题

同一个场景，two-way disabled 和 enabled 的 `body_f` 是否不同？

### 运行

直接跑 focused test：

```powershell
uv run --python 3.11 --extra dev python -m unittest newton.tests.test_lbm_rigid_coupling.TestLbmRigidCoupling.test_two_way_feedback_accumulates_body_force_when_enabled -v
```

### 结果

期望：

```text
ok
```

测试内部做了：

```python
coupling.step(...)
one_way_force = rigid_domain.state.get_body_force(body_id)
self.assertEqual(norm(one_way_force), 0.0)

coupling.set_two_way_feedback_enabled(True)
coupling.step(...)
two_way_force = rigid_domain.state.get_body_force(body_id)
self.assertGreater(norm(two_way_force[:3]), 0.0)
```

### 思考

这个对照非常重要。

如果 one-way 也会产生 `body_f`，说明默认行为破坏了已有示例。

如果 two-way 开启后仍然是 0，说明 feedback kernel 没跑、`solid_body_id` 没映射、或 `body_f` 没写入。

### 回到源码

读：

```text
GridLbmRigidCoupling.set_two_way_feedback_enabled()
GridLbmRigidCoupling.step()
```

找：

```python
if self._two_way_feedback_enabled:
    rigid_state.clear_forces()
    wp.launch(ck.accumulate_lbm_boundary_feedback_all_bodies, ...)
```

## 14. 实验十四：单 face torque 显微镜

### 观察

力矩不是凭空出现的。它来自：

```text
torque = cross(face_pos - com_world, df)
```

### 问题

如果只有一个 off-center fluid-solid face，能不能单独观察到 torque？

### 运行

直接跑 focused test：

```powershell
uv run --python 3.11 --extra dev python -m unittest newton.tests.test_lbm_rigid_coupling.TestLbmRigidCoupling.test_two_way_feedback_accumulates_torque_for_off_center_face -v
```

### 结果

期望：

```text
ok
```

### 思考

这个测试手动 seed：

```python
density[5, 6, 4] = 1.0
velocity_x[5, 6, 4] = -0.2
solid_phi[4, 6, 4] = -1.0
solid_body_id[4, 6, 4] = 0
```

意思是：

- fluid cell 在 `(5, 6, 4)`。
- 它左边 `(4, 6, 4)` 是 solid。
- 流体速度朝左，冲向 solid。
- 接触 face 相对 COM 有偏移。

所以应当产生：

- x 向力。
- z 向 torque。

测试断言：

```python
self.assertLess(float(body_force[0]), 0.0)
self.assertGreater(float(abs(body_force[5])), 1.0e-8)
```

这证明 feedback kernel 不只是加 force，也确实累加 torque。

### 回到源码

读：

```text
coupling_kernels.py
```

重点看：

```python
df = rho * rel_un * face_area * normal
torque = wp.cross(face_pos - com_world, df)
wp.atomic_add(body_f, body_id, wp.spatial_vector(...))
```

## 15. 实验十五：把 force_scale 当旋钮

### 观察

当前 two-way feedback 输出在 lattice units 下，再乘 `feedback_force_scale`。

### 问题

`force_scale` 改成 10，输出力是否大约放大 10 倍？

### 运行

分别运行：

```powershell
uv run --python 3.11 python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi --steps 2 --force-scale 1.0
```

```powershell
uv run --python 3.11 python -m wanphys.examples.lbm.fluid_grid_lbm_twoway_fsi --steps 2 --force-scale 10.0
```

### 结果

你应该看到 `|force|` 大约放大 10 倍。

### 思考

这说明：

- feedback kernel 先算 lattice-scale 近似力。
- 最后统一乘 `feedback_force_scale`。
- 这不是完整物理单位换算，只是一个清晰命名的缩放入口。

### 回到源码

读：

```text
grid_lbm_rigid_coupling.py
coupling_kernels.py
fluid_grid_lbm_twoway_fsi.py
```

找：

```python
self._feedback_force_scale
feedback_force_scale
--force-scale
```

## 16. 实验十六：读 step 顺序并手画流程

### 观察

很多耦合 bug 不是公式错，而是顺序错。

### 问题

为什么 two-way feedback 要在 fluid step 之后、rigid step 之前？

### 运行

定位注释：

```powershell
rg -n --encoding utf-8 -e "Reset solid" -e "Rasterise" -e "Embed rigid" -e "Advance fluid" -e "feedback" -e "advance rigid" wanphys/_src/fluid/fluid_grid/coupling/grid_lbm_rigid_coupling.py
```

然后手画：

```text
clear fields
-> rasterize rigid
-> embed solid velocity
-> fluid step
-> accumulate body_f
-> rigid step
```

### 结果

你应该能在源码中看到这个顺序。

### 思考

如果 feedback 放在 fluid step 之前：

- 它看到的是旧流体速度。
- 当前帧流体-边界交互还没发生。

如果 rigid step 放在 feedback 之前：

- 刚体 solver 看不到本帧 `body_f`。

所以正确顺序是：

```text
fluid produces feedback
then rigid consumes feedback
```

## 17. 实验十七：可视化速度扰动切片

### 观察

Moving-wall 不是只看一个最大值，也可以看速度场切片。

### 问题

移动球附近的 `velocity_x` 是否形成局部扰动？

### 运行

新建：

```text
scratch/lbm_exp_09_velocity_slice_png.py
```

填入：

```python
import matplotlib.pyplot as plt
import numpy as np
import warp as wp

from wanphys._src.fluid.fluid_grid.coupling import GridLbmRigidCoupling
from wanphys._src.fluid.fluid_grid.lbm import LbmDomain, LbmModel
from wanphys.rigid import RigidDomain, RigidModelBuilder

nx = ny = nz = 24
dh = 0.05
body_id = 0
sphere_speed = 0.03

fluid_model = LbmModel(fluid_grid_res=(nx, ny, nz), fluid_grid_cell_size=dh, tau=0.55)
fluid_domain = LbmDomain(fluid_model)
fluid_domain.create_state()
fluid_domain.solver.initialize_equilibrium(fluid_domain.state, rho0=1.0, u0=(0.0, 0.0, 0.0))

rigid_builder = RigidModelBuilder(gravity=0.0)
rigid_builder.add_body(position=(0.6, 0.6, 0.6), label="moving_sphere")
rigid_builder.add_shape_sphere(body_id, radius=0.18)
rigid_domain = RigidDomain(rigid_builder.finalize(device=fluid_model._device))
rigid_domain.create_state()
rigid_domain.state._body_qd = wp.array(
    [[sphere_speed, 0.0, 0.0, 0.0, 0.0, 0.0]],
    dtype=wp.spatial_vector,
    device=fluid_model._device,
)

coupling = GridLbmRigidCoupling(fluid_domain, rigid_domain)
coupling.add_body_sphere(body_idx=body_id, radius=0.18)
coupling.set_rigid_dynamics_enabled(False)

for _ in range(8):
    coupling.step(dt=1.0 / 60.0)

wp.synchronize_device(fluid_model._device)
ux = fluid_domain.state.velocity_x.numpy()
solid = fluid_domain.state.solid_phi.numpy() < 0.0
mid = nz // 2

plt.figure(figsize=(6, 5))
plt.imshow(ux[:, :, mid].T, origin="lower", cmap="coolwarm")
plt.contour(solid[:, :, mid].T, levels=[0.5], colors="black", linewidths=1)
plt.colorbar(label="velocity_x")
plt.title("moving-wall velocity_x slice")
plt.tight_layout()
plt.savefig("scratch/lbm_velocity_x_slice.png", dpi=160)
print("saved scratch/lbm_velocity_x_slice.png")
```

运行：

```powershell
uv run --python 3.11 python scratch/lbm_exp_09_velocity_slice_png.py
```

### 结果

打开：

```text
scratch/lbm_velocity_x_slice.png
```

你应该看到球附近有 `velocity_x` 扰动，黑色 contour 是 solid mask。

### 思考

这张图能同时检查：

- solid boundary 是否在预期位置。
- moving-wall 是否影响流体。
- 流体扰动是否集中在刚体附近。

如果速度场全 0，回去查：

- `rigid_domain.state._body_qd` 是否设置。
- `vel_solid_u` 是否非零。
- `collide_stream_bounceback_kernel` 是否传入了 `vel_solid_*`。

## 18. 实验十八：故障定位练习

### 观察

理解代码的最好方式之一是反推故障。

### 问题

如果某个测试失败，应该从哪里开始查？

### 运行与思考

#### 18.1 import 失败

现象：

```text
ImportError: cannot import name 'GridLbmRigidCoupling'
```

检查：

```text
wanphys/_src/fluid/fluid_grid/coupling/__init__.py
```

重点：

```python
from .grid_lbm_rigid_coupling import GridLbmRigidCoupling
__all__ = [...]
```

#### 18.2 static rasterize 没有 solid cell

现象：

```text
solid cell count 0
```

检查：

- 刚体位置是否在 domain 内。
- radius 是否太小。
- 是否调用 `coupling.add_body_sphere(...)`。
- `rasterize_all_body_sdf_warp` 是否 launch。

#### 18.3 moving-wall 里 `solid_u` 非零但 `fluid_ux` 为 0

说明：

- rigid velocity stamping 可能成功了。
- LBM kernel 可能没用 moving-wall correction。

检查：

```text
solver.py
kernels.py
```

重点：

```python
state_out.vel_solid_u, state_out.vel_solid_v, state_out.vel_solid_w
_moving_wall_correction(...)
```

#### 18.4 two-way enabled 但 force 为 0

检查：

- `set_two_way_feedback_enabled(True)` 是否调用。
- `solid_body_id` 是否有正确 body id。
- 流体速度是否真的冲向 solid boundary。
- `rigid_state.body_f` 是否存在。
- feedback kernel 是否在 fluid step 后 launch。

#### 18.5 force 有但 torque 没有

检查：

- contact position 是否相对 COM 有偏移。
- `face_pos - com_world` 是否非零且不与 `df` 平行。
- `wp.cross(...)` 的结果是否被写入 spatial vector 后 3 分量。

## 19. 实验十九：源码阅读打卡表

做完上面的实验后，按这个表回读源码。

| 问题 | 你应该能指到的源码 |
| --- | --- |
| `vel_solid_u` 在哪里分配？ | `LbmState.__init__` |
| `vel_solid_u` 为什么 shape 有 `nx + 1`？ | MAC face layout |
| `clear()` 为什么要清 solid velocity？ | 避免旧刚体边界残留 |
| 双缓冲为什么要 copy solid fields？ | `LbmSolver.step()` persistent copy |
| `solid_phi < 0` 在哪里被使用？ | `collide_stream_bounceback_kernel` |
| moving-wall correction 公式在哪里？ | `_moving_wall_correction` |
| diagonal 速度采样在哪里？ | `_solid_wall_velocity_dot` |
| rigid surface velocity 在哪里 stamp？ | `embed_all_solid_velocity_u/v/w` |
| two-way 开关在哪里？ | `set_two_way_feedback_enabled` |
| `body_f` 在哪里被清空？ | `GridLbmRigidCoupling.step()` two-way 分支 |
| `body_f` 在哪里被写入？ | `accumulate_lbm_boundary_feedback_all_bodies` |
| torque 在哪里计算？ | `_accumulate_lbm_face_feedback` |
| 哪个测试证明 static boundary？ | `test_static_sphere_rasterizes_into_lbm_solid_fields` |
| 哪个测试证明 moving-wall？ | `test_moving_sphere_stamps_surface_velocity_and_moves_fluid` |
| 哪个测试证明 two-way force？ | `test_two_way_feedback_accumulates_body_force_when_enabled` |
| 哪个测试证明 torque？ | `test_two_way_feedback_accumulates_torque_for_off_center_face` |

## 20. 实验二十：用一句话复述每条数据流

最后不要看源码，试着自己写出这些句子。

### 20.1 静态边界

```text
Rigid body geometry -> rasterize -> solid_phi / solid_body_id -> LBM bounce-back -> no penetration
```

### 20.2 移动边界

```text
Rigid body q/qd -> embed surface velocity -> vel_solid_u/v/w -> moving-wall correction -> fluid velocity perturbation
```

### 20.3 Two-way feedback

```text
LBM macro velocity near solid boundary -> boundary feedback scan -> force/torque -> rigid_state.body_f -> rigid step
```

### 20.4 当前近似边界

```text
This is macro-velocity boundary feedback, not final D3Q19 population-level momentum exchange.
```

如果你能不看文档写出这四句话，再回到源码能指出每一步在哪里实现，就说明你已经把这次工作吃透了。

## 21. 建议的完整学习路线

按这个顺序做一遍：

1. 实验一：源码地图。
2. 实验二：import 空文件。
3. 实验三：state shape。
4. 实验四：clear / clone。
5. 实验五：双缓冲 copy。
6. 实验六：静态 rasterize。
7. 实验七：ASCII solid slice。
8. 实验八：PNG solid slice。
9. 实验九：moving-wall 数值观察。
10. 实验十：moving-wall 公式量级。
11. 实验十一：diagonal 采样规则。
12. 实验十二：two-way example。
13. 实验十三：two-way on/off 对照。
14. 实验十四：torque 显微镜。
15. 实验十五：force_scale 旋钮。
16. 实验十六：step 顺序手画。
17. 实验十七：速度扰动 PNG。
18. 实验十八：故障定位。
19. 实验十九：源码阅读打卡表。
20. 实验二十：一句话复述数据流。

这条路线覆盖了：

- 命令行输出。
- focused unittest。
- 空文件独立小实验。
- ASCII 可视化。
- PNG 可视化。
- 数学量级检查。
- 源码定位。
- 故障反推。

做完以后，你理解的不只是“代码能跑”，而是能解释“为什么这些字段存在、为什么 step 顺序这样排、为什么测试这样写、为什么当前 two-way 仍是近似”。

