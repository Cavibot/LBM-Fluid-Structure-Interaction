# P1 Authoritative VOF 状态与初始化代码工程计划

## 1. 文档状态

| 项目 | 内容 |
|---|---|
| 阶段 | P1 |
| 类型 | 可执行代码工程计划 |
| 前置阶段 | P0 `CPU_ACCEPTED` |
| 当前实施状态 | `IMPLEMENTED_CPU_VERIFIED`；等待提交冻结 |
| 目标状态 | `CPU_ACCEPTED` |
| 物理范围 | authoritative VOF 状态、初始化、双缓冲和不变量 |
| 明确不包含 | 固体、气泡、泡沫、质量平流、自由面补全、类型转换、几何、表面张力 |

P0 冻结记录见 [07-p0-baseline.md](07-p0-baseline.md)。P1 必须在该基线上增加正式
VOF 状态，不得重新设计或修改已经冻结的 LBM 数值算法。

## 2. P1 目标

P1 的目标是让以下操作成立：

```text
构造 interface_model="vof" 的模型
→ 分配 state_in.vof / state_out.vof
→ LBM 使用 rho0/u0 初始化 FullF 或 HOME kinetic
→ LBM 写出统一 state.density
→ VOF 使用实际 density 和 phi0 初始化 mass/phi/cell_type
→ 校验拓扑和不变量
```

P1 完成后，以下操作仍必须 fail-fast：

```text
domain.step()
质量平流
gas-to-interface population completion
类型转换和质量重分配
新运行时 interface kinetic 初始化
normal / PLIC / curvature / surface tension
solid/free-surface coupling
bubble / foam topology
```

P1 不是“可以运行自由表面”的阶段，而是正式 VOF 状态能够被安全构造、初始化、
复制、清理和验证的阶段。

## 3. 已冻结架构边界

### 3.1 Existing LBM 是只读后端

P1 不修改：

```text
FullF/HOME kinetic encoding
HOME population reconstruction
streaming
moment collection
force closure
SRT/TRT/MRT/NOCM collision
FullF/HOME 输出编码
普通 domain boundary
```

VOF 对 density 的唯一读取接口是：

```python
state.density
```

VOF 不得直接读取：

```text
HomeLbmState.rho
HomeLbmState.rho_u_*
HomeLbmState.rho_s_*
```

FullF 与 HOME 都必须由现有 `LbmSolver.initialize_equilibrium()` 写出一致的
`state.density`。encoding-specific logical population provider 只在 P2 第一次需要
质量通量时接入，P1 不新增 population adapter。

### 3.2 mass 是唯一守恒权威

持久关系固定为：

```text
mass      persistent authoritative
phi       persistent derived cache
cell_type persistent topology
```

初始化时：

\[
mass^0=\rho^0\phi_{\mathrm{input}},
\qquad
\phi^0=\frac{mass^0}{\rho^0}.
\]

其中 \(\rho^0\) 必须读取 LBM 实际初始化后的 `state.density`，不能直接使用 API
参数 `rho0` 代替。

### 3.3 最小持久状态

P1 的 authoritative 状态只有：

```python
class VofGridState:
    mass
    phi
    cell_type
```

P1 不加入：

```text
initialized
epoch
active_mask
normal
plic_offset
curvature
density 副本
HOME moment 副本
transition scratch
redistribution scratch
```

没有派生几何缓存就没有缓存失效问题，因此 P1 不需要 epoch。active mask 未来由
`cell_type != GAS` 即时派生，不保存第二份权威状态。

### 3.4 phi0 是规范初始化输入

第一版公共初始化接受：

```python
rho0
u0
phi0
```

`phi0` 保持一等输入，用于：

- 小网格手算测试；
- NumPy oracle；
- 外部 VOF 场导入；
- 精确构造 interface cell；
- 非法拓扑测试；
- 后续 checkpoint/restart 的基础。

未来可以增加：

```python
geometry -> rasterize/sample -> phi0
```

但几何初始化器只能生成 `phi0`，不能形成第二套 VOF 初始化逻辑。P1 不实现
geometry-to-phi rasterizer。

### 3.5 初始 L-G 邻接策略

P1 不静默修复 `LIQUID ↔ GAS` 直接邻接。初始化后使用 D3Q19 的 18 个非零格链检查：

```text
发现 LIQUID-GAS 直接邻接
→ ValueError
→ 报告总数和首个 cell/方向
```

用户必须提供：

```text
LIQUID ↔ INTERFACE ↔ GAS
```

Home-FSLBM 参考实现会把邻接 FLUID 的 GAS 自动改为 `INTERFACE, phi=0.5`，同时平均
邻居 kinetic。P1 不复制该策略，因为它会改变初始几何和质量，并提前引入 P5 的
新 interface kinetic 初始化决策。

### 3.6 GAS 与不支持 domain 的边界

P1 将所有 GAS cell 统一视为后续恒压 atmosphere 模型的一部分，不检查：

```text
GAS 连通分量数量
GAS 是否接触外边界
封闭 GAS 是否代表气泡
```

用户必须保证 `phi0` 不包含需要独立压力演化的气泡或泡沫。P1 仍检查 D3Q19
LIQUID-GAS 直接邻接，确保自由面至少有一层 INTERFACE。

P1 允许普通 domain 外边界和静态 bounce-back 外墙，但拒绝：

```text
任一 solid_phi < 0
moving_wall
cut_link
刚体耦合
```

## 4. 初始化数据流

公共入口：

```python
domain.initialize_vof(
    phi0,
    rho0=1.0,
    u0=(0.0, 0.0, 0.0),
)
```

严格执行顺序：

```text
1. 检查 interface_model、rho0/u0 和不支持的 solid 配置
2. 验证原始 phi0 shape、浮点 dtype、finite、[0,1]
3. 转为 C-contiguous float32，并使用精确 0/1 分类
4. 使用 model._periodic_ints 检查 D3Q19 L-G 邻接
5. 创建不属于 domain 当前状态的 candidate_in
6. LBM.initialize_equilibrium(candidate_in, rho0, u0)
7. 检查 candidate_in 所有 density finite 且 >0
8. 使用实际 candidate_in.density 生成 canonical mass/phi/type
9. 运行 initialized-state 不变量检查
10. candidate_out = candidate_in.clone()
11. 检查两个 candidate buffer 数值相同、数组独立
12. 全部成功后一次性替换 domain 的 state_in/state_out
```

步骤 8 按类型规范化：

```text
GAS:
    mass = 0
    phi = 0

LIQUID:
    mass = density
    phi = 1

INTERFACE:
    mass = density * phi_input
    phi = mass / density
```

已有 LBM 的 `solid_phi` 只用于检查是否存在 solid cell，不参与 VOF 分类、质量计算或
拓扑规则。P1 不定义 solid cell 的 VOF 状态。

重复调用 `initialize_vof()` 表示完整重新初始化。新初始化失败时保留上一次成功的
两个 buffer；首次初始化失败时不提交 candidate。

## 5. 文件新增清单

### 5.1 `vof/initialization.py`

建议路径：

```text
wanphys/_src/fluid/fluid_grid/lbm/vof/initialization.py
```

职责：

- host-side `phi0` 验证和分类；
- D3Q19 topology validation；
- 调度一次性 Warp 初始化 kernel；
- 只读取调用者传入的 `density`；
- 在进入初始化 kernel 前检查现有 `solid_phi`，发现 solid 立即拒绝；
- 只写目标 `VofGridState`。

关键函数：

```python
def classify_initial_phi(
    phi0: np.ndarray,
) -> np.ndarray:
    """Return canonical uint8 GAS/INTERFACE/LIQUID labels."""
```

```python
def validate_initial_topology(
    cell_type: np.ndarray,
    *,
    periodic: tuple[bool, bool, bool],
) -> None:
    """Reject LIQUID-GAS links and unsupported gas topology."""
```

```python
def prepare_initial_vof(
    phi0: np.ndarray,
    *,
    shape: tuple[int, int, int],
    periodic: tuple[bool, bool, bool],
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return canonical phi input and cell types."""
```

```python
def validate_no_solid_cells(solid_phi: wp.array3d) -> None:
    """Reject any domain containing an LBM solid cell."""
```

```python
@wp.kernel
def initialize_vof_mass_kernel(
    density: wp.array3d(dtype=float),
    phi_input: wp.array3d(dtype=float),
    cell_type_input: wp.array3d(dtype=wp.uint8),
    mass_out: wp.array3d(dtype=float),
    phi_out: wp.array3d(dtype=float),
    cell_type_out: wp.array3d(dtype=wp.uint8),
) -> None:
    """Build canonical mass/phi/type from initialized LBM density."""
```

```python
def initialize_vof_fields(
    density: wp.array3d,
    target: VofGridState,
    phi0: np.ndarray,
    cell_type: np.ndarray,
) -> None:
    """Initialize one VOF state from an already initialized LBM density."""
```

P1 第一版可以只接受 NumPy `phi0`。初始化不是热路径；不需要为了接受任意 device
array 提前增加复杂输入多态。

### 5.2 `vof/validation.py`

建议路径：

```text
wanphys/_src/fluid/fluid_grid/lbm/vof/validation.py
```

职责是只读检查实际数组，不存储 `initialized/epoch`。

关键函数：

```python
def validate_empty_vof_state(vof: VofGridState) -> None:
    """Validate the canonical all-GAS state created by clear()."""
```

```python
def validate_initialized_vof_state(
    state: LbmStateBase,
    *,
    periodic: tuple[bool, bool, bool],
    atol: float,
    rtol: float,
) -> None:
    """Validate density/mass/phi/type/topology invariants."""
```

若要进一步减少模块数量，P1 可以暂时把 validation 放入 `initialization.py`。不得为
三个数组引入大型 validator class hierarchy。

### 5.3 `test_lbm_vof_p1.py`

建议路径：

```text
newton/tests/test_lbm_vof_p1.py
```

P1 先使用一个测试文件，不按 state/config/initialization 拆成多个文件。

## 6. 文件修改清单

### 6.1 `vof/state.py`

在现有 `DebugMockScToVofState` 旁新增 `VofGridState`，不搬迁 P0 debug class，减少
基线扰动。

关键方法：

```python
class VofGridState:
    def __init__(self, shape, device) -> None
    def clear(self) -> None
    def copy_to(self, target: VofGridState) -> None
    def clone(self) -> VofGridState
```

构造和 clear 后：

```text
mass      = 0
phi       = 0
cell_type = GAS
```

clone/copy 只处理三个数组。

### 6.2 `lbm/state.py`

修改：

```text
LbmStateBase.__init__
LbmStateBase._clear_common
LbmStateBase._copy_common_to
```

分配矩阵：

| interface model | `state.vof` | `state.debug_mock_sc_to_vof` |
|---|---:|---:|
| `off` | `None` | `None` |
| `shan_chen`, debug off | `None` | `None` |
| `shan_chen`, debug on | `None` | allocated |
| `vof` | allocated | `None` |

`FullFLbmState.clone()` 和 `HomeLbmState.clone()` 继续通过 `_copy_common_to()` 复制正式
VOF，不增加 encoding-specific VOF 分支。

### 6.3 `model.py`

修改 `LbmModel.__post_init__()`：

1. 删除模型构造阶段对 `interface_model="vof"` 的 `NotImplementedError`；
2. 将 observation 规则收紧为只允许 Shan-Chen：

```python
if (
    resolved_interface is not InterfaceModel.SHAN_CHEN
    and self.debug_vof_observation
):
    raise ValueError(
        "debug_vof_observation requires interface_model='shan_chen'"
    )
```

3. 保留“非 Shan-Chen 模式配置非零 `G`”的现有拒绝规则；
4. 不新增尚未被 P1 使用的 `VofConfig`。

以下配置按首次消费者推迟：

```text
atmosphere_pressure -> P3
epsilon_phi         -> P4
surface_tension     -> P6
```

### 6.4 `solver.py`

修改 `LbmSolver.create_state()`：

```python
if self.model.interface_model == "vof" and requires_grad:
    raise NotImplementedError(
        "authoritative VOF does not support requires_grad"
    )
```

在 `LbmSolver.step()` 任何 copy/stream/写入之前加入：

```python
if self.model.interface_model == "vof":
    raise NotImplementedError(
        "P1 supports authoritative VOF initialization only; "
        "VOF time stepping is not implemented"
    )
```

新增：

```python
def initialize_vof_state(
    self,
    state: LbmStateBase,
    phi0: np.ndarray,
) -> None:
    """Initialize state.vof from the state's already initialized density."""
```

前置条件：

```text
model.interface_model == "vof"
state.vof is not None
state.density 已由 initialize_equilibrium 写出
```

该函数先调用 `prepare_initial_vof()`，再调用 `initialize_vof_fields()`，不得读取
encoding-specific kinetic 字段。domain 已经准备过输入时可以使用 solver 的内部
prepared-data 路径，避免在 candidate 提交阶段重复分类。

现有 `initialize_equilibrium()` 不修改。

### 6.5 `domain.py`

新增唯一公共入口：

```python
def initialize_vof(
    self,
    phi0: np.ndarray,
    *,
    rho0: float = 1.0,
    u0: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> LbmStateBase:
    """Initialize LBM kinetic state and authoritative VOF buffers."""
```

实现调度：

```python
prepared_phi, prepared_type = prepare_initial_vof(phi0, ...)
candidate_in = solver.create_state()
solver.initialize_equilibrium(candidate_in, rho0, u0)
solver._initialize_prepared_vof_state(candidate_in, prepared_phi, prepared_type)
validate_initialized_vof_state(candidate_in, ...)
candidate_out = candidate_in.clone()
validate buffer equality and non-aliasing
self._state_in, self._state_out = candidate_in, candidate_out
return candidate_in
```

不为测试增加公开 `next_state` 属性。测试可以分别测试 `clone()`，或在模块内部允许的
范围检查 `_state_out`。

### 6.6 导出文件

修改：

```text
wanphys/_src/fluid/fluid_grid/lbm/vof/__init__.py
wanphys/_src/fluid/fluid_grid/lbm/__init__.py
```

公共导出最小化为：

```text
VofGridState
VofCellType
```

用户通过 `LbmDomain.initialize_vof()` 初始化。不将内部
`initialize_vof_fields()`、validator 或 kernel 提升为顶层公共 API。

## 7. 明确禁止修改

P1 不应改动：

```text
encoding.py
streaming.py
moments.py
collisions.py
forcing.py
boundaries.py
vof/geometry.py
vof/debug.py
vof/visualization.py
```

P1 不新增：

```text
population.py
mass_advection.py
surface.py
transition.py
redistribution.py
kinetic_init.py
geometry cache
```

若实现 P1 时需要修改上述文件，必须先证明不是把 P2-P6 或已有 LBM 工作提前混入。

## 8. 关键调用链

```text
LbmDomain.initialize_vof(phi0, rho0, u0)
    │
    ├─ prepare_initial_vof(phi0, effective_periodic)
    │
    ├─ create candidate_in
    │
    ├─ LbmSolver.initialize_equilibrium(candidate_in, rho0, u0)
    │      ├─ initialize FullF or HOME kinetic
    │      └─ write candidate_in.density / velocity
    │
    ├─ initialize prepared VOF state
    │      ├─ validate_no_solid_cells(candidate_in.solid_phi)
    │      └─ initialize_vof_fields(
    │             candidate_in.density,
    │             candidate_in.vof,
    │             prepared_phi,
    │             prepared_type,
    │         )
    │
    ├─ validate_initialized_vof_state(candidate_in)
    │
    ├─ candidate_out = candidate_in.clone()
    │
    └─ commit candidate_in / candidate_out
```

P1 关键符号总表：

```text
VofGridState.__init__
VofGridState.clear
VofGridState.copy_to
VofGridState.clone

classify_initial_phi
validate_initial_topology
prepare_initial_vof
validate_no_solid_cells
initialize_vof_mass_kernel
initialize_vof_fields

validate_empty_vof_state
validate_initialized_vof_state

LbmSolver.initialize_vof_state
LbmDomain.initialize_vof
LbmSolver.step                    # authoritative VOF fail-fast
```

## 9. 不变量

### 9.1 空状态

构造和 clear 后：

```text
mass 全部有限且为 0
phi 全部有限且为 0
cell_type 全部为 GAS
```

空状态是合法状态，不需要 `initialized=False`。

### 9.2 初始化状态

通用：

```text
数组 shape/device/dtype 正确
density 在所有 GAS/INTERFACE/LIQUID cell 上均 finite 且 > 0
mass 有限且 mass >= 0
phi 有限
cell_type 只有 GAS/INTERFACE/LIQUID
```

按类型：

```text
GAS:
    mass == 0
    phi == 0

INTERFACE:
    density finite and > 0
    0 < phi < 1
    mass ≈ density * phi

LIQUID:
    density finite and > 0
    phi == 1
    mass ≈ density
```

拓扑：

```text
所有 D3Q19 非零格链上不存在 LIQUID-GAS 直接邻接
periodic axis 跨 seam 同样检查
domain 中不存在 solid_phi < 0
不检查 GAS 连通性或是否接触外边界
```

双缓冲：

```text
state_in 与 state_out 的 kinetic/VOF 数值相同
mass/phi/cell_type storage 均不共享
```

## 10. 测试计划

建议单一入口：

```bash
uv run python -m unittest newton.tests.test_lbm_vof_p1 -v
```

### 10.1 配置与分配

```text
[ ] interface_model="vof" 可以构造 model/domain
[ ] vof + debug_vof_observation 被拒绝
[ ] vof + non-zero G 被拒绝
[ ] off/shan_chen 的 P0 分配行为不变
[ ] VOF FullF/Home 都分配 VofGridState
[ ] VOF requires_grad 被拒绝
```

### 10.2 状态生命周期

```text
[ ] 构造后是 canonical all-GAS
[ ] clear 后恢复 canonical all-GAS
[ ] copy_to 只复制三个数组
[ ] clone 数值相同
[ ] clone 的三个数组全部不共享
[ ] VOF state 不能复制到无 VOF storage 的 state
```

### 10.3 LBM 初始化边界

```text
[ ] FullF initialize_equilibrium 后 state.density == rho0
[ ] HOME initialize_equilibrium 后 state.density == rho0
[ ] VOF 初始化只读取 state.density
[ ] 修改 API rho0 后仍以实际 state.density 计算 mass
```

最后一项应通过构造一个测试替身或可控的 initialized density 验证，不能仅因为当前
`state.density.fill_(rho0)` 而让错误实现也通过。

### 10.4 phi0 输入

```text
[ ] 合法 shape 和 dtype
[ ] phi0=0 -> GAS/mass0/phi0
[ ] phi0=1 -> LIQUID/mass=density/phi1
[ ] 0<phi0<1 -> INTERFACE/mass=density*phi
[ ] NaN/Inf 被拒绝
[ ] phi0<0 或 phi0>1 被拒绝
[ ] shape mismatch 被拒绝
[ ] 任意 solid cell 被拒绝，与该 cell 的 phi0 无关
```

### 10.5 topology

```text
[ ] LIQUID-INTERFACE-GAS 合法
[ ] face-direction L-G 被拒绝
[ ] edge-direction L-G 被拒绝
[ ] periodic seam L-G 被拒绝
[ ] 错误包含 count 和首个 cell/方向
[ ] 多个不连通 GAS 区域不触发额外分析
[ ] 全液体、无 GAS 的初始化合法
```

D3Q19 不包含三轴 corner link；P1 不应使用 D3Q27 的 26 邻居规则替代项目格链。

### 10.6 双缓冲和门禁

```text
[ ] domain.initialize_vof 初始化两个 buffer
[ ] 两个 buffer 数值相同且 storage 独立
[ ] domain.step() 在任何状态写入前 fail-fast
[ ] step 失败后两个 buffer 未交换
[ ] step 失败后 VOF 和 kinetic 数组未改变
```

### 10.7 P0 回归

必须重新运行 [07-p0-baseline.md](07-p0-baseline.md) 中固定的：

```text
39 个核心配置/碰撞/力/streaming/观察测试
6 个 directional streaming 测试
P1 变更文件的 Ruff F/I 检查
```

P1 不能通过修改 P0 测试期望来消除回归。

## 11. 提交拆分

### P1-1：authoritative state

```text
新增 VofGridState 三数组
接入 LbmStateBase allocate/clear/copy/clone
增加状态生命周期测试
不允许构造 VOF model 的旧 fail-fast 暂时保留或同提交移动到 step
```

### P1-2：mode construction and step guard

```text
允许 interface_model="vof" 构造 model/domain
收紧 debug_vof_observation 规则
requires_grad fail-fast
将 authoritative VOF fail-fast 移到 solver.step 开头
增加状态无修改测试
```

### P1-3：initialization

```text
新增 initialization.py
实现 phi0 validation/classification/topology/scope gate
使用实际 state.density 构造 mass/phi/type
增加 solid、GAS 连通性不分析、L-G 拒绝和 density 权威测试
```

### P1-4：domain API and double buffer

```text
新增 LbmDomain.initialize_vof
先初始化 LBM，再初始化 VOF
clone state_in 到 state_out
验证数值相同和 storage 独立
```

### P1-5：validation and acceptance

```text
新增/固化 invariant validation
运行 P1 测试和完整 P0 基线
更新 capability status
记录代码 SHA、环境、命令、结果和已知限制
```

每个提交只解决一个可判定问题，不在 P1 合入 population provider 或质量平流。

## 12. P1 退出门禁

```text
[ ] VofGridState 只有 mass/phi/cell_type
[ ] off/shan_chen/vof 分配矩阵正确
[ ] FullF/Home 都通过同一 state.density 初始化契约
[ ] mass 使用实际初始化后的 density
[ ] phi0 立即通过 mass/density 反算并规范化
[ ] GAS/INTERFACE/LIQUID 不变量通过
[ ] D3Q19 L-G 邻接和 periodic seam 检查通过
[ ] 任意 solid domain 在初始化期 fail-fast
[ ] GAS 连通性和气泡压力不在 P1 中推断
[ ] 首次/重复初始化都使用成功后提交的 candidate buffer
[ ] state_in/state_out 数值相同、storage 独立
[ ] clear/copy/clone 只处理三个数组
[ ] requires_grad 明确 fail-fast
[ ] domain.step 在状态修改前 fail-fast
[ ] P1 针对性测试全部通过
[ ] P0 CPU 基线全部通过
[ ] capability status 与 P1 冻结记录已更新
```

## 13. P1 完成后允许的声明

可以声明：

```text
authoritative VOF model 可以构造
正式 mass/phi/cell_type 可以分配
FullF/Home 统一通过 state.density 初始化 VOF
phi0、mass、type 和拓扑可以验证
双缓冲 VOF 状态独立且一致
未实现时间推进不会被误启用
P1 CPU 基线可重复
```

不能声明：

```text
VOF 质量守恒时间推进
界面能够移动
gas-to-interface population 已补全
类型转换和重分配正确
新运行时 interface kinetic 已实现
normal/PLIC/curvature/surface tension 已实现
CUDA 已验收
```

## 14. 后续阶段交接

P2 可以依赖且只能依赖以下 P1 产物：

```text
state.vof.mass
state.vof.phi
state.vof.cell_type
state.density
已验证的 D3Q19 初始拓扑
```

P2 第一次需要 `f_i^n(x)` 时，才增加 FullF/HOME logical population 的薄适配，并
复用现有 HOME reconstruction。P2 不应反向扩展 P1 状态或重新引入 density/moment
副本。

## 15. 实施验收记录

| 项目 | 结果 |
|---|---|
| 验收日期 | 2026-07-30 |
| 基础 HEAD | `981d944158e02c7ad70a773bd212123e427b6ce5` |
| 工作区状态 | P1 已实现并通过 CPU 测试，尚未提交冻结 |
| P1 针对性测试 | 18 项通过 |
| P0 冻结回归 | 39 项核心 + 6 项 directional 全部通过 |
| Ruff | P1 变更 Python 文件 `F/I` 检查通过 |
| CUDA | 未验收 |

P1 针对性命令：

```bash
WARP_CACHE_PATH=/tmp/wanphys-warp-cache PYTHONPATH=. \
uv run python -m unittest newton.tests.test_lbm_vof_p1 -v
```

P0 回归命令继续使用 [07-p0-baseline.md](07-p0-baseline.md) 的 39 项固定集合，并追加：

```bash
uv run python -m unittest newton.tests.test_lbm_directional_streaming -v
```

当前状态可以声明 `IMPLEMENTED_CPU_VERIFIED`，但只有在形成提交并记录对应代码 SHA
之后才能升级为冻结的 `CPU_ACCEPTED`。
